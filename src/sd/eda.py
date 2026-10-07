"""EDA датасета (руководство §3.1): что показывают данные, прежде чем выбирать модели и пороги.

Равномерная выборка кадров по каждому видео -> позa-модель (она же детектор людей) -> высоты людей в px, уверенности ключевых точек
(нос, запястья, локти, плечи) по размерам, число людей, яркость и засветка. Итог — parquet-таблицы и графики для docs/ANALYSIS.md.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .models import local_weights
from .paths import OUTPUTS, video_id, weak_label
from .video_io import choose_stride, iter_frames, probe

KP_NAMES = ["nose", "leye", "reye", "lear", "rear", "lsho", "rsho", "lelb", "relb", "lwri", "rwri"]
HEIGHT_BINS = [0, 60, 80, 120, 160, 250, 400, 10_000]
HEIGHT_LABELS = ["<60", "60-80", "80-120", "120-160", "160-250", "250-400", ">400"]


def run_eda(videos: list[Path], cfg: dict, model_id: str = "yolo26n-pose", imgsz: int = 960, every_sec: float = 2.0,
            max_samples: int = 250, conf: float = 0.25, save_frames: Path | None = None, progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    model = YOLO(str(local_weights(model_id)), task="pose")
    frames_rows, person_rows = [], []
    t_all = time.perf_counter()
    for vi, vp in enumerate(videos):
        info = probe(vp)
        interval = max(every_sec, info.duration / max_samples)
        stride = choose_stride(info.fps, 1.0 / interval)
        vid, lab = video_id(vp), weak_label(vp) or "-"
        best = []  # кандидаты кадров для набора бенчмарка: (n_persons, t, img)
        for fr in iter_frames(vp, 0.0, None, stride):
            g = cv2.cvtColor(fr.img, cv2.COLOR_BGR2GRAY)
            r = model.predict(fr.img, imgsz=imgsz, conf=conf, classes=[0], verbose=False, device="cpu")[0]
            n = 0 if r.boxes is None else len(r.boxes)
            frames_rows.append(dict(video=vid, folder=lab, t=fr.t, n_persons=n, brightness=float(g.mean()),
                                    sat_frac=float((g >= 250).mean()), dark_frac=float((g <= 20).mean())))
            if n:
                xyxy = r.boxes.xyxy.cpu().numpy()
                sc = r.boxes.conf.cpu().numpy()
                kxy = r.keypoints.xy.cpu().numpy()
                kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kxy.shape[:2])
                for j in range(n):
                    h, w = xyxy[j, 3] - xyxy[j, 1], xyxy[j, 2] - xyxy[j, 0]
                    row = dict(video=vid, folder=lab, t=fr.t, h=float(h), w=float(w), score=float(sc[j]),
                               edge=bool(xyxy[j, 0] < 3 or xyxy[j, 1] < 3 or xyxy[j, 2] > info.width - 3 or xyxy[j, 3] > info.height - 3))
                    for k, nm in enumerate(KP_NAMES):
                        row[f"c_{nm}"] = float(kc[j, k])
                    row["kp_vis"] = int((kc[j] >= 0.3).sum())
                    person_rows.append(row)
                if save_frames is not None:
                    hs = xyxy[:, 3] - xyxy[:, 1]
                    best.append((n, float(np.median(hs)), fr.t, fr.img.copy()))
                    best = sorted(best, key=lambda b: (-b[0], -b[1]))[:3]
            if progress:
                progress(vi, len(videos), vid, fr.t / max(info.duration, 1))
        if save_frames is not None:
            save_frames.mkdir(parents=True, exist_ok=True)
            for n, mh, t, img in best:
                cv2.imwrite(str(save_frames / f"{vid}__{t:07.2f}s__n{n}_h{mh:.0f}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    fr_df, pr_df = pd.DataFrame(frames_rows), pd.DataFrame(person_rows)
    fr_df.attrs["wall_sec"] = time.perf_counter() - t_all
    return fr_df, pr_df


def pick_windows(frames: pd.DataFrame, persons: pd.DataFrame, length: float = 30.0, min_h: float = 60.0) -> pd.DataFrame:
    """Для каждого видео — окно длиной `length` с, где больше всего людей (по выборке EDA). Короткие видео берутся целиком."""
    ok = persons[persons.h >= min_h].groupby(["video", "t"]).size().rename("n_ok").reset_index()
    f = frames.merge(ok, on=["video", "t"], how="left").fillna({"n_ok": 0})
    rows = []
    for vid, g in f.groupby("video"):
        g = g.sort_values("t")
        dur_est = float(g.t.max())
        if dur_est <= length * 1.15:
            rows.append(dict(video=vid, start=0.0, end=None, score=float(g.n_ok.sum()), note="целиком"))
            continue
        best, best_s = 0.0, -1.0
        for t0 in g.t.values:
            s = g[(g.t >= t0) & (g.t < t0 + length)].n_ok.sum()
            if s > best_s:
                best, best_s = float(t0), float(s)
        rows.append(dict(video=vid, start=round(best, 1), end=round(best + length, 1), score=best_s, note="окно с максимумом людей"))
    return pd.DataFrame(rows)


def summarize(frames: pd.DataFrame, persons: pd.DataFrame) -> dict:
    out: dict = {}
    if persons.empty:
        return out
    persons = persons.copy()
    persons["hbin"] = pd.cut(persons.h, HEIGHT_BINS, labels=HEIGHT_LABELS, right=False)
    out["n_frames"], out["n_persons"] = len(frames), len(persons)
    out["frames_with_person"] = float((frames.n_persons > 0).mean())
    out["height_quantiles"] = persons.h.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(0).to_dict()
    out["share_h_lt80"] = float((persons.h < 80).mean())
    out["share_h_lt150"] = float((persons.h < 150).mean())
    out["by_folder_median_h"] = persons.groupby("folder").h.median().round(0).to_dict()
    kc = [f"c_{k}" for k in ("nose", "lsho", "rsho", "lelb", "relb", "lwri", "rwri")]
    g = persons.groupby("hbin", observed=True)
    out["kp_conf_by_height"] = g[kc].mean().round(2).assign(n=g.size()).reset_index().to_dict("records")
    wr = persons[["c_lwri", "c_rwri"]].max(axis=1)
    out["wrist_vis_by_height"] = (wr >= 0.3).groupby(persons.hbin, observed=True).mean().round(2).to_dict()
    return out


def plots(frames: pd.DataFrame, persons: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    # 1) гистограмма высот людей
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.logspace(np.log10(15), np.log10(1500), 40)
    for lab, col in (("smoking", "#d62728"), ("fake", "#1f77b4")):
        h = persons[persons.folder == lab].h
        ax.hist(h, bins=bins, alpha=0.55, label=f"{'курение' if lab == 'smoking' else 'лжекурение'} (n={len(h)}, медиана {h.median():.0f} px)", color=col)
    ax.axvline(80, color="k", ls="--", lw=1)
    ax.text(82, ax.get_ylim()[1] * 0.92, "80 px — порог IGNORE", fontsize=8)
    ax.axvline(150, color="gray", ls=":", lw=1)
    ax.text(152, ax.get_ylim()[1] * 0.80, "150 px", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("высота человека в кадре, px (лог. шкала)")
    ax.set_ylabel("число людей в выборке")
    ax.set_title("Распределение размеров людей (выборка кадров)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "eda_person_height_hist.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    # 2) уверенность ключевых точек по размеру человека
    pp = persons.copy()
    pp["hbin"] = pd.cut(pp.h, HEIGHT_BINS, labels=HEIGHT_LABELS, right=False)
    g = pp.groupby("hbin", observed=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, col in (("nose", "#2ca02c"), ("lwri", "#d62728"), ("rwri", "#ff7f0e"), ("lelb", "#9467bd"), ("lsho", "#7f7f7f")):
        m = g[f"c_{k}"].mean()
        ax.plot(range(len(m)), m.values, marker="o", label=k, color=col)
    ax.set_xticks(range(len(g.size())))
    ax.set_xticklabels(g.size().index.astype(str))
    ax2 = ax.twinx()
    ax2.bar(range(len(g.size())), g.size().values, alpha=0.12, color="k")
    ax2.set_ylabel("людей в корзине")
    ax.set_xlabel("высота человека, px")
    ax.set_ylabel("средняя уверенность ключевой точки")
    ax.set_ylim(0, 1.0)
    ax.set_title("Качество ключевых точек падает на мелких людях")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = out_dir / "eda_kp_conf_vs_height.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    # 3) присутствие людей и яркость по видео
    v = frames.groupby("video").agg(present=("n_persons", lambda x: (x > 0).mean()), bright=("brightness", "mean"),
                                    sat=("sat_frac", "mean"), folder=("folder", "first")).reset_index()
    v = v.sort_values(["folder", "video"])
    fig, ax = plt.subplots(figsize=(9, 6))
    cols = ["#d62728" if f == "smoking" else "#1f77b4" for f in v.folder]
    ax.barh(range(len(v)), v.present.values, color=cols)
    ax.set_yticks(range(len(v)))
    ax.set_yticklabels([s.split("__")[-1][:26] for s in v.video], fontsize=7)
    ax.set_xlabel("доля выбранных кадров, где найден хотя бы один человек")
    ax.set_title("Присутствие людей по видео (красные — «курение», синие — «лжекурение»)")
    fig.tight_layout()
    p = out_dir / "eda_presence.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    return paths
