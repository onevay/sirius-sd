"""Бенчмарки частей пайплайна на этом ПК: скорость и качество поз-моделей, сравнение трекеров.

Качество позы считаем относительно «учителя» (YOLO26x-pose на повышенном разрешении) на ОДНИХ И ТЕХ ЖЕ кадрах — это не разметка людьми,
а мера согласия малой модели с сильной (честно называем «согласие с учителем»). Метрики по людям >= 80 px (граница регламента):
  * recall людей (IoU рамок >= 0.5), лишние детекции на кадр;
  * ошибка ключевых точек в ширинах плеч учителя (нос, запястья, локти) и доля точек с ошибкой <= 0.35 ширины плеч — это ровно масштаб
    порога th_in из регламента, поэтому показывает, годится ли точка для автомата циклов;
  * мс/кадр (медиана, p95) без прогрева.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .evaluate import iou
from .models import load_registry, local_weights
from .paths import OUTPUTS
from .pose_track import ensure_openvino, imgsz_hw

NOSE, LSHO, RSHO, LELB, RELB, LWRI, RWRI = 0, 5, 6, 7, 8, 9, 10


def make_runner(spec: dict, frame_hw: tuple[int, int] = (1080, 1920)):
    """spec: name, backend(ultralytics|rtmlib), weights(id), runtime(torch|openvino), device, imgsz -> callable(img)->(boxes,scores,kps)."""
    if spec["backend"] == "rtmlib":
        from rtmlib import Body

        mode = (load_registry()[spec["weights"]].rtmlib or {}).get("mode", "balanced")
        body = Body(mode=mode, backend="onnxruntime", device="cpu")

        def run(img):
            b = np.asarray(body.det_model(img))
            if len(b) == 0:
                return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 17, 3))
            k, s = body.pose_model(img, bboxes=b)
            return b[:, :4], np.full(len(b), 0.9), np.concatenate([k, s[..., None]], -1)

        return run
    from ultralytics import YOLO, settings

    from .pose_track import enable_openvino_cache

    settings.update({"sync": False})
    w = local_weights(spec["weights"])
    if spec["runtime"] == "openvino":
        enable_openvino_cache()
        hw = imgsz_hw(spec["imgsz"], frame_hw)
        model, sz, dev = YOLO(str(ensure_openvino(w, hw)), task="pose"), list(hw), spec["device"]
    else:
        model, sz, dev = YOLO(str(w), task="pose"), spec["imgsz"], "cpu"

    ref_sz = spec.get("refine")  # второй проход позы: int — YOLO по кропу (imgsz кропа); "rtmpose-s|m" — RTMPose (top-down) по рамкам
    rtm = None
    if isinstance(ref_sz, str) and ref_sz.startswith("rtmpose"):
        from rtmlib import Body

        rtm = Body(mode="lightweight" if ref_sz.endswith("s") else "balanced", backend="onnxruntime", device="cpu").pose_model
        ref_sz = None

    def run(img):
        r = model.predict(img, imgsz=sz, conf=0.25, iou=0.7, classes=[0], device=dev, verbose=False, max_det=30)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 17, 3))
        kxy = r.keypoints.xy.cpu().numpy()
        kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kxy.shape[:2])
        boxes, scores = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        kp = np.concatenate([kxy, kc[..., None]], -1)
        if rtm is not None:
            k2, s2 = rtm(img, bboxes=boxes)
            kp = np.concatenate([k2, s2[..., None]], -1)
        if ref_sz:
            H, W = img.shape[:2]
            for j, (x1, y1, x2, y2) in enumerate(boxes):
                h = y2 - y1
                if h > 300 or h < 20:
                    continue
                pw, ph = 0.25 * (x2 - x1), 0.25 * h
                cx1, cy1, cx2, cy2 = max(0, int(x1 - pw)), max(0, int(y1 - ph)), min(W, int(x2 + pw)), min(H, int(y2 + ph))
                crop = img[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue
                rr = model.predict(crop, imgsz=ref_sz, conf=0.1, classes=[0], device=dev, verbose=False, max_det=5)[0]
                if rr.boxes is None or len(rr.boxes) == 0:
                    continue
                bx = rr.boxes.xyxy.cpu().numpy()
                c0 = np.array([(x1 + x2) / 2 - cx1, (y1 + y2) / 2 - cy1])
                k = int(np.argmin(np.hypot((bx[:, 0] + bx[:, 2]) / 2 - c0[0], (bx[:, 1] + bx[:, 3]) / 2 - c0[1])))
                kk = rr.keypoints.xy.cpu().numpy()[k] + np.array([cx1, cy1], np.float32)
                cc = rr.keypoints.conf.cpu().numpy()[k] if rr.keypoints.conf is not None else np.ones(len(kk))
                kp[j] = np.concatenate([kk, cc[:, None]], -1)
        return boxes, scores, kp

    return run


def _match(tb: np.ndarray, cb: np.ndarray, thr: float = 0.5) -> list[tuple[int, int]]:
    if len(tb) == 0 or len(cb) == 0:
        return []
    m = np.array([[iou(tuple(t), tuple(c)) for c in cb] for t in tb])
    r, c = linear_sum_assignment(-m)
    return [(int(i), int(j)) for i, j in zip(r, c) if m[i, j] >= thr]


def teacher_cache(frames: list[Path], spec: dict, cache: Path) -> dict:
    """Предсказания «учителя» на кадрах (кэш в npz — учитель дорогой)."""
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["data"].item()
    run = make_runner(spec)
    data = {}
    for p in frames:
        img = cv2.imread(str(p))
        data[p.name] = run(img)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, data=np.array(data, dtype=object))
    return data


def compare_to_teacher(pred, teacher, min_h: float = 80.0, score_min: float = 0.4, conf_vis: float = 0.5) -> dict:
    tb, ts, tk = teacher
    keep = ((tb[:, 3] - tb[:, 1]) >= min_h) & (ts >= score_min) if len(tb) else np.zeros(0, bool)
    tb, tk = tb[keep], tk[keep]
    cb, cs, ck = pred
    pairs = _match(tb, cb)
    errs = {"nose": [], "wrist": [], "elbow": []}
    wrist_conf, vis_total, vis_ok = [], 0, 0
    for ti, ci in pairs:
        sh = np.hypot(*(tk[ti, LSHO, :2] - tk[ti, RSHO, :2])) if tk[ti, LSHO, 2] > conf_vis and tk[ti, RSHO, 2] > conf_vis else 0.25 * (tb[ti, 3] - tb[ti, 1])
        sh = max(sh, 0.15 * (tb[ti, 3] - tb[ti, 1]))
        for name, idxs in (("nose", [NOSE]), ("wrist", [LWRI, RWRI]), ("elbow", [LELB, RELB])):
            for k in idxs:
                if tk[ti, k, 2] >= conf_vis:
                    e = float(np.hypot(*(ck[ci, k, :2] - tk[ti, k, :2])) / sh)
                    errs[name].append(e)
                    if name == "wrist":
                        vis_total += 1
                        vis_ok += int(e <= 0.35 and ck[ci, k, 2] >= 0.3)
                        wrist_conf.append(float(ck[ci, k, 2]))
    return dict(t_persons=len(tb), matched=len(pairs), extra=max(len(cb) - len(pairs), 0), errs=errs, wrist_conf=wrist_conf, vis_total=vis_total, vis_ok=vis_ok)


def bench_pose(frames: list[Path], specs: list[dict], teacher_spec: dict, out_csv: Path, progress=None,
               scales: tuple[float, ...] = (1.0, 0.5, 0.33), min_h_px: float = 80.0) -> pd.DataFrame:
    """Для каждой конфигурации и масштаба кадра (имитация удаления: человек в `scale` раз мельче) считаем скорость и согласие с учителем.

    Учитель считается один раз на исходных кадрах. На масштабе s оцениваются люди, чья высота ПОСЛЕ уменьшения >= min_h_px
    (т.е. высота учителя >= min_h_px/s) — это люди «в зоне регламента» (>= 80 px); предсказания возвращаются в исходные координаты.
    """
    teacher = teacher_cache(frames, teacher_spec, OUTPUTS / "analysis" / "teacher.npz")
    rows = []
    imgs = {p.name: cv2.imread(str(p)) for p in frames}
    for si, spec in enumerate(specs):
        t_load = time.perf_counter()
        try:
            run = make_runner(spec, imgs[frames[0].name].shape[:2])
            run(imgs[frames[0].name])  # прогрев (в т.ч. компиляция OpenVINO)
            run(imgs[frames[0].name])
        except Exception as e:
            rows.append(dict(name=spec["name"], error=f"{type(e).__name__}: {str(e)[:120]}"))
            continue
        load_s = time.perf_counter() - t_load
        for sc in scales:
            ms, agg = [], dict(t_persons=0, matched=0, extra=0, vis_total=0, vis_ok=0)
            errs = {"nose": [], "wrist": [], "elbow": []}
            wconf = []
            for p in frames:
                img = imgs[p.name] if sc == 1.0 else cv2.resize(imgs[p.name], None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                t0 = time.perf_counter()
                b, s_, k = run(img)
                ms.append((time.perf_counter() - t0) * 1000)
                if sc != 1.0 and len(b):                      # назад в исходные координаты
                    b, k = b / sc, k.copy()
                    k[..., :2] /= sc
                c = compare_to_teacher((b, s_, k), teacher[p.name], min_h=min_h_px / sc)
                for key in agg:
                    agg[key] += c[key]
                for key in errs:
                    errs[key] += c["errs"][key]
                wconf += c["wrist_conf"]
            rows.append(dict(name=spec["name"], backend=spec["backend"], weights=spec["weights"], runtime=spec.get("runtime", "onnx"),
                             device=spec.get("device", "cpu"), imgsz=spec.get("imgsz", ""), scale=sc, frames=len(frames),
                             ms_median=round(float(np.median(ms)), 1), ms_p95=round(float(np.percentile(ms, 95)), 1), fps=round(1000 / float(np.median(ms)), 2),
                             load_warmup_s=round(load_s, 1),
                             person_recall=round(agg["matched"] / max(agg["t_persons"], 1), 3), teacher_persons=agg["t_persons"],
                             extra_per_frame=round(agg["extra"] / len(frames), 2),
                             nose_err_med=round(float(np.median(errs["nose"])), 3) if errs["nose"] else None,
                             wrist_err_med=round(float(np.median(errs["wrist"])), 3) if errs["wrist"] else None,
                             wrist_ok35=round(agg["vis_ok"] / max(agg["vis_total"], 1), 3),
                             wrist_conf_mean=round(float(np.mean(wconf)), 3) if wconf else None,
                             elbow_err_med=round(float(np.median(errs["elbow"])), 3) if errs["elbow"] else None))
        pd.DataFrame(rows).to_csv(out_csv, index=False)  # инкрементально: можно смотреть по ходу
        if progress:
            progress(si + 1, len(specs))
    return pd.DataFrame(rows)


def _short(nm: str) -> str:
    return (nm.replace("openvino-gpu", "iGPU").replace("openvino-cpu", "OV-CPU").replace("+refrtmpose-s", "+RTMPose-s").replace("+refrtmpose-m", "+RTMPose-m")
              .replace(" (YOLOX-tiny+RTMPose-s)", "").replace(" (YOLOX-m+RTMPose-m)", "").replace(" torch", " CPU"))


def plot_pose_bench(df: pd.DataFrame, out_png: Path) -> None:
    """Три панели: скорость vs точность запястий (подписан Парето-фронт), скорость vs recall людей, точность запястий при «удалении»."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    d_all = df.dropna(subset=["fps"]).copy()
    d = d_all[d_all.scale == 1.0].copy() if "scale" in d_all else d_all.copy()
    d["short"] = d["name"].map(_short)
    d["hyb"] = d["name"].str.contains(r"\+ref", regex=True)

    def grp(r) -> str:
        if r.backend == "rtmlib":
            return "rtmlib (ONNX CPU)"
        if "gpu" in str(r.device):
            return "OpenVINO iGPU"
        if r.runtime == "openvino":
            return "OpenVINO CPU"
        return "torch CPU"

    colors = {"torch CPU": "#1f77b4", "OpenVINO CPU": "#ff7f0e", "OpenVINO iGPU": "#d62728", "rtmlib (ONNX CPU)": "#2ca02c"}
    d["grp"] = d.apply(grp, axis=1)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))

    def scatter(a, ycol):
        for _, r in d.iterrows():
            a.scatter(r.fps, r[ycol], color=colors[r.grp], marker="*" if r.hyb else "o", s=170 if r.hyb else 60, edgecolor="k", linewidth=0.5, zorder=3)
        a.set_xscale("log")
        a.grid(alpha=0.3)
        a.set_xlabel("кадров/с, только инференс (медиана, 20 кадров)")

    # A: Парето-фронт по (fps, wrist_ok35)
    scatter(ax[0], "wrist_ok35")
    front, best = [], -1.0
    for _, r in d.sort_values("fps", ascending=False).iterrows():
        if r.wrist_ok35 > best:
            front.append(r.name)
            best = r.wrist_ok35
    pf = d.loc[front].sort_values("fps")
    ax[0].plot(pf.fps, pf.wrist_ok35, color="k", lw=0.8, ls="--", zorder=2)
    for i, (_, r) in enumerate(pf.iterrows()):
        ax[0].annotate(r.short, (r.fps, r.wrist_ok35), fontsize=7.5, xytext=(6, 6 if i % 2 == 0 else -12), textcoords="offset points")
    ax[0].set_ylabel("запястья: доля с ошибкой ≤ 0.35 ширины плеч (согласие с учителем)")
    ax[0].set_title("Скорость vs точность запястий (подписан Парето-фронт)")
    # B: recall людей
    scatter(ax[1], "person_recall")
    for _, r in d[d.person_recall < 0.95].iterrows():
        ax[1].annotate(r.short, (r.fps, r.person_recall), fontsize=7, xytext=(5, 3), textcoords="offset points")
    ax[1].set_ylabel("recall людей ≥ 80 px (по учителю)")
    ax[1].set_title("Скорость vs recall людей (подписаны только recall < 0.95)")
    # C: «удаление»
    pick = ["yolo26n@960+refrtmpose-s openvino-gpu", "yolo26n@960 openvino-gpu", "yolo26s@960 openvino-gpu", "yolo26n@640 torch", "yolo26n@960 torch",
            "rtmlib lightweight (YOLOX-tiny+RTMPose-s)"]
    for nm in pick:
        g = d_all[d_all["name"] == nm].sort_values("scale")
        if len(g):
            ax[2].plot(g.scale, g.wrist_ok35, marker="o", lw=1.6, label=_short(nm))
    ax[2].invert_xaxis()
    ax[2].set_xlabel("масштаб кадра (1.0 = как есть; 0.33 = человек в 3 раза дальше)\nна масштабе s оцениваются люди, оставшиеся ≥ 80 px")
    ax[2].set_ylabel("запястья: доля с ошибкой ≤ 0.35 ширины плеч")
    ax[2].set_title("Точность запястий при «удалении» людей")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    handles = [Line2D([], [], marker="o", ls="", color=c, label=k, markeredgecolor="k") for k, c in colors.items()]
    handles.append(Line2D([], [], marker="*", ls="", color="gray", markersize=12, label="гибрид YOLO + RTMPose (звезда)"))
    ax[0].legend(handles=handles, fontsize=8, loc="lower right")
    fig.suptitle("Модели позы на ноутбуке i3-1115G4 (без NVIDIA): 20 кадров из ваших видео, учитель — YOLO26x@1280", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=125)
    plt.close(fig)


def replay_tracking(raw: pd.DataFrame, frame_t: pd.DataFrame, cfg: dict, tracker: str = "botsort", buffer_sec: float = 3.0,
                    match_thresh: float | None = None, new_track_thresh: float | None = None) -> pd.DataFrame:
    """Переигрывание трекера по сохранённым сырым детекциям (без повторного инференса позы). Возвращает таблицу треков."""
    import yaml
    import ultralytics
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import IterableSimpleNamespace

    from .pose_track import _Dets

    fn = {"botsort": "botsort", "bytetrack": "bytetrack", "ocsort": "ocsort", "fasttrack": "fasttrack", "tracktrack": "tracktrack", "deepocsort": "deepocsort"}[tracker]
    base = yaml.safe_load((Path(ultralytics.__file__).parent / "cfg" / "trackers" / f"{fn}.yaml").read_text(encoding="utf-8"))
    ft = frame_t.sort_values("frame")
    proc_fps = 1.0 / max(float(np.median(np.diff(ft.t.values))), 1e-3) if len(ft) > 2 else 10.0
    tr = cfg["tracking"]
    base.update(track_buffer=max(5, int(round(buffer_sec * proc_fps))), match_thresh=match_thresh or tr["match_thresh"],
                new_track_thresh=new_track_thresh or tr["new_track_thresh"], track_low_thresh=0.1, track_high_thresh=min(cfg["pose"]["conf"], 0.25))
    if "gmc_method" in base:
        base["gmc_method"] = "none"
    if "with_reid" in base:
        base["with_reid"] = False
    ns = IterableSimpleNamespace(**base)
    ns.device = "cpu"
    trk = TRACKER_MAP[base["tracker_type"]](args=ns)
    by = {f: g for f, g in raw.groupby("frame")}
    rows = []
    for f, t in zip(ft.frame.values, ft.t.values):
        g = by.get(f)
        det = _Dets(g[["x1", "y1", "x2", "y2"]].values, g.conf.values) if g is not None else _Dets(np.zeros((0, 4)), np.zeros(0))
        for r in trk.update(det, None):
            rows.append((int(f), float(t), int(r[4]), *[float(v) for v in r[:4]], float(r[5]), float(r[3] - r[1])))
    return pd.DataFrame(rows, columns=["frame", "t", "tid", "x1", "y1", "x2", "y2", "score", "h"])


def track_metrics(df: pd.DataFrame, raw: pd.DataFrame, frame_t: pd.DataFrame, cfg: dict) -> dict:
    """Прокси-метрики трекинга без разметки (идеал «треков на человека» = 1.0, дубликатов 0, склеек 0)."""
    from .tracks import Tracks, stitch_tracks

    if df.empty:
        return dict(tracks_raw=0, tracks=0, n_people=0, tracks_per_person=None, stitched=0, dup_frames=0, median_len_s=None, coverage=None)
    big = raw[(raw.conf >= 0.4) & ((raw.y2 - raw.y1) >= cfg["video"]["min_person_height_px"])]
    n_people = float(big.groupby("frame").size().quantile(0.95)) if len(big) else 1.0   # устойчивая оценка «сколько людей одновременно»
    tr = Tracks(df, np.zeros((len(df), 17, 3), np.float32), frame_t, {})
    st = stitch_tracks(tr, cfg["tracking"])
    s = st.summary(cfg["video"]["min_person_height_px"])
    dup = 0
    for _, g in df.groupby("frame"):
        b = g[["x1", "y1", "x2", "y2"]].values
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                dup += iou(tuple(b[i]), tuple(b[j])) > 0.6
    return dict(tracks_raw=int(df.tid.nunique()), tracks=int(len(s)), n_people=n_people, tracks_per_person=round(len(s) / max(n_people, 1.0), 2),
                stitched=len(st.meta.get("stitch", {}).get("merged", {})), dup_frames=int(dup),
                median_len_s=round(float(s.dur.median()), 1) if len(s) else None, coverage=round(float(s.coverage.mean()), 2) if len(s) else None)
