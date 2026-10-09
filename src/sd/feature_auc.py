"""Проверка признаков цикла на разметке глазами: насколько каждый отделяет «курение» от «не курение» (AUC) — руководство §3.4, группы D/E.

Разметка `docs/review/assistant_visual_labels.csv` — суждения ассистента по кадрам циклов, а НЕ эталон организаторов, и выборка мала (десятки
циклов, часть видео даёт по 5 циклов одного человека). Поэтому AUC здесь — грубая проверка «признак вообще несёт сигнал и в какую сторону»,
а доверительные интервалы считаются бутстрэпом по ВИДЕО, а не по циклам (циклы одного ролика зависимы). Решения по порогам на этих числах не принимаются.
"""
from __future__ import annotations

from . import _env  # noqa: F401

from pathlib import Path

import numpy as np
import pandas as pd

from .paths import OUTPUTS, ROOT

LABELS = ROOT / "docs" / "review" / "assistant_visual_labels.csv"
AN = OUTPUTS / "analysis"
POS = ("smoke",)
NEG = ("drink", "touch_face", "phone", "eat", "other_neg")
DATASET = OUTPUTS / "dataset" / "cycles.parquet"
KEY = ["video", "tid", "start"]
XCLIP_SMOKE = "p_a person smoking a cigarette"


def load_labelled(path: Path = LABELS) -> pd.DataFrame:
    """Размеченные циклы без `unsure`; y=1 для курения, 0 для питья/касания лица/телефона/прочего."""
    lab = pd.read_csv(path)
    lab = lab[lab.label.isin(POS + NEG)].copy()
    lab["y"] = lab.label.isin(POS).astype(int)
    return lab


def _asof_merge(lab: pd.DataFrame, cyc: pd.DataFrame, tol: float = 0.06) -> pd.DataFrame:
    """В CSV `start` округлён до 0.1 с, в таблице циклов — до 0.001: стыкуем по (video, tid) с ближайшим start."""
    parts = []
    for (v, t), g in lab.groupby(["video", "tid"]):
        c = cyc[(cyc.video == v) & (cyc.tid == t)].sort_values("start")
        if c.empty:
            continue
        m = pd.merge_asof(g.sort_values("start"), c.drop(columns=["video", "tid"]).rename(columns={"start": "start_cyc"}),
                          left_on="start", right_on="start_cyc", direction="nearest", tolerance=tol)
        m["video"], m["tid"] = v, t
        parts.append(m)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _merge_by_peak(lab: pd.DataFrame, cyc: pd.DataFrame, ds: pd.DataFrame, tol_t: float = 1.0, tol_px: float = 150.0) -> pd.DataFrame:
    """Метки проекта (`labels/cycle_labels.csv`: video, peak_t, cx, cy — не зависят от tid) -> циклы; допуски те же, что у `dataset.attach_labels`.

    Положение (cx, cy) берётся из таблицы датасета (`outputs/dataset/cycles.parquet`), в таблице циклов его нет.
    """
    c = cyc.merge(ds[KEY + ["cx", "cy"]].assign(start=ds.start.round(3)), on=KEY, how="left")   # ключ start в таблицах округляется до 0.001
    rows = []
    for r in c.itertuples():
        m = lab[(lab.video == r.video) & ((lab.peak_t - r.peak_t).abs() <= tol_t) & (np.hypot(lab.cx - r.cx, lab.cy - r.cy) <= tol_px)]
        if len(m):
            best = m.iloc[int(np.argmin((m.peak_t - r.peak_t).abs().values))]
            rows.append(dict(video=r.video, tid=r.tid, start=r.start, label=best.label, y=int(best.label in POS)))
    return pd.DataFrame(rows).merge(cyc, on=KEY, how="left") if rows else pd.DataFrame()


def feature_table(labels: Path = LABELS, an: Path = AN, dataset: Path = DATASET, vlm: Path | None = None, with_dataset: bool = False) -> pd.DataFrame:
    """Размеченные циклы + все признаки: кинематика, предмет (D), X-CLIP и VideoMAE (E). Недостающие файлы пропускаются.

    Понимает два формата меток: визуальные ассистента (`video, tid, start, label`) и проекта (`video, peak_t, cx, cy, label` — то, что пишет вкладка «Разметка»).
    `vlm` — parquet из `sd vlm-eval`: оценки `score_<режим>` становятся колонками `vlm_<режим>` (группа G); таблица сужается до циклов, которые VLM успел
    обработать, чтобы все группы сравнивались на одних и тех же циклах.
    """
    lab = load_labelled(labels)
    cyc = _cycles_with_durations(an)
    if "start" in lab.columns:
        # допуск 0.06 с — округление CSV; не нашедшие пару циклы остаются без признаков и видны в `n_matched`
        tab = _asof_merge(lab.drop(columns=["end", "peak_t", "hold", "d_min"], errors="ignore"), cyc)
        tab = tab.rename(columns={"start": "start_csv"}).rename(columns={"start_cyc": "start"})
    else:
        tab = _merge_by_peak(lab, cyc, pd.read_parquet(dataset))
        if tab.empty:
            return tab
    return _attach_features(tab, an, vlm, only_vlm=True, dataset=dataset if with_dataset else None)


def _cycles_with_durations(an: Path) -> pd.DataFrame:
    """Все циклы (`cycles_all.parquet`) + длительности. Колонка `run` остаётся (каталог запуска нужен vlm_eval и event_check), в признаки она не входит."""
    cyc = pd.read_parquet(an / "cycles_all.parquet")
    return cyc.assign(dur=cyc.end - cyc.start, mouth_dur=cyc.mouth_out - cyc.mouth_in).drop(columns=["end", "idx"], errors="ignore")


DS_SKIP = ("group", "weak_label", "mouth_in", "mouth_out", "end", "peak_t", "brightness", "cx", "cy")   # служебные колонки таблицы датасета — в признаки не берём


def _attach_dataset(tab: pd.DataFrame, dataset: Path | None) -> pd.DataFrame:
    """Колонки кинематики/ритма/качества наблюдения (группы A, B, C таблицы `outputs/dataset/cycles.parquet`), которых ещё нет в таблице, — по ключу цикла."""
    if dataset is None or not Path(dataset).exists():
        return tab
    ds = pd.read_parquet(dataset)
    ds["start"] = ds["start"].round(3)
    keep = KEY + [c for c in ds.columns if c not in KEY and c not in DS_SKIP and c not in tab.columns]
    return tab.assign(start=tab["start"].round(3)).merge(ds[keep], on=KEY, how="left")


def _attach_features(tab: pd.DataFrame, an: Path, vlm: Path | None, only_vlm: bool, dataset: Path | None = None) -> pd.DataFrame:
    """Присоединяет по ключу цикла предмет (D), X-CLIP и VideoMAE (E), VLM (G), позу (H), фото-модель (I) и производные признаки (+ кинематику A–C из таблицы датасета,
    если передан `dataset`). `only_vlm`: оставить только циклы, обработанные VLM."""
    tab = _attach_dataset(tab, dataset)
    for fn in ("evidence_cycles.parquet", "tube_xclip_cycles.parquet", "tube_videomae_cycles.parquet", "pose_cycles.parquet", "photo_cycles.parquet"):
        p = an / fn
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        drop = [c for c in df.columns if c in ("idx", "t0", "t1", "window", "n_det_frames") or c.endswith("_top1") or c.endswith("_top3")]
        df = df.drop(columns=drop, errors="ignore")
        df["start"] = df["start"].round(3)                       # ключ start в таблицах округляется до 1 мс
        tab = tab.assign(start=tab["start"].round(3)).merge(df, on=KEY, how="left", suffixes=("", "_dup"))
        tab = tab.drop(columns=[c for c in tab.columns if c.endswith("_dup")])
    if vlm is not None:
        g = pd.read_parquet(vlm)
        sc = {c: f"vlm_{c[len('score_'):]}" for c in g.columns if c.startswith("score_")}
        tab = tab.merge(g[g.ok.astype(bool)][KEY + list(sc)].rename(columns=sc), on=KEY, how="inner" if only_vlm else "left")
    return _derived(tab)


def all_cycles_features(an: Path = AN, vlm: Path | None = None, dataset: Path | None = None) -> pd.DataFrame:
    """ВСЕ циклы (а не только размеченные) с теми же признаками — для оценки сборки событий; без меток. `dataset` — добавить кинематику A–C."""
    return _attach_features(_cycles_with_durations(an), an, vlm, only_vlm=False, dataset=dataset)


def with_detectors(t: pd.DataFrame, detectors: list[str]) -> pd.DataFrame:
    """Копия таблицы, где `obj_any_*` — максимум только по перечисленным детекторам (остальные колонки предмета остаются): классификатор для профиля с меньшим числом детекторов
    обучается на тех же признаках, что получит в работе."""
    t = t.copy()
    for suffix in ("max_conf", "hit_frames"):
        cols = [f"obj_{d}_{suffix}" for d in detectors]
        miss = [c for c in cols if c not in t.columns]
        if miss:
            raise KeyError(f"в таблице нет колонок {miss}: детекторы не считались (`sd enrich`)")
        t[f"obj_any_{suffix}"] = t[cols].max(axis=1)
    hit = [f"obj_{d}_hit" for d in detectors if f"obj_{d}_hit" in t.columns]
    if hit:
        t["obj_any_hit"] = t[hit].astype(float).max(axis=1)
    return t


def _derived(t: pd.DataFrame) -> pd.DataFrame:
    """Производные признаки, которые попадут в вектор цикла: «любой детектор предмета», X-CLIP «курение против остального», VideoMAE smoking."""
    t = t.copy()
    mc = [c for c in t.columns if c.startswith("obj_") and c.endswith("_max_conf")]
    if mc:
        t["obj_any_max_conf"] = t[mc].max(axis=1)
        hf = [c for c in t.columns if c.startswith("obj_") and c.endswith("_hit_frames")]
        t["obj_any_hit_frames"] = t[hf].max(axis=1)
        hit = [c for c in t.columns if c.startswith("obj_") and c.endswith("_hit")]
        t["obj_any_hit"] = t[hit].astype(float).max(axis=1)
    xp = [c for c in t.columns if c.startswith("p_a person")]
    if XCLIP_SMOKE in t.columns:
        others = [c for c in xp if c != XCLIP_SMOKE]
        t["xclip_smoke_minus_max_other"] = t[XCLIP_SMOKE] - t[others].max(axis=1)
        eps = 1e-6
        t["xclip_smoke_logit"] = np.log((t[XCLIP_SMOKE] + eps) / (1 - t[XCLIP_SMOKE] + eps))
        vape = "p_a person vaping"
        if vape in t.columns:
            t["xclip_smoke_or_vape"] = t[XCLIP_SMOKE] + t[vape]
    if "videomae_k_smoking" in t.columns:
        rel = [c for c in t.columns if c.startswith("videomae_k_")]
        t["videomae_smoke_share"] = t["videomae_k_smoking"] / t[rel].sum(axis=1).clip(lower=1e-9)
        t["videomae_smoke_minus_drink"] = t["videomae_k_smoking"] - t[[c for c in rel if "drink" in c or "tasting" in c]].max(axis=1)
    return t


def feature_columns(t: pd.DataFrame) -> dict[str, list[str]]:
    """Группы признаков по руководству §3.4 (A/B/C — кинематика, D — предмет, E — внешний вид)."""
    kin = [c for c in ("hold", "d_min", "dur", "mouth_dur") if c in t.columns]
    obj = [c for c in t.columns if c.startswith("obj_")]
    xcl = [c for c in t.columns if c.startswith("p_a person") or c.startswith("xclip_")]
    vm = [c for c in t.columns if c.startswith("videomae_") and c not in ("videomae_top1_p",)] + (["videomae_top1_p"] if "videomae_top1_p" in t.columns else [])
    vl = [c for c in t.columns if c.startswith("vlm_")]
    from .pose_feats import POSE_COLS

    po = [c for c in POSE_COLS if c in t.columns]
    ph = [c for c in t.columns if c.startswith("photo_") and c != "photo_n_frames"]
    out = {"kinematics": kin, "object (D)": obj, "x-clip (E)": xcl, "videomae (E)": vm}
    if vl:
        out["vlm (G)"] = vl
    if po:
        out["pose (H)"] = po
    if ph:
        out["photo (I)"] = ph
    return out


def _auc(y: np.ndarray, x: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    ok = np.isfinite(x)
    y, x = y[ok], x[ok]
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return float("nan")
    return float(roc_auc_score(y, x))


def auc_by_video_bootstrap(t: pd.DataFrame, col: str, n_boot: int = 500, seed: int = 0) -> tuple[float, float, float]:
    """AUC признака и 95% интервал бутстрэпа с повторной выборкой видео."""
    y, x = t.y.values, t[col].astype(float).values
    point = _auc(y, x)
    vids = t.video.unique()
    by_v = {v: np.flatnonzero(t.video.values == v) for v in vids}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(vids, size=len(vids), replace=True)
        idx = np.concatenate([by_v[v] for v in pick])
        a = _auc(y[idx], x[idx])
        if np.isfinite(a):
            vals.append(a)
    if len(vals) < 20:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def per_feature_auc(t: pd.DataFrame, n_boot: int = 500) -> pd.DataFrame:
    """AUC каждого признака: курение против всех не-курений + против каждого вида ложного жеста отдельно (если ≥ 4 примеров)."""
    rows = []
    groups = feature_columns(t)
    neg_classes = [c for c in NEG if (t.label == c).sum() >= 4]
    for g, cols in groups.items():
        for c in cols:
            x = t[c].astype(float)
            if x.notna().sum() < 10:
                continue
            a, lo, hi = auc_by_video_bootstrap(t, c, n_boot)
            row = dict(group=g, feature=c, auc=a, lo=lo, hi=hi, n=int(x.notna().sum()), nonzero=float((x.fillna(0) != 0).mean()))
            for nc in neg_classes:
                s = t[t.label.isin(POS + (nc,))]
                row[f"auc_vs_{nc}"] = _auc(s.y.values, s[c].astype(float).values)
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df["signal"] = np.where(df.lo > 0.5, "+", np.where(df.hi < 0.5, "-", ""))   # интервал целиком по одну сторону от 0.5
        df = df.sort_values(["group", "auc"], ascending=[True, False]).reset_index(drop=True)
    return df


def folder_of(video: pd.Series) -> pd.Series:
    """Папка данных по `video_id` (`<папка>__<имя>`): значения из `paths.WEAK_LABEL_BY_FOLDER` (smoking / fake), иначе unknown."""
    from .paths import WEAK_LABEL_BY_FOLDER

    pref = {f"{k}__": v for k, v in WEAK_LABEL_BY_FOLDER.items()}
    return video.map(lambda v: next((lab for p, lab in pref.items() if str(v).startswith(p)), "unknown"))


def object_recall(t: pd.DataFrame) -> pd.DataFrame:
    """Доля циклов с «устойчиво виденным» предметом (≥ k из n кадров) по классам меток и по детекторам — это recall детектора на кропах (§3.4: < 50% → только бонус)."""
    cols = [c for c in t.columns if c.startswith("obj_") and c.endswith("_hit")]
    if not cols:
        return pd.DataFrame()
    g = t.assign(**{c: t[c].astype(float) for c in cols}).groupby("label")[cols].mean().T.rename(columns={"smoke": "smoke (recall)"})
    g["не курение (FP rate)"] = t[t.y == 0][cols].astype(float).mean()
    g.index.name = "detector"
    return g.reset_index()


def confound_check(t: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Не выучил ли признак СЦЕНУ вместо жеста. Все затяжки в данных из папки «курение», а большинство негативов — из «лжекурения» (другие камеры, люди, свет).

    * `auc_all` — как в основной таблице;
    * `auc_in_smoking_dir` — затяжки против негативов ТОЛЬКО из папки «курение» (одна и та же камера/сцены, но мало негативов);
    * `auc_dir_among_neg` — насколько признак различает папку среди одних негативов (оба класса — не курение!). 0.5 — не различает;
      заметно выше 0.5 — признак несёт сведения о сцене, и его AUC в основной таблице завышен.
    """
    d = t.assign(folder=folder_of(t.video))
    cols = cols or [c for g in feature_columns(d).values() for c in g]
    neg = d[d.y == 0]
    sm = d[d.folder == "smoking"]
    rows = []
    for c in cols:
        x = d[c].astype(float)
        if x.notna().sum() < 10:
            continue
        row = dict(feature=c, auc_all=_auc(d.y.values, x.values), n_pos_dir=int((sm.y == 1).sum()), n_neg_dir=int((sm.y == 0).sum()),
                   auc_in_smoking_dir=_auc(sm.y.values, sm[c].astype(float).values))
        row["auc_dir_among_neg"] = _auc((neg.folder == "smoking").astype(int).values, neg[c].astype(float).values)
        rows.append(row)
    return pd.DataFrame(rows)


def pick_columns(t: pd.DataFrame) -> dict[str, list[str]]:
    """По нескольку «главных» признаков на группу, выбранных ЗАРАНЕЕ (не по AUC на этих данных): на 60–70 циклах регрессия переобучается на десятках коррелирующих колонок."""
    pick = {
        "kinematics": [c for c in ("hold", "d_min", "mouth_dur") if c in t.columns],
        "object (D)": [c for c in ("obj_any_max_conf", "obj_any_hit_frames") if c in t.columns],
        "x-clip (E)": [c for c in ("xclip_smoke_logit", "xclip_smoke_minus_max_other", "xclip_smoke_or_vape") if c in t.columns],
        "videomae (E)": [c for c in ("videomae_k_smoking", "videomae_smoke_share", "videomae_smoke_minus_drink") if c in t.columns],
        "vlm (G)": [c for c in ("vlm_yesno", "vlm_letter") if c in t.columns],
        # выбраны по смыслу до просмотра AUC: расстояние кисти до рта в удержании, сгиб локтя в пике, наклон предплечья (рука «ко рту» против «к уху/груди»), вторая рука поднята?
        "pose (H)": [c for c in ("p_wm_dist", "p_elbow_peak", "p_forearm_deg", "p_other_wrist_dy") if c in t.columns],
        "photo (I)": [c for c in ("photo_p_mean", "photo_p_max") if c in t.columns],
    }
    return {k: v for k, v in pick.items() if v}


def group_ablation(t: pd.DataFrame, repeats: int = 8, seed: int = 0) -> pd.DataFrame:
    """Грубая абляция групп: логистическая регрессия, фолды по видео (StratifiedGroupKFold, несколько перемешиваний), AUC на отложенных циклах.

    Это не итоговый классификатор (он — LightGBM на настоящей разметке, docs/TRAINING.md), а проверка «какие группы вообще добавляют сигнал».
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pick = pick_columns(t)
    combos = {k: v for k, v in pick.items()}
    names = list(pick)
    combos["kinematics + object"] = pick.get("kinematics", []) + pick.get("object (D)", [])
    combos["kinematics + x-clip"] = pick.get("kinematics", []) + pick.get("x-clip (E)", [])
    combos["kinematics + videomae"] = pick.get("kinematics", []) + pick.get("videomae (E)", [])
    if "vlm (G)" in pick:
        combos["kinematics + vlm"] = pick.get("kinematics", []) + pick["vlm (G)"]
    for g, nm in (("pose (H)", "pose"), ("photo (I)", "photo")):
        if g in pick:
            combos[f"kinematics + {nm}"] = pick.get("kinematics", []) + pick[g]
    combos["all groups"] = [c for n in names for c in pick[n]]
    y, vid = t.y.values, t.video.values
    rows = []
    for name, cols in combos.items():
        cols = list(dict.fromkeys(cols))
        if not cols:
            continue
        aucs = []
        for r in range(repeats):
            cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + r)
            oof = np.full(len(t), np.nan)
            try:
                for tr, te in cv.split(t, y, vid):
                    if len(np.unique(y[tr])) < 2:
                        continue
                    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=500))
                    model.fit(t.iloc[tr][cols].astype(float), y[tr])
                    oof[te] = model.predict_proba(t.iloc[te][cols].astype(float))[:, 1]
            except ValueError:
                continue
            ok = np.isfinite(oof)
            if ok.sum() > 10 and len(np.unique(y[ok])) == 2:
                aucs.append(roc_auc_score(y[ok], oof[ok]))
        if aucs:
            rows.append(dict(combo=name, features=", ".join(cols), auc_mean=float(np.mean(aucs)), auc_min=float(np.min(aucs)), auc_max=float(np.max(aucs)), repeats=len(aucs)))
    return pd.DataFrame(rows)


GROUP_COLORS = {"kinematics": "#2a78d6", "object (D)": "#eb6834", "x-clip (E)": "#1baf7a", "videomae (E)": "#eda100",
                "vlm (G)": "#e87ba4", "pose (H)": "#008300", "photo (I)": "#4a3aa7"}   # слоты 1–7 валидированной палитры (CVD/контраст проверены validate_palette.js)
GROUP_TITLES = {"kinematics": "Кинематика цикла (A–C)", "object (D)": "Предмет на кропе кисть–рот (D)", "x-clip (E)": "X-CLIP, zero-shot по промптам (E)",
                "videomae (E)": "VideoMAE, классы Kinetics-400 (E)", "vlm (G)": "VLM, оценка по логитам ответа (G)",
                "pose (H)": "Положение точек, углы руки, время (H)", "photo (I)": "Фото-классификатор по кропам рта (I)"}
KEY_FEATURES = ("hold", "obj_any_hit", XCLIP_SMOKE, "videomae_k_smoking", "vlm_yesno")   # показываем всегда: «главные» признаки группы, даже если они слабые


def _short(feature: str) -> str:
    return (feature.replace("obj_smoking_", "obj_").replace("p_a person ", "промпт «").replace("videomae_k_", "K400: ")) + ("»" if feature.startswith("p_a person") else "")


def plot_auc(df: pd.DataFrame, out: Path, per_group: int = 4, info: str = "") -> Path:
    """Столбцы AUC от линии случайности 0.5 с 95% интервалом (бутстрэп по видео), по группам: в каждой — `per_group` самых сильных признаков + главные.

    Подписи — в цвете текста (не серии); группа названа заголовком блока, значения и интервалы напечатаны числами, поэтому цвет не единственный носитель смысла.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, muted, grid, surface = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
    d = df.dropna(subset=["auc"]).copy()
    d["dev"] = (d.auc - 0.5).abs()
    blocks = []
    for g in GROUP_COLORS:
        sub = d[d.group == g]
        if sub.empty:
            continue
        keep = set(sub.sort_values("dev", ascending=False).feature.head(per_group)) | (set(KEY_FEATURES) & set(sub.feature))
        blocks.append((g, sub[sub.feature.isin(keep)].sort_values("auc", ascending=False)))
    nrows = sum(len(b) + 1.0 for _, b in blocks)
    fig, ax = plt.subplots(figsize=(10.5, 0.33 * nrows + 2.1), facecolor=surface)
    ax.set_facecolor(surface)
    y, ticks, labels = 0.0, [], []
    for g, b in blocks:
        ax.text(0.0, y, GROUP_TITLES[g], transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9, fontweight="bold", color=ink, clip_on=False)
        y += 1
        for r in b.itertuples():
            ax.barh(y, r.auc - 0.5, left=0.5, height=0.52, color=GROUP_COLORS[g], zorder=3)
            if np.isfinite(r.lo):
                ax.plot([r.lo, r.hi], [y, y], color=ink2, lw=1.2, zorder=4, solid_capstyle="butt")
                for xe in (r.lo, r.hi):
                    ax.plot([xe, xe], [y - 0.16, y + 0.16], color=ink2, lw=1.2, zorder=4)
            ax.text(1.012, y, f"{r.auc:.2f}  [{r.lo:.2f}–{r.hi:.2f}]", transform=ax.get_yaxis_transform(), va="center", fontsize=8.5, color=ink)
            ticks.append(y)
            labels.append(_short(r.feature))
            y += 1
    ax.set_ylim(y - 0.4, -0.7)
    ax.set_xlim(0.0, 1.0)
    ax.set_yticks(ticks, labels, fontsize=8.5, color=ink2)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.tick_params(axis="x", labelsize=8.5, colors=muted, length=0)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.grid(axis="x", color=grid, lw=0.7, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.axvline(0.5, color=ink2, lw=1.0, zorder=5)
    ax.axvline(0.8, color=ink2, lw=0.9, ls=(0, (2, 3)), zorder=5)
    ax.text(0.5, -0.9, "случайность 0.5", ha="center", va="bottom", fontsize=8, color=ink2, clip_on=False)
    ax.text(0.8, -0.9, "порог руководства 0.80", ha="center", va="bottom", fontsize=8, color=ink2, clip_on=False)
    ax.set_xlabel("AUC «курение против питья / касания лица / телефона»; усы — 95% интервал бутстрэпом по видео. AUC < 0.5: признак выше у не-курения", fontsize=8.5, color=ink2, labelpad=8)
    fig.suptitle("Какие признаки цикла отделяют курение: грубая проверка на 68 циклах", x=0.01, ha="left", fontsize=12, fontweight="bold", color=ink, y=0.995)
    fig.text(0.01, 0.945 if nrows > 8 else 0.93, info or "метки — оценка ассистента по кадрам, не эталон; негативы в основном из другой папки видео (см. ANALYSIS.md, 7.5)",
             ha="left", fontsize=8.5, color=ink2)
    present = [g for g, _ in blocks]
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in present]
    legend = {"kinematics": "кинематика", "object (D)": "предмет (D)", "x-clip (E)": "X-CLIP (E)", "videomae (E)": "VideoMAE (E)", "vlm (G)": "VLM (G)",
              "pose (H)": "поза/время (H)", "photo (I)": "фото-модель (I)"}
    fig.legend(handles, [legend[g] for g in present], loc="lower center", ncol=len(present), frameon=False, fontsize=8.5, labelcolor=ink2,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.2)
    fig.subplots_adjust(left=0.30, right=0.84, top=0.87, bottom=0.135)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=surface)
    plt.close(fig)
    return out
