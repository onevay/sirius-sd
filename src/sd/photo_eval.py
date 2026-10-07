"""Оценка фото-классификатора: внутри набора (CV по группам дублей), перенос между наборами, «трудные» негативы, страты яркости, zero-shot CLIP.

Результат — таблицы в outputs/photo/eval_*.csv; всё считается по эмбеддингам из кэша (`sd photos embed`), сеть и backbone не нужны.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import numpy as np
import pandas as pd

from . import photo_clf as PC
from . import photo_feats as PF

KINDS = ("logreg", "svm", "et")


def cached_matrix(name: str, paths: list[str]) -> np.ndarray:
    """Эмбеддинги из кэша; строки без эмбеддинга — NaN."""
    cache = PF.load_cache(name)
    out = np.full((len(paths), PF.DIM[name]), np.nan, np.float32)
    for i, p in enumerate(paths):
        v = cache.get(PF._key(p))
        if v is not None:
            out[i] = v
    return out


def feature_sets(df: pd.DataFrame, backbones=("clip", "convnext")) -> dict[str, np.ndarray]:
    """{'clip': X, 'convnext': X, 'clip+convnext': concat}; только строки, где есть все backbone'ы."""
    mats = {bb: cached_matrix(bb, df.path.tolist()) for bb in backbones}
    out = dict(mats)
    if len(backbones) > 1:
        out["+".join(backbones)] = np.concatenate([mats[bb] for bb in backbones], axis=1)
    return out


def usable(df: pd.DataFrame, X: dict[str, np.ndarray]) -> pd.DataFrame:
    ok = np.ones(len(df), bool)
    for m in X.values():
        ok &= ~np.isnan(m).any(1)
    return df[ok & df.label.notna().to_numpy()]


def within_source(df: pd.DataFrame, X: dict[str, np.ndarray], kinds=KINDS, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """OOF-оценки по каждому набору (группы = xdup_group). Возвращает (таблица метрик, {(source, member): OOF})."""
    rows, store = [], {}
    for src, g in df.groupby("source"):
        y, grp = g.label.to_numpy(int), g.xdup_group.to_numpy()
        if len(np.unique(y)) < 2:
            continue
        idx = g.index.to_numpy()
        member_scores = {}
        for fs, M in X.items():
            for k in kinds:
                s = PC.oof_scores(M[idx], y, grp, k, seed=seed)
                member_scores[f"{fs}:{k}"] = s
                rows.append(dict(source=src, member=f"{fs}:{k}", **PC.summarize(y, s)))
        for name, sel in (("ens:clip", lambda m: m.startswith("clip:")), ("ens:convnext", lambda m: m.startswith("convnext:")),
                          ("ens:all-single-backbone", lambda m: ":" in m and "+" not in m.split(":")[0])):
            mem = [m for m in member_scores if sel(m)]
            if len(mem) > 1:
                s = np.mean([PC.rank01(member_scores[m]) for m in mem], axis=0)
                member_scores[name] = s
                rows.append(dict(source=src, member=name, **PC.summarize(y, s, thr=0.5)))
        for m, s in member_scores.items():
            store[(src, m)] = (idx, s)
    return pd.DataFrame(rows), store


def transfer(df: pd.DataFrame, X: dict[str, np.ndarray], kinds=KINDS, seed: int = 0) -> pd.DataFrame:
    """Обучение на одном наборе → проверка на другом (оба направления)."""
    rows, srcs = [], sorted(df.source.unique())
    for a in srcs:
        for b in srcs:
            if a == b:
                continue
            ga, gb = df[df.source == a], df[df.source == b]
            for fs, M in X.items():
                for k in kinds:
                    s = PC.fit_predict(k, M[ga.index.to_numpy()], ga.label.to_numpy(int), M[gb.index.to_numpy()], seed)
                    rows.append(dict(train=a, test=b, member=f"{fs}:{k}", **PC.summarize(gb.label.to_numpy(int), s)))
    return pd.DataFrame(rows)


def official_split(df: pd.DataFrame, X: dict[str, np.ndarray], kinds=KINDS, seed: int = 0) -> pd.DataFrame:
    """Обучение на train(+val), проверка на test — как у авторов набора; для сравнения с опубликованными числами."""
    rows = []
    for src, g in df.groupby("source"):
        tr, te = g[g.split.isin(["train", "val"])], g[g.split == "test"]
        if len(tr) < 50 or len(te) < 50 or te.label.nunique() < 2:
            continue
        for fs, M in X.items():
            for k in kinds:
                s = PC.fit_predict(k, M[tr.index.to_numpy()], tr.label.to_numpy(int), M[te.index.to_numpy()], seed)
                rows.append(dict(source=src, member=f"{fs}:{k}", train_n=len(tr), **PC.summarize(te.label.to_numpy(int), s)))
    return pd.DataFrame(rows)


def zero_shot_table(df: pd.DataFrame, emb_clip: np.ndarray, text: np.ndarray) -> pd.DataFrame:
    """AUC CLIP без обучения: сходство с промптом «smoking» и разности с остальными промптами."""
    zs = PF.zero_shot(emb_clip, text)
    cand = {"smoke": zs["smoke"], "hold": zs["hold"], "smoke-none": zs["smoke"] - zs["none"], "smoke-max(drink,phone)": zs["smoke"] - np.maximum(zs["drink"], zs["phone"]),
            "smoke-max(all others)": zs["smoke"] - np.maximum.reduce([zs["hold"] * 0, zs["drink"], zs["phone"], zs["none"]])}
    rows = []
    for src, g in df.groupby("source"):
        y = g.label.to_numpy(int)
        for name, s in cand.items():
            rows.append(dict(source=src, prompt=name, auc=PC.auc(y, s[g.index.to_numpy()])))
    return pd.DataFrame(rows)


FAMILY_COLORS = {"clip": "#2a78d6", "convnext": "#eb6834", "clip+convnext": "#1baf7a", "ens": "#eda100"}   # слоты 1–4 валидированной палитры


def _family(member: str) -> str:
    return "ens" if member.startswith("ens:") else member.split(":")[0]


def shared_duplicate_groups(df: pd.DataFrame, min_groups: int = 20) -> dict[frozenset, int]:
    """Пары наборов, у которых есть общие группы почти-дубликатов (колонки `xdup_group`, `source`): перенос между ними нельзя считать независимой проверкой."""
    from itertools import combinations

    g = df.groupby("xdup_group").source.agg(lambda s: tuple(sorted(set(s))))
    cnt: dict[frozenset, int] = {}
    for srcs in g[g.map(len) > 1]:
        for a, b in combinations(srcs, 2):
            cnt[frozenset((a, b))] = cnt.get(frozenset((a, b)), 0) + 1
    return {k: v for k, v in cnt.items() if v >= min_groups}


def plot_eval(within: pd.DataFrame, transfer_df: pd.DataFrame, out, meta_auc: dict[str, float] | None = None, title: str = "", shared: dict | None = None) -> object:
    """Четыре панели: AUC внутри каждого набора (CV по группам дублей) и перенос между наборами; линия 0.5 — случайность, пунктир — «только метаданные»."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    ink, ink2, muted, grid, surface = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
    sources = sorted(within.source.unique())
    panels = [(f"Внутри набора: {s}", within[within.source == s].set_index("member").auc, (meta_auc or {}).get(s)) for s in sources]
    for a in sources:
        for b in sources:
            if a != b:
                t = transfer_df[(transfer_df.train == a) & (transfer_df.test == b)]
                warn = f"  (⚠ {shared[frozenset((a, b))]} общих групп дублей)" if shared and frozenset((a, b)) in shared else ""
                panels.append((f"Перенос: {a} → {b}{warn}", t.set_index("member").auc, None))
    n = len(panels)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(12.5, 3.6 * rows), facecolor=surface, squeeze=False)
    for ax, (ttl, ser, meta) in zip(axes.ravel(), panels):
        ser = ser.dropna().sort_values()
        y = np.arange(len(ser))
        ax.set_facecolor(surface)
        ax.barh(y, ser.values - 0.5, left=0.5, height=0.62, color=[FAMILY_COLORS[_family(m)] for m in ser.index], zorder=3)
        for yi, v in zip(y, ser.values):
            ax.text(min(v, 0.995) + 0.008 if v < 0.95 else v - 0.008, yi, f"{v:.3f}", va="center", ha="left" if v < 0.95 else "right", fontsize=7.5,
                    color=ink if v < 0.95 else surface)
        ax.set_yticks(y, ser.index, fontsize=7.5, color=ink2)
        ax.set_xlim(0.4, 1.0)
        ax.axvline(0.5, color=ink2, lw=1.0, zorder=5)
        if meta:
            ax.axvline(meta, color=ink2, lw=1.0, ls=(0, (3, 3)), zorder=5)
            ax.text(meta, 1.01, f"метаданные {meta:.2f}", transform=ax.get_xaxis_transform(), fontsize=7, color=ink2, ha="center", va="bottom")
        ax.set_title(ttl, loc="left", fontsize=9.5, fontweight="bold", color=ink)
        ax.grid(axis="x", color=grid, lw=0.7, zorder=0)
        ax.tick_params(axis="x", labelsize=8, colors=muted, length=0)
        ax.tick_params(axis="y", length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    fig.legend(handles, ["CLIP", "ConvNeXt-B", "CLIP+ConvNeXt (конкатенация)", "ансамбль (средний ранг)"], loc="lower center", ncol=4, frameon=False, fontsize=8.5, labelcolor=ink2)
    kinds = sorted({m.split(":")[1] for m in within.member if ":" in m and not m.startswith("ens:")})
    fig.suptitle(title or f"Фото-классификатор «курит / не курит»: AUC на эмбеддингах ({' / '.join(kinds)})", x=0.01, ha="left", fontsize=12, fontweight="bold", color=ink)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=surface)
    plt.close(fig)
    return out


def run_all(df: pd.DataFrame, backbones=("clip", "convnext"), kinds=KINDS, seed: int = 0) -> dict:
    X = feature_sets(df, backbones)
    d = usable(df, X).reset_index(drop=True).copy()
    X = feature_sets(d, backbones)
    within, store = within_source(d, X, kinds, seed)
    return dict(table=d, X=X, within=within, store=store, transfer=transfer(d, X, kinds, seed), official=official_split(d, X, kinds, seed))
