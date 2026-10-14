"""Модели по временным рядам для оценки цикла «рука ко рту»: вместо ручных признаков цикла смотрят на форму самих рядов (d_t, скорость, угол локтя, высота кисти, наклон головы).

Окно цикла [start − pad, end + pad] сжимается линейной интерполяцией до `length` отсчётов по каждому каналу — получается тензор (циклов × каналов × отсчётов). К нему добавляются два
скаляра (длительность окна и пауза у рта), потому что передискретизация стирает абсолютное время. Три семейства моделей — от самой простой к самой гибкой:
  stats    описательные статистики формы (минимум, максимум, положение пика, наклон, число пиков …) + логистическая регрессия: мало параметров, хорошо работает на сотне циклов;
  rocket   случайные свёрточные ядра (в духе ROCKET/MiniRocket: доля положительных значений и максимум свёртки) + логистическая регрессия: ловит форму без ручных признаков;
  cnn1d    маленькая 1D-свёрточная сеть (torch, если установлен): аугментации сдвигом, шумом и масштабом, ранняя регуляризация весом.
Модели сохраняются без pickle (JSON для stats/rocket, npz + JSON для cnn1d) и оцениваются так же честно, как классификатор: фолды по видео или по сценам, повторы, интервал по видео.
Подключение к конвейеру не меняет пакет классификатора: оценку этой модели можно сравнить и смешать с оценкой пакета (`blend_curve`), чтобы понять, стоит ли она отдельной интеграции.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

CHANNELS = {
    "d": "расстояние кисть–рот (сглаженное)",
    "d_raw": "расстояние кисть–рот (сырое)",
    "v": "скорость сближения",
    "elbow": "угол локтя активной руки",
    "wrist_h": "высота кисти относительно плеч",
    "head_tilt": "наклон головы",
    "wrist_conf": "уверенность запястья",
}
DEFAULT_CHANNELS = ("d", "v", "elbow", "wrist_h", "head_tilt")
MODELS = {
    "stats": "статистики формы ряда + логистическая регрессия",
    "rocket": "случайные свёрточные ядра (ROCKET) + логистическая регрессия",
    "cnn1d": "маленькая 1D-свёрточная сеть (нужен torch)",
}
DEFAULT_PARAMS = {
    "stats": {"C": 0.3},
    "rocket": {"n_kernels": 300, "C": 0.05, "seed": 0},
    "cnn1d": {"hidden": 16, "epochs": 80, "lr": 0.003, "weight_decay": 0.001, "seed": 0},
}


@dataclass
class TSSpec:
    model: str = "stats"
    channels: list[str] = field(default_factory=lambda: list(DEFAULT_CHANNELS))
    length: int = 64
    pad: float = 1.0
    params: dict = field(default_factory=dict)

    def full_params(self) -> dict:
        return {**DEFAULT_PARAMS[self.model], **self.params}

    def to_dict(self) -> dict:
        return dict(model=self.model, channels=list(self.channels), length=int(self.length), pad=float(self.pad), params=self.full_params())

    @classmethod
    def from_dict(cls, d: dict) -> "TSSpec":
        return cls(d["model"], list(d["channels"]), int(d["length"]), float(d["pad"]), dict(d.get("params") or {}))


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:       # noqa: BLE001 — любой сбой импорта означает «нет»
        return False


# ---------------------------------------------------------------------------------------------- данные
def _channel(s: pd.DataFrame, name: str, hand: int) -> np.ndarray:
    if name == "elbow":
        col = "elbowL" if hand == 0 else "elbowR"
        return s[col].to_numpy(float) if col in s else np.full(len(s), np.nan)
    return s[name].to_numpy(float) if name in s else np.full(len(s), np.nan)


def cycle_tensor(cycles: pd.DataFrame, series_by_video: dict[str, pd.DataFrame], channels: Sequence[str] = DEFAULT_CHANNELS, length: int = 64, pad: float = 1.0
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(X, S): X — (циклов, каналов, length) с пропусками NaN; S — (циклов, 2): длительность окна и пауза. Строки идут в порядке `cycles`."""
    n = len(cycles)
    X = np.full((n, len(channels), length), np.nan, np.float32)
    S = np.zeros((n, 2), np.float32)
    ser_tid = {v: {int(t): g.sort_values("t") for t, g in s.groupby("tid")} for v, s in series_by_video.items()}
    for i, r in enumerate(cycles.itertuples()):
        s = ser_tid.get(r.video, {}).get(int(r.tid))
        t0, t1 = float(r.start) - pad, float(r.end) + pad
        S[i] = (t1 - t0, float(getattr(r, "hold", np.nan)) if np.isfinite(getattr(r, "hold", np.nan)) else 0.0)
        if s is None or len(s) < 3:
            continue
        grid = np.linspace(t0, t1, length)
        tt = s.t.to_numpy(float)
        hand = int(getattr(r, "hand", 0)) if np.isfinite(getattr(r, "hand", 0)) else 0
        for k, ch in enumerate(channels):
            y = _channel(s, ch, hand)
            ok = np.isfinite(y)
            if ok.sum() >= 3:
                X[i, k] = np.interp(grid, tt[ok], y[ok], left=np.nan, right=np.nan)
    return X, S


def series_for_clips(clips: dict, cfg_for) -> dict[str, pd.DataFrame]:
    """Ряды признаков каждого клипа для конфигурации `cfg_for(run_dir)`; считаются из кэша позы (`stages.stage_features` кэширует результат по хэшу параметров)."""
    from . import stages
    from .tracks import Tracks

    out = {}
    for name, c in clips.items():
        rd = Path(c.run_dir)
        if not (rd / "pose" / "tracks.parquet").exists():
            continue
        tr = Tracks.load(rd / "pose")
        out[name] = stages.stage_features(tr, cfg_for(rd), rd)
    return out


def prototypes(X: np.ndarray, y: np.ndarray, channels: Sequence[str]) -> pd.DataFrame:
    """Средняя форма и разброс каждого канала по классам (для графика «как выглядит затяжка»): колонки channel, cls, i, mean, lo, hi."""
    rows = []
    for k, ch in enumerate(channels):
        for cls, name in ((1, "затяжка"), (0, "не затяжка")):
            a = X[y == cls, k, :]
            if not len(a):
                continue
            m, sd = np.nanmean(a, axis=0), np.nanstd(a, axis=0)
            rows += [dict(channel=ch, cls=name, i=i, mean=float(m[i]), lo=float(m[i] - sd[i]), hi=float(m[i] + sd[i])) for i in range(X.shape[2])]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- признаки
def _stat_features(X: np.ndarray) -> np.ndarray:
    from scipy.signal import find_peaks

    n, C, L = X.shape
    out = []
    x01 = np.linspace(0, 1, L)
    for i in range(n):
        f = []
        for k in range(C):
            a = X[i, k]
            sd = float(a.std()) if a.std() > 1e-9 else 1.0
            pk, _ = find_peaks(a, prominence=0.15 * sd) if sd > 1e-9 else ([], None)
            f += [a.mean(), a.std(), a.min(), a.max(), a.max() - a.min(), x01[a.argmax()], x01[a.argmin()], a[0], a[-1], a[L // 2], np.polyfit(x01, a, 1)[0],
                  np.abs(np.diff(a)).mean(), float(len(pk))]
        out.append(f)
    return np.asarray(out, float)


def _rocket_kernels(C: int, L: int, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(n):
        ln = int(rng.choice([5, 7, 9]))
        w = rng.normal(size=ln)
        w -= w.mean()
        max_exp = np.log2(max((L - 1) / (ln - 1), 1.0))
        ks.append(dict(w=w.tolist(), bias=float(rng.uniform(-1, 1)), dilation=int(2 ** rng.uniform(0, max_exp)), channel=int(rng.integers(0, C))))
    return ks


def _rocket_features(X: np.ndarray, kernels: list[dict]) -> np.ndarray:
    n = X.shape[0]
    out = np.zeros((n, 2 * len(kernels)))
    for j, k in enumerate(kernels):
        w = np.asarray(k["w"])
        d = k["dilation"]
        dk = np.zeros((len(w) - 1) * d + 1)
        dk[::d] = w
        for i in range(n):
            c = np.convolve(X[i, k["channel"]], dk, mode="same") + k["bias"]
            out[i, 2 * j] = float((c > 0).mean())
            out[i, 2 * j + 1] = float(c.max())
    return out


# ---------------------------------------------------------------------------------------------- модель
class TSModel:
    """Обучаемая модель по тензору циклов. `fit(X, S, y, w)` → `predict(X, S)` (вероятность затяжки). Состояние сериализуемо без pickle."""

    def __init__(self, spec: TSSpec):
        if spec.model not in MODELS:
            raise ValueError(f"неизвестная модель {spec.model!r} (допустимо {', '.join(MODELS)})")
        self.spec = spec
        self.fill: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.state: dict = {}

    # ---- предобработка
    def _norm(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X = np.array(X, np.float64)       # float64: параметры нормировки после записи в JSON и чтения дают тот же результат, что и в памяти
        if fit:
            self.fill = np.nanmedian(X, axis=(0, 2))
            self.fill = np.where(np.isfinite(self.fill), self.fill, 0.0)
        for k in range(X.shape[1]):
            m = ~np.isfinite(X[:, k])
            X[:, k][m] = self.fill[k]
        if fit:
            self.mean = X.mean(axis=(0, 2))
            self.std = np.where(X.std(axis=(0, 2)) > 1e-6, X.std(axis=(0, 2)), 1.0)
        return (X - self.mean[None, :, None]) / self.std[None, :, None]

    def _feats(self, Xn: np.ndarray, S: np.ndarray) -> np.ndarray:
        if self.spec.model == "stats":
            F = _stat_features(Xn)
        else:
            F = _rocket_features(Xn, self.state["kernels"])
        return np.hstack([F, np.asarray(S, float)])

    # ---- обучение
    def fit(self, X: np.ndarray, S: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> "TSModel":
        p = self.spec.full_params()
        Xn = self._norm(X, fit=True)
        y = np.asarray(y).astype(int)
        if self.spec.model == "cnn1d":
            self._fit_cnn(Xn, np.asarray(S, np.float32), y, w, p)
            return self
        if self.spec.model == "rocket":
            self.state["kernels"] = _rocket_kernels(Xn.shape[1], Xn.shape[2], int(p["n_kernels"]), int(p["seed"]))
        F = self._feats(Xn, S)
        mu, sd = F.mean(0), np.where(F.std(0) > 1e-9, F.std(0), 1.0)
        lr = LogisticRegression(C=float(p["C"]), max_iter=3000, class_weight="balanced").fit((F - mu) / sd, y, sample_weight=w)
        self.state.update(mu=mu.tolist(), sd=sd.tolist(), coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]))
        return self

    def predict(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        Xn = self._norm(X)
        if self.spec.model == "cnn1d":
            return self._predict_cnn(Xn, np.asarray(S, np.float32))
        F = (self._feats(Xn, S) - np.array(self.state["mu"])) / np.array(self.state["sd"])
        z = F @ np.array(self.state["coef"]) + self.state["intercept"]
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    # ---- нейросеть
    def _net(self, C: int, p: dict):
        import torch
        from torch import nn

        h = int(p["hidden"])

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Sequential(nn.Conv1d(C, h, 5, padding=2), nn.ReLU(), nn.Conv1d(h, h, 5, stride=2, padding=2), nn.ReLU())
                self.head = nn.Linear(2 * h + 2, 1)

            def forward(self, x, s):
                z = self.conv(x)
                return self.head(torch.cat([z.mean(-1), z.amax(-1), s], 1)).squeeze(-1)

        return Net()

    def _fit_cnn(self, Xn: np.ndarray, S: np.ndarray, y: np.ndarray, w: np.ndarray | None, p: dict) -> None:
        import torch

        torch.manual_seed(int(p["seed"]))
        Xn = Xn.astype(np.float32)
        rng = np.random.default_rng(int(p["seed"]))
        self.s_mu, self.s_sd = S.mean(0), np.where(S.std(0) > 1e-6, S.std(0), 1.0)
        Sn = (S - self.s_mu) / self.s_sd
        net = self._net(Xn.shape[1], p)
        opt = torch.optim.Adam(net.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"]))
        pos = max(float((y == 1).sum()), 1.0)
        pw = torch.tensor((len(y) - pos) / pos)
        sw = torch.tensor(np.ones(len(y)) if w is None else np.asarray(w, float), dtype=torch.float32)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        Xt, St, yt = torch.tensor(Xn), torch.tensor(Sn, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        for _ in range(int(p["epochs"])):
            shift = int(rng.integers(-3, 4))
            xb = torch.roll(Xt, shift, dims=2) * float(rng.uniform(0.9, 1.1)) + 0.03 * torch.randn_like(Xt)       # аугментация: сдвиг, масштаб, шум
            perm = torch.randperm(len(y))
            for i in range(0, len(y), 32):
                idx = perm[i:i + 32]
                opt.zero_grad()
                loss = (lossf(net(xb[idx], St[idx]), yt[idx]) * sw[idx]).mean()
                loss.backward()
                opt.step()
        self.state = {k: v.detach().numpy() for k, v in net.state_dict().items()}

    def _predict_cnn(self, Xn: np.ndarray, S: np.ndarray) -> np.ndarray:
        import torch

        Xn = Xn.astype(np.float32)
        net = self._net(Xn.shape[1], self.spec.full_params())
        net.load_state_dict({k: torch.tensor(v) for k, v in self.state.items()})
        net.eval()
        with torch.no_grad():
            z = net(torch.tensor(Xn), torch.tensor((S - self.s_mu) / self.s_sd, dtype=torch.float32))
        return torch.sigmoid(z).numpy()

    # ---- диск (без pickle)
    def save(self, d: str | Path, meta: dict | None = None) -> Path:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        man = dict(kind="ts_classifier", format=1, spec=self.spec.to_dict(), fill=self.fill.tolist(), mean=self.mean.tolist(), std=self.std.tolist(), **(meta or {}))
        if self.spec.model == "cnn1d":
            np.savez(d / "weights.npz", **self.state)
            man.update(s_mu=self.s_mu.tolist(), s_sd=self.s_sd.tolist())
        else:
            (d / "model.json").write_text(json.dumps(self.state), encoding="utf-8")
        (d / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
        return d

    @classmethod
    def load(cls, d: str | Path) -> "TSModel":
        d = Path(d)
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if man.get("kind") != "ts_classifier":
            raise ValueError(f"{d}: не модель по временным рядам")
        m = cls(TSSpec.from_dict(man["spec"]))
        m.fill, m.mean, m.std = np.array(man["fill"]), np.array(man["mean"]), np.array(man["std"])
        if m.spec.model == "cnn1d":
            m.state = {k: v for k, v in np.load(d / "weights.npz", allow_pickle=False).items()}
            m.s_mu, m.s_sd = np.array(man["s_mu"]), np.array(man["s_sd"])
        else:
            m.state = json.loads((d / "model.json").read_text(encoding="utf-8"))
        return m


# ---------------------------------------------------------------------------------------------- оценка
def cv_oof(X: np.ndarray, S: np.ndarray, y: np.ndarray, groups: np.ndarray, spec: TSSpec, n_splits: int = 5, repeats: int = 2, seed: int = 0,
           permute_channel: int | None = None, w: np.ndarray | None = None) -> np.ndarray:
    """Out-of-fold вероятности затяжки: фолды по группам (видео или сцена), оценки усредняются по повторам. `permute_channel` перемешивает канал среди тестовых циклов (важность)."""
    out, cnt = np.zeros(len(y)), np.zeros(len(y))
    rng = np.random.default_rng(seed + 7)
    for r in range(repeats):
        cv = StratifiedGroupKFold(n_splits=min(n_splits, len(set(groups))), shuffle=True, random_state=seed + r)
        for tr, te in cv.split(X, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            m = TSModel(TSSpec(spec.model, spec.channels, spec.length, spec.pad, {**spec.params, "seed": seed + r})).fit(X[tr], S[tr], y[tr], None if w is None else w[tr])
            Xte = X[te].copy()
            if permute_channel is not None:
                Xte[:, permute_channel] = Xte[rng.permutation(len(te)), permute_channel]
            out[te] += m.predict(Xte, S[te])
            cnt[te] += 1
    return out / np.maximum(cnt, 1)


def channel_importance(X: np.ndarray, S: np.ndarray, y: np.ndarray, groups: np.ndarray, spec: TSSpec, base_auc: float, **kw) -> pd.DataFrame:
    """Падение AUC при перемешивании канала среди тестовых циклов: больше — канал важнее."""
    from .feature_auc import _auc

    rows = []
    for k, ch in enumerate(spec.channels):
        p = cv_oof(X, S, y, groups, spec, permute_channel=k, **kw)
        rows.append(dict(канал=ch, описание=CHANNELS.get(ch, ""), падение_AUC=round(base_auc - _auc(y, p), 4)))
    return pd.DataFrame(rows).sort_values("падение_AUC", ascending=False, ignore_index=True)


def blend_curve(y: np.ndarray, p_tab: np.ndarray, p_ts: np.ndarray, groups: np.ndarray | None = None, steps: int = 11) -> pd.DataFrame:
    """AUC смеси (1−w)·табличная + w·временные ряды при w от 0 до 1. Если даны группы — 95 % интервал бутстрэпом по группам."""
    from .classifier import _boot_auc
    from .feature_auc import _auc

    rows = []
    for w in np.linspace(0, 1, steps):
        s = (1 - w) * p_tab + w * p_ts
        lo, hi = _boot_auc(y, s, groups, n=150) if groups is not None else (np.nan, np.nan)
        rows.append(dict(w=round(float(w), 2), AUC=round(_auc(y, s), 4), lo=round(lo, 3), hi=round(hi, 3)))
    return pd.DataFrame(rows)
