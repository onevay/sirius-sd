"""Конечный автомат цикла жеста: REST -> APPROACH -> AT_MOUTH (0.5–5 с) -> RETRACT -> REST (руководство §5.4, шаг 4).

Потоковый: `CycleDetector.update(t, d, hand)` вызывается на каждом обработанном кадре одного трека и возвращает цикл, когда тот закрыт
(рука отведена). Тот же код работает offline (`find_cycles`) и в real-time. Гистерезис (th_in < th_out) убирает дребезг на границе.

Причины отбраковки жестов сохраняются в `rejected` — это материал для разбора ошибок (§9.5) и для UI:
  short_touch — пауза у рта < hold_min (касание лица/чесание), long_hold — пауза > hold_max (телефон у уха, долгое питьё),
  lost — данные пропали дольше max_gap_sec посреди жеста, track_end — трек оборвался у рта.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

REST, APPROACH, AT_MOUTH, RETRACT, LONG_HOLD = "REST", "APPROACH", "AT_MOUTH", "RETRACT", "LONG_HOLD"
STATE_ID = {REST: 0, APPROACH: 1, AT_MOUTH: 2, RETRACT: 3, LONG_HOLD: 4}


@dataclass
class Cycle:
    tid: int
    start: float       # начало подъёма руки (уточнено назад по скорости сближения)
    mouth_in: float    # вход в зону рта (d < th_in)
    mouth_out: float   # выход из зоны рта (d > th_out)
    end: float         # рука отведена (d > th_out + retract_margin)
    hold: float        # пауза у рта, с
    hand: int          # 0 левая, 1 правая, -1 неизвестно
    d_min: float       # минимум d в паузе (в ширинах плеч)

    @property
    def peak_t(self) -> float:
        """Середина паузы у рта — кандидат в peak_sec (руководство, шаг 7)."""
        return 0.5 * (self.mouth_in + self.mouth_out)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["peak_t"] = self.peak_t
        return d


@dataclass
class Rejected:
    tid: int
    t0: float
    t1: float
    reason: str
    hold: float = float("nan")


class CycleDetector:
    def __init__(self, cfg_cycles: dict, tid: int = 0):
        self.c = cfg_cycles
        self.tid = tid
        self.state = REST
        self.hist: deque[tuple[float, float]] = deque()  # недавние валидные (t, d) для поиска начала подъёма назад
        self.last_valid_t: float | None = None
        self.t_approach = self.t_in = self.t_out = self.t_start = None
        self.hands: list[int] = []
        self.d_min = np.inf
        self.last_end = -np.inf
        self.pending: dict | None = None
        self.cycles: list[Cycle] = []
        self.rejected: list[Rejected] = []

    # ------------------------------------------------------------------ helpers
    def _reject(self, reason: str, t: float, hold: float = float("nan")) -> None:
        t0 = self.t_in if self.t_in is not None else (self.t_approach if self.t_approach is not None else t)
        self.rejected.append(Rejected(self.tid, float(t0), float(t), reason, hold))

    def _reset(self) -> None:
        self.state = REST
        self.t_approach = self.t_in = self.t_out = self.t_start = None
        self.hands, self.d_min, self.pending = [], np.inf, None

    def _backtrack_start(self) -> float:
        """Вызывается в момент входа в зону рта (последний элемент истории = текущий отсчёт).

        Идём назад, пока рука заметно сближается со ртом (скорость сближения > v_min) — это и есть начало подъёма.
        """
        h = list(self.hist)
        j = len(h) - 1
        t_in = h[j][0]
        while j > 0 and t_in - h[j - 1][0] <= self.c["backtrack_max_sec"]:
            dt = h[j][0] - h[j - 1][0]
            if dt <= 0:
                break
            speed = (h[j - 1][1] - h[j][1]) / dt  # > 0: расстояние уменьшается (рука приближается)
            if speed > self.c["backtrack_v_min"]:
                j -= 1
            else:
                break
        return max(h[j][0], self.last_end)

    def _close_pending(self, end_t: float) -> Cycle:
        p = self.pending
        hand = max(set(p["hands"]), key=p["hands"].count) if p["hands"] else -1
        cyc = Cycle(self.tid, p["start"], p["t_in"], p["t_out"], float(end_t), p["t_out"] - p["t_in"], int(hand), float(p["d_min"]))
        self.cycles.append(cyc)
        self.last_end = float(end_t)
        return cyc

    # ------------------------------------------------------------------ main step
    def update(self, t: float, d: float, hand: int = -1) -> Cycle | None:
        c = self.c
        if not np.isfinite(d):
            if self.state != REST and self.last_valid_t is not None and t - self.last_valid_t > c["max_gap_sec"]:
                if self.state == RETRACT and self.pending is not None:
                    cyc = self._close_pending(self.last_valid_t)
                    self._reset()
                    return cyc
                self._reject("lost", t)
                self._reset()
            return None
        self.last_valid_t = t
        self.hist.append((t, float(d)))
        while self.hist and t - self.hist[0][0] > c["backtrack_max_sec"] + 1.0:
            self.hist.popleft()
        th_in, th_out = c["th_in"], c["th_out"]
        emitted: Cycle | None = None

        if self.state == RETRACT:
            if d < th_out:  # рука вернулась ко рту, не дойдя до полного отведения: закрываем прошлый цикл и начинаем новый
                emitted = self._close_pending(self.t_out)
                self._reset()
            elif d > th_out + c["retract_margin"]:
                emitted = self._close_pending(t)
                self._reset()
                return emitted

        if self.state == REST and d < th_out:
            self.state, self.t_approach = APPROACH, t
        if self.state == APPROACH:
            if d < th_in:
                self.state, self.t_in, self.d_min, self.hands = AT_MOUTH, t, float(d), [hand]
                self.t_start = self._backtrack_start()
            elif d > th_out + c["abort_margin"]:
                self._reset()  # рука не дошла до рта
        elif self.state == AT_MOUTH:
            self.d_min = min(self.d_min, float(d))
            self.hands.append(hand)
            # emit_long_hold: курильщик может держать сигарету у лица > 5 с подряд (пример sm_4: ~7 с). Такие удержания (до hold_cap_sec)
            # остаются кандидатами; «правдоподобие» длинной паузы оценивает классификатор (эвристический бейзлайн её занижает)
            hold_limit = c["hold_cap_sec"] if c.get("emit_long_hold") else c["hold_max_sec"]
            if d > th_out:
                hold = t - self.t_in
                if c["hold_min_sec"] <= hold <= hold_limit:
                    self.t_out = t
                    self.pending = dict(start=self.t_start, t_in=self.t_in, t_out=t, hands=self.hands, d_min=self.d_min)
                    self.state = RETRACT
                else:
                    self._reject("short_touch" if hold < c["hold_min_sec"] else "long_hold", t, hold)
                    self._reset()
            elif t - self.t_in > hold_limit:
                self.state = LONG_HOLD
        elif self.state == LONG_HOLD and d > th_out:
            self._reject("long_hold", t, t - self.t_in)
            self._reset()
        return emitted

    def flush(self, t_end: float) -> Cycle | None:
        """Конец трека. Закрывает цикл, если рука уже отводилась; жест у рта без отведения отбрасывает."""
        cyc = None
        if self.state == RETRACT and self.pending is not None:
            cyc = self._close_pending(self.last_valid_t if self.last_valid_t is not None else t_end)
        elif self.state in (AT_MOUTH, LONG_HOLD):
            self._reject("track_end", t_end, (t_end - self.t_in) if self.t_in is not None else float("nan"))
        self._reset()
        return cyc


def find_cycles(series: pd.DataFrame, cfg: dict, tid: int = 0) -> tuple[list[Cycle], list[Rejected], np.ndarray]:
    """Offline-обёртка: прогон автомата по ряду признаков одного трека.

    Возвращает циклы, отброшенные жесты и код состояния FSM для каждого отсчёта (для видео и графиков).
    """
    det = CycleDetector(cfg["cycles"], tid)
    states = np.zeros(len(series), np.int8)
    t, d, hand = series["t"].values, series["d"].values, series["hand"].values
    for i in range(len(series)):
        det.update(float(t[i]), float(d[i]), int(hand[i]))
        states[i] = STATE_ID[det.state]
    if len(series):
        det.flush(float(t[-1]))
    return det.cycles, det.rejected, states
