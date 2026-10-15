"""Сводка для отчётов: тревоги за период по дням или часам, причины «ложных», разрез по камерам, суточный профиль.

Подсчёт идёт в Python по выборке (`MAX_ROWS` строк): не зависит от функций даты конкретной СУБД. Если тревог больше — в ответе `truncated=true`, период стоит сузить.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import clock
from ..models import ALERT_STATUSES, Alert, AlertReview
from .alerts import AlertError

MAX_ROWS = 50000
MAX_DAYS = 366
BUCKETS = ("day", "hour")


def _bucket_key(t: datetime, bucket: str) -> str:
    return t.strftime("%Y-%m-%d") if bucket == "day" else t.strftime("%Y-%m-%dT%H:00")


def _bucket_starts(since: datetime, until: datetime, bucket: str) -> list[datetime]:
    step = timedelta(days=1) if bucket == "day" else timedelta(hours=1)
    cur = since.replace(hour=0, minute=0, second=0) if bucket == "day" else since.replace(minute=0, second=0)
    out = []
    while cur <= until:
        out.append(cur)
        cur += step
    return out


def summary(s: Session, *, days: int = 7, bucket: str = "day", camera: str | None = None, district: str | None = None, now: datetime | None = None, include_demo: bool = True) -> dict:
    if bucket not in BUCKETS:
        raise AlertError(f"bucket ∈ {', '.join(BUCKETS)}")
    if not 1 <= days <= MAX_DAYS:
        raise AlertError(f"days ∈ [1, {MAX_DAYS}]")
    if bucket == "hour" and days > 14:
        raise AlertError("почасовая разбивка — не больше 14 дней")
    until = now or clock.now()
    since = (until - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0)
    q = select(Alert.id, Alert.t_abs, Alert.status, Alert.camera_id, Alert.district, Alert.cam_index, Alert.confidence).where(Alert.t_abs >= since, Alert.t_abs <= until + timedelta(days=1))
    if camera:
        q = q.where(Alert.camera_id == camera)
    if district:
        q = q.where(Alert.district == district)
    if not include_demo:
        q = q.where(Alert.demo.is_(False))
    rows = s.execute(q.order_by(Alert.t_abs).limit(MAX_ROWS + 1)).all()
    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]

    zero = lambda: {st: 0 for st in ALERT_STATUSES}                                                  # noqa: E731
    end = max([until] + [r[1] for r in rows])          # часы камеры могут уйти вперёд: такие тревоги не должны пропадать из ряда
    series = {_bucket_key(b, bucket): zero() for b in _bucket_starts(since, end, bucket)}
    cams: dict[str, dict] = defaultdict(lambda: dict(zero(), total=0, conf_sum=0.0, district=None, cam_index=None))
    hours = [0] * 24
    totals = zero()
    conf_sum = 0.0
    for _id, t, st, cam, dist, idx, conf in rows:
        totals[st] += 1
        conf_sum += conf or 0.0
        k = _bucket_key(t, bucket)
        if k in series:
            series[k][st] += 1
        c = cams[cam]
        c[st] += 1
        c["total"] += 1
        c["conf_sum"] += conf or 0.0
        c["district"], c["cam_index"] = dist, idx
        hours[t.hour] += 1

    ids_false = [r[0] for r in rows if r[2] == "false"]
    reasons: Counter = Counter()
    if ids_false:
        for chunk_start in range(0, len(ids_false), 500):
            part = ids_false[chunk_start:chunk_start + 500]
            latest = select(func.max(AlertReview.id)).where(AlertReview.alert_id.in_(part)).group_by(AlertReview.alert_id)
            for reason, n in s.execute(select(AlertReview.reason, func.count()).where(AlertReview.id.in_(latest)).group_by(AlertReview.reason)).all():
                reasons[reason or "—"] += n

    total = sum(totals.values())
    decided = totals["confirmed"] + totals["false"]
    return dict(
        since=clock.fmt(since), until=clock.fmt(end), days=days, bucket=bucket, truncated=truncated,
        totals=dict(totals, total=total, precision=(totals["confirmed"] / decided) if decided else None, mean_confidence=(conf_sum / total) if total else None),
        series=[dict(t=k, total=sum(v.values()), **v) for k, v in series.items()],
        reasons=[dict(reason=r, n=n) for r, n in reasons.most_common()],
        cameras=sorted((dict(camera_id=cid, district=c["district"], cam_index=c["cam_index"], total=c["total"], confirmed=c["confirmed"], false=c["false"], unsure=c["unsure"], pending=c["new"],
                             mean_confidence=(c["conf_sum"] / c["total"]) if c["total"] else None) for cid, c in cams.items()), key=lambda x: -x["total"]),
        hours=hours,
    )
