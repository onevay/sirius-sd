"""Версии файлов разметки: снимки перед любой массовой правкой, ручная фиксация версии с комментарием, просмотр отличий и откат.

Зачем: эталон событий и метки жестов — основа всех цифр качества. Раньше правка таблицы жестов делала копию в outputs/backups без описания, а импорт с заменой,
перестройка эталона (`sd gt-from-gestures`) и удаление интервалов не сохраняли ничего: ошибку разметки нельзя было ни найти, ни откатить, а «улучшение F1»
нельзя было отличить от «поправили эталон». Теперь:

    labels/history/<файл>/<время>_<отпечаток>.csv   — снимок файла (содержимое как было)
    labels/history/log.jsonl                         — журнал: когда, какой файл, отпечаток, строк, кто, зачем, действие

Снимок пишется, только если содержимое отличается от последнего снимка (повторные нажатия не плодят копии). Автоматические снимки «на всякий случай»
(action=auto) делаются не чаще раза в `AUTO_EVERY_SEC` на файл; перед массовыми операциями (import, rebuild, restore, table-save) — всегда.
Отпечаток содержимого (sha1 байтов, 10 знаков) не путать с `gt.fingerprint` (по смысловым колонкам): в журнале хранятся оба, чтобы прогон оценки
(meta.gt_fingerprint) можно было сопоставить с версией разметки.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import LABELS

HIST = LABELS / "history"
AUTO_EVERY_SEC = 600
MAX_AUTO = 200          # старые автоматические снимки одного файла удаляются сверх этого числа (ручные — никогда)


def files() -> dict[str, Path]:
    """Файлы разметки под версионированием: ключ → путь (пути читаются при вызове: тесты и SD_LABELS могут их подменить)."""
    from . import dataset as DS
    from . import gt as GT
    from . import start_eval as SE

    return {"events": GT.default_path(), "gestures": Path(SE.GT_CSV), "cycles": Path(DS.LABELS_CSV)}


TITLES = {"events": "эталон событий (events_gt.csv)", "gestures": "метки жестов (gesture_gt.csv)", "cycles": "метки циклов (cycle_labels.csv)"}


def _path(kind: str | Path) -> tuple[str, Path]:
    if isinstance(kind, Path):
        return kind.stem, kind
    f = files()
    if kind not in f:
        raise KeyError(f"неизвестный файл разметки {kind!r}; доступны: {', '.join(f)}")
    return kind, f[kind]


def content_sha(path: Path) -> str | None:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:10] if path.exists() else None


def semantic_fp(kind: str, path: Path) -> str | None:
    """Смысловой отпечаток (как в журнале экспериментов) — только для эталона событий."""
    if kind != "events" or not path.exists():
        return None
    from . import gt as GT

    df = GT.load(path)
    return GT.fingerprint(df) if len(df) else None


def _log(root: Path) -> Path:
    return root / "log.jsonl"


def journal(root: Path | None = None) -> pd.DataFrame:
    f = _log(root or HIST)
    rows = []
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    cols = ["ts", "kind", "file", "snapshot", "sha", "gt_fp", "rows", "author", "note", "action"]
    return pd.DataFrame(rows, columns=cols)


def versions(kind: str | Path, root: Path | None = None) -> pd.DataFrame:
    """Снимки одного файла (новые сверху), только существующие на диске."""
    j = journal(root)
    k, _ = _path(kind)
    j = j[j.kind == k]
    base = root or HIST
    j = j[[(base / s).exists() for s in j.snapshot]] if len(j) else j
    return j.iloc[::-1].reset_index(drop=True)


def _rows(path: Path) -> int:
    try:
        return int(len(pd.read_csv(path, encoding="utf-8-sig")))
    except Exception:
        return -1


def snapshot(kind: str | Path, author: str = "", note: str = "", action: str = "manual", root: Path | None = None, force: bool = False) -> dict | None:
    """Снимок текущего файла. None — файла нет, содержимое не менялось с прошлого снимка или (action=auto) прошлый снимок свежее AUTO_EVERY_SEC."""
    k, p = _path(kind)
    base = root or HIST
    sha = content_sha(p)
    if sha is None:
        return None
    prev = versions(kind, base)
    if len(prev) and not force:
        last = prev.iloc[0]
        if last.sha == sha:
            return None
        if action == "auto" and time.time() - time.mktime(time.strptime(last.ts, "%Y-%m-%dT%H:%M:%S")) < AUTO_EVERY_SEC:
            return None
    d = base / k
    d.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    snap = d / f"{ts.replace(':', '').replace('-', '')}_{sha}{p.suffix}"
    shutil.copy2(p, snap)
    rec = dict(ts=ts, kind=k, file=str(p.name), snapshot=snap.relative_to(base).as_posix(), sha=sha, gt_fp=semantic_fp(k, p), rows=_rows(p), author=author or "", note=note or "",
               action=action)
    with _log(base).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _prune(kind, base)
    return rec


def _prune(kind: str | Path, base: Path) -> None:
    v = versions(kind, base)
    auto = v[v.action == "auto"]
    for s in auto.snapshot.iloc[MAX_AUTO:]:
        (base / s).unlink(missing_ok=True)


def read_version(kind: str, snapshot_rel: str, root: Path | None = None) -> pd.DataFrame:
    return pd.read_csv((root or HIST) / snapshot_rel, encoding="utf-8-sig")


def _atomic_copy(src: Path, dst: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent), suffix=".tmp")
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        for attempt in range(8):
            try:
                os.replace(tmp, dst)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def restore(kind: str, snapshot_rel: str, author: str = "", root: Path | None = None) -> dict:
    """Вернуть файл к снимку. Текущее содержимое сначала само сохраняется снимком (откат тоже можно откатить)."""
    k, p = _path(kind)
    base = root or HIST
    src = base / snapshot_rel
    if not src.exists():
        raise FileNotFoundError(f"снимок не найден: {snapshot_rel}")
    before = snapshot(k, author, f"перед откатом к {snapshot_rel}", "before-restore", base, force=False)
    if k == "events":
        from . import gt as GT

        with GT.locked(p):
            _atomic_copy(src, p)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        _atomic_copy(src, p)
    after = snapshot(k, author, f"откат к {snapshot_rel}", "restore", base, force=True)
    return dict(before=before, after=after)


KEYS = {"events": ["clip_id", "person_gt_id", "start_sec", "end_sec", "label"], "gestures": ["video", "tid", "peak_t", "label"], "cycles": ["video", "tid", "peak_t", "label"]}
IDS = {"events": ["clip_id", "person_gt_id", "start_sec", "end_sec"], "gestures": ["video", "tid", "peak_t"], "cycles": ["video", "tid", "peak_t"]}


def _key(v) -> str:
    """Значение идентифицирующей колонки → строка для сравнения версий: 1, 1.0 и «1» совпадают, числа округляются до 0.01, пропуск — пустая строка."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v) in ("", "nan", "None"):
        return ""
    try:
        return f"{round(float(v), 2):g}"
    except (TypeError, ValueError):
        return str(v)


def diff(kind: str, old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Что изменилось между двумя версиями: добавлено, удалено, изменена метка (по идентифицирующим колонкам). Числа округляются до 0.01 с."""
    ids = IDS.get(kind, list(old.columns[:3]))
    lab = "label" if "label" in old.columns or "label" in new.columns else None

    def norm(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        for c in ids + ([lab] if lab else []):
            if c not in d:
                d[c] = None
        for c in ids:
            d[c] = d[c].map(_key)
        return d

    o, n = norm(old), norm(new)
    m = o.merge(n, on=ids, how="outer", suffixes=("_old", "_new"), indicator=True)
    added = m[m._merge == "right_only"]
    removed = m[m._merge == "left_only"]
    both = m[m._merge == "both"]
    changed = both[both[f"{lab}_old"].astype(str) != both[f"{lab}_new"].astype(str)] if lab else both.iloc[0:0]
    cols = ids + ([f"{lab}_old", f"{lab}_new"] if lab else [])
    return dict(added=added[cols].reset_index(drop=True), removed=removed[cols].reset_index(drop=True), changed=changed[cols].reset_index(drop=True),
                summary=f"добавлено {len(added)}, удалено {len(removed)}, изменена метка {len(changed)}")
