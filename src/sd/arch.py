"""Архитектура конвейера для конструктора Dev Studio: этапы, их параметры с описаниями, что пересчитывается при правке, схема-граф.

Описания параметров берутся из комментариев `configs/default.yaml` (единственный источник правды — тот же файл, что читает конвейер). Цена правки выводится
из того, как этапы кэшируются (`stages.run_id_for`, `_feat_hash`, `_cyc_hash`, `_ev_hash`, кэш `enrich` по циклам):
    поза и трекинг — ключ запуска позы (video.process_fps, pose.*, tracking.*) → пересчёт позы: дольше всего (≈ длительность видео и больше)
    признаки ряда / автомат — хэш секций features / cycles → новые циклы → признаки предмета и фото по ним тоже заново (секунды на цикл)
    события — хэш events + минимальный рост → секунды
    предмет / фото / VLM — подпись детекторов, пакета фото-модели, модели VLM → секунды на цикл
    классификатор цикла — только оценка циклов → секунды
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .paths import CONFIGS

# этап → (подпись, уровень цены 0..4, пояснение)
STAGES = {
    "pose": ("Поза и трекинг", 4, "пересчёт позы по всем клипам — самое долгое (≈ длительность видео и больше)"),
    "series": ("Ряды d_t", 2, "новые ряды → новые циклы → признаки предмета/фото по ним заново"),
    "cycles": ("Автомат циклов", 2, "новые циклы → признаки предмета/фото по ним заново (секунды на цикл)"),
    "evidence": ("Признаки предмета", 2, "детектор на кропе рта для каждого цикла (≈ 1.8 с на цикл на iGPU)"),
    "photo": ("Фото-признак", 1, "CLIP по кропам рта (≈ 0.7 с на цикл)"),
    "vlm": ("VLM", 3, "секунды на цикл (≈ 11 с); только серая зона или все циклы"),
    "tube": ("Видео-модели", 3, "X-CLIP / VideoMAE по «трубке» человека"),
    "classifier": ("Классификатор цикла", 0, "только оценка циклов — секунды"),
    "fusion": ("Слияние сигналов", 0, "поправка оценки — секунды"),
    "events": ("Правила события", 0, "сборка событий — секунды"),
    "render": ("Отрисовка", 0, "только видео с разметкой"),
    "project": ("Проект", 0, "служебное"),
}
COST = {0: "секунды", 1: "минуты", 2: "минуты–десятки минут", 3: "долго (секунды на цикл)", 4: "долго (поза заново)"}


def stage_of(key: str) -> str:
    """Этап, к которому относится ключ конфигурации (`pose.imgsz`) или опция профиля (`options.objects`)."""
    k = key.removeprefix("cfg.")
    if k.startswith(("options.", "opt.")):
        o = k.split(".", 1)[1]
        return {"cycle_bundle": "classifier", "cycle_bundle_full": "vlm", "photo_bundle": "photo", "objects": "evidence", "vlm_model": "vlm", "vlm_mode": "vlm", "grey": "vlm",
                "fusion": "fusion", "backend": "photo", "allow_heuristic": "classifier"}.get(o.split(".")[0], "classifier")
    sec = k.split(".")[0]
    if k in ("video.min_person_height_px",):
        return "events"
    if k in ("video.render_max_width",):
        return "render"
    return {"video": "pose", "pose": "pose", "tracking": "pose", "features": "series", "cycles": "cycles", "events": "events", "evidence": "evidence", "tube": "tube",
            "vlm": "vlm", "render": "render", "project": "project"}.get(sec, "events")


def impact(keys: list[str]) -> dict:
    """Что пересчитается при изменении этих ключей: самый дорогой затронутый этап, список этапов и словесная цена."""
    st = sorted({stage_of(k) for k in keys}, key=lambda s: -STAGES[s][1])
    if not st:
        return dict(stages=[], level=-1, text="изменений нет — прогон возьмётся из кэша")
    lvl = STAGES[st[0]][1]
    chain = st[0]
    downstream = {"pose": ["series", "cycles", "evidence", "photo", "classifier", "events"], "series": ["cycles", "evidence", "photo", "classifier", "events"],
                  "cycles": ["evidence", "photo", "classifier", "events"], "evidence": ["classifier", "events"], "photo": ["classifier", "events"], "vlm": ["classifier", "events"],
                  "classifier": ["events"], "fusion": ["events"]}.get(chain, [])
    return dict(stages=st, recompute=[chain, *downstream], level=lvl, text=f"{STAGES[chain][0]}: {STAGES[chain][2]}; цена — {COST[lvl]}")


@dataclass
class Param:
    key: str
    value: object
    comment: str
    stage: str


def parse_defaults(path: Path | None = None) -> pd.DataFrame:
    """Все параметры `configs/default.yaml` с комментариями: key (через точку), default, comment, section_comment, stage."""
    import yaml

    f = path or (CONFIGS / "default.yaml")
    text = f.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    comments: dict[str, str] = {}
    sec_comments: dict[str, list[str]] = {}
    stack: list[tuple[int, str]] = []
    pending: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            pending.append(line.strip().lstrip("#").strip())
            continue
        m = re.match(r"^(\s*)([A-Za-z_][\w]*)\s*:(.*)$", line)
        if not m:
            continue
        ind, key, rest = len(m.group(1)), m.group(2), m.group(3)
        while stack and stack[-1][0] >= ind:
            stack.pop()
        path_ = ".".join([s for _, s in stack] + [key])
        cm = rest.split("#", 1)[1].strip() if "#" in rest else ""
        if cm:
            comments[path_] = cm
        if pending:
            sec_comments.setdefault(path_, []).extend(pending)
            pending = []
        stack.append((ind, key))

    rows = []

    def walk(d: dict, prefix: str = "") -> None:
        for k, v in d.items():
            p = f"{prefix}{k}"
            if isinstance(v, dict):
                walk(v, p + ".")
            else:
                rows.append(dict(key=p, default=v, comment=comments.get(p, ""), note=" ".join(sec_comments.get(p, [])), stage=stage_of(p)))

    walk(data)
    df = pd.DataFrame(rows)
    df["section"] = df.key.str.split(".").str[0]
    df["parent_comment"] = [comments.get(k.rsplit(".", 1)[0], "") if "." in k else "" for k in df.key]
    return df


def effective(profile) -> dict:
    """Действующие значения профиля: {key: value} по всем параметрам конфигурации."""
    from .compare import _flat

    return {k: v for k, v in _flat(profile.cfg().to_dict()).items()}


def diff(base, new) -> pd.DataFrame:
    """Отличия двух профилей по действующей конфигурации и опциям: key, было, стало, этап."""
    a, b = effective(base), effective(new)
    rows = [dict(key=k, было=a.get(k), стало=v, этап=STAGES[stage_of(k)][0]) for k, v in b.items() if str(a.get(k)) != str(v)]
    oa, ob = base.opts(), new.opts()
    rows += [dict(key=f"options.{k}", было=oa.get(k), стало=v, этап=STAGES[stage_of('options.' + k)][0]) for k, v in ob.items() if str(oa.get(k)) != str(v)]
    return pd.DataFrame(rows, columns=["key", "было", "стало", "этап"])


def dot(profile, changed: set[str] | None = None, recompute: set[str] | None = None) -> str:
    """Схема конвейера (Graphviz DOT) с моделями этапов: изменённые этапы — оранжевые, пересчитываемые — с пунктиром, выключенные — серые."""
    c, o = profile.cfg(), profile.opts()
    changed, recompute = changed or set(), recompute or set()
    p, ref = c["pose"], c["pose"]["refine"]

    def node(nid: str, title: str, sub: list, stages: tuple, enabled: bool = True) -> str:
        ch = any(s in changed for s in stages)
        rc = any(s in recompute for s in stages)
        fill = "#fde7c7" if ch else ("#e6f6f3" if enabled else "#f3f4f6")
        color = "#b45309" if ch else ("#0f766e" if enabled else "#9ca3af")
        style = "rounded,filled" + (",dashed" if (rc and not ch) or not enabled else "")
        esc = lambda x: str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")      # noqa: E731
        lines_ = "".join(f'<br/><font point-size="12">{esc(x)}</font>' for x in sub if x)
        return f'{nid} [label=<<b>{title}</b>{lines_}>, shape=box, style="{style}", fillcolor="{fill}", color="{color}", fontname="Helvetica", fontsize=15, margin="0.12,0.06"];'

    objs = o.get("objects") or []
    vlm_on = bool(o.get("vlm_model")) and o.get("vlm_mode") != "off"
    fus = o.get("fusion")
    name = lambda x: Path(str(x)).name if x else None      # noqa: E731
    pose_sub = [f"{p['weights']} @{p['imgsz']}", f"{c['video']['process_fps']:g} к/с · {p['runtime']}", f"трек: {c['tracking']['tracker']}",
                (f"уточн.: {ref['method']}" + (f" (рука у лица)" if ref.get("gate_s") else "")) if ref["enabled"] else "без уточнения"]
    clf_sub = [name(o.get("cycle_bundle")) or "НЕ ЗАДАН", ("слияние: " + ", ".join(f"{k}×{v}" for k, v in (fus or {}).get("weights", {}).items())) if fus else ""]
    lines = ["digraph G {", 'rankdir=LR; bgcolor=transparent; nodesep=0.18; ranksep=0.3; edge [color="#6b7280", arrowsize=0.7];',
             node("pose", "Поза и трек", pose_sub, ("pose",)),
             node("ser", "Ряды d_t", [f"рот +{c['features']['mouth_offset']}", f"сглаж. {c['features']['smooth_sec']} с"], ("series",)),
             node("cyc", "Автомат циклов", [f"вход {c['cycles']['th_in']} / выход {c['cycles']['th_out']}", f"пауза {c['cycles']['hold_min_sec']}–{c['cycles']['hold_max_sec']} с"],
                  ("cycles",)),
             node("obj", "Предмет", [*objs[:2], f"+{len(objs) - 2}" if len(objs) > 2 else ""] if objs else ["не считается"], ("evidence",), bool(objs)),
             node("pho", "Фото-признак", [name(o.get("photo_bundle")) or "не считается"], ("photo",), bool(o.get("photo_bundle"))),
             node("vlm", "VLM", [str(o.get("vlm_model")), o.get("vlm_mode")] if vlm_on else ["выключен"], ("vlm",), vlm_on),
             node("clf", "Классификатор", clf_sub, ("classifier", "fusion"), bool(o.get("cycle_bundle"))),
             node("evt", "События", [f"цикл ≥ {c['events']['cycle_th']}", f"событие ≥ {c['events']['confidence_threshold']}", f"склейка {c['events']['merge_gap_sec']:g} с"],
                  ("events",)),
             "pose -> ser; ser -> cyc; cyc -> obj; cyc -> pho; cyc -> vlm [style=dashed]; obj -> clf; pho -> clf; vlm -> clf [style=dashed]; cyc -> clf; clf -> evt;", "}"]
    return "\n".join(lines)
