"""Вкладка «Конструктор»: гибкая правка архитектуры конвейера с видом схемы и ценой каждой правки.

Слева — этапы конвейера (поза и трекинг → ряды и автомат → признаки цикла → классификатор и слияние → события) с ключевыми параметрами и «все параметры»
таблицей (описания — из комментариев configs/default.yaml). Справа — живая сводка: что изменено, что придётся пересчитать и сколько это стоит,
проверка устройства и моделей, память, утечка обучения. Действия: оценить без сохранения (фоновая задача `sd eval -s …`), сохранить как профиль,
построить серию вариантов (сетка значений параметра поверх текущих правок).

Правки хранятся как отличия от выбранного базового профиля — файл профиля меняется только кнопкой «Сохранить».
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from sd import arch as A
from sd import catalog as CAT
from sd import gt as GT
from sd import library as LIB
from sd import profiles as PR
from sd import runner as RN
from sd import solver as SV
from sd import tasks as T
from sd.ui import common as UC

GROUPS = [
    ("Поза и трекинг", ["video.process_fps", "pose.weights", "pose.imgsz", "pose.runtime", "pose.device", "pose.conf", "pose.refine.enabled", "pose.refine.method",
                        "pose.refine.gate_s", "tracking.tracker", "tracking.track_buffer_sec"]),
    ("Ряды и автомат циклов", ["features.mouth_offset", "features.smooth_sec", "features.hand_extend", "cycles.th_in", "cycles.th_out", "cycles.hold_min_sec",
                               "cycles.hold_max_sec", "cycles.hold_cap_sec", "cycles.emit_long_hold"]),
    ("Признаки цикла", ["options.objects", "evidence.runtime", "evidence.device", "evidence.conf", "options.photo_bundle", "options.vlm_model", "options.vlm_mode"]),
    ("Классификатор и слияние", ["options.cycle_bundle", "options.fusion"]),
    ("События", ["events.cycle_th", "events.confidence_threshold", "events.merge_gap_sec", "events.window_sec", "events.window_mode", "video.min_person_height_px"]),
]
CURATED = {k for _, ks in GROUPS for k in ks}
NAMES = {   # подписи ключевых параметров (ключ конфигурации — в подсказке)
    "video.process_fps": "Частота обработки, к/с", "pose.weights": "Модель позы", "pose.imgsz": "Вход позы, px", "pose.runtime": "Рантайм позы", "pose.device": "Устройство позы",
    "pose.conf": "Порог детекции человека", "pose.refine.enabled": "Уточнять точки по кропу", "pose.refine.method": "Метод уточнения",
    "pose.refine.gate_s": "Уточнять только «руку у лица» (0 — всех)", "tracking.tracker": "Трекер", "tracking.track_buffer_sec": "Держать трек без детекций, с",
    "features.mouth_offset": "Смещение рта от носа", "features.smooth_sec": "Сглаживание d_t, с", "features.hand_extend": "Продление кисти от запястья",
    "cycles.th_in": "Вход «у рта» (ширины плеч)", "cycles.th_out": "Выход «от рта» (ширины плеч)", "cycles.hold_min_sec": "Пауза у рта: минимум, с",
    "cycles.hold_max_sec": "Пауза у рта: максимум, с", "cycles.hold_cap_sec": "Отбрасывать паузы длиннее, с", "cycles.emit_long_hold": "Оставлять длинные паузы кандидатами",
    "options.objects": "Детекторы предмета", "evidence.runtime": "Рантайм детектора", "evidence.device": "Устройство детектора", "evidence.conf": "Порог детекции предмета",
    "options.photo_bundle": "Фото-модель", "options.vlm_model": "VLM (Ollama)", "options.vlm_mode": "Режим VLM", "options.cycle_bundle": "Классификатор цикла",
    "options.fusion": "Слияние сигналов", "events.cycle_th": "Порог оценки цикла", "events.confidence_threshold": "Порог события",
    "events.merge_gap_sec": "Склейка циклов, с", "events.window_sec": "Окно «два цикла», с", "events.window_mode": "Правило окна", "video.min_person_height_px": "Минимальный рост человека, px",
}
IMGSZ = [480, 640, 800, 960, 1120, 1280, 1440, 1600, 1920]
LEVEL_UI = {-1: st.success, 0: st.success, 1: st.info, 2: st.info, 3: st.warning, 4: st.warning}


@st.cache_data(show_spinner=False)
def _defaults(mtime: float) -> pd.DataFrame:
    return A.parse_defaults()


def _base_values(base: PR.Profile) -> dict:
    vals = dict(A.effective(base))
    vals.update({f"options.{k}": v for k, v in base.opts().items()})
    return vals


def _choices(key: str, cur, vals: dict) -> list | None:
    if key == "pose.weights":
        return list(dict.fromkeys([c.id for c in CAT.pose_models()] + [cur]))
    if key == "pose.runtime":
        return list(CAT.RUNTIMES)
    if key == "pose.device":
        rt = st.session_state.get("ds_pose.runtime", vals.get("pose.runtime"))
        return list(dict.fromkeys(CAT.RUNTIMES.get(rt, []) + [cur]))
    if key == "pose.refine.method":
        return list(CAT.REFINE_METHODS)
    if key == "tracking.tracker":
        return list(CAT.TRACKERS)
    if key == "evidence.runtime":
        return ["torch", "openvino"]
    if key == "evidence.device":
        return list(dict.fromkeys(["cpu", "0", "intel:gpu", "intel:cpu", str(cur)]))
    if key == "events.window_mode":
        return ["contain", "start_gap"]
    if key == "options.vlm_mode":
        return list(CAT.VLM_MODES)
    if key == "options.cycle_bundle":
        return list(dict.fromkeys([c.id for c in CAT.cycle_bundles()] + ([cur] if cur else [])))
    if key == "options.photo_bundle":
        return [None, *dict.fromkeys([c.id for c in CAT.photo_bundles()] + ([cur] if cur else []))]
    if key == "options.vlm_model":
        return [None, *dict.fromkeys([c.id for c in CAT.vlm_models()] + ([cur] if cur else []))]
    return None


def _reset(base: PR.Profile, vals: dict | None = None) -> None:
    for k in [k for k in st.session_state if str(k).startswith("ds_")]:
        del st.session_state[k]
    st.session_state["ds_ident"] = base.name
    if vals:
        st.session_state["ds_load"] = vals


def _widget(key: str, cur, desc: str, vals: dict):
    """Виджет параметра по типу значения и известным вариантам; возвращает выбранное значение."""
    wk = f"ds_{key}"
    preload = st.session_state.get("ds_load") or {}
    if wk not in st.session_state:
        st.session_state[wk] = preload.get(key, cur) if key != "options.fusion" else None
    label = NAMES.get(key, key)
    help_ = f"`{key}` — {desc}" if desc else f"`{key}`"
    ch = _choices(key, cur, vals)
    if key == "options.objects":
        opts = list(dict.fromkeys([d.id for d in CAT.detector_models()] + list(cur or [])))
        st.session_state[wk] = [x for x in (st.session_state[wk] or []) if x in opts]
        return st.multiselect(label, opts, key=wk, help="`options.objects` — детекторы сигареты на кропе рта; пусто — без признаков предмета")
    if key == "options.fusion":
        f = preload.get(key, cur) or {}
        w = f.get("weights", {}) if isinstance(f, dict) else {}
        c = st.columns(4)
        ws = {n: c[i].number_input(f"вес {n}", 0.0, 5.0, float(w.get(n, 0.0)), 0.1, key=f"ds_fz_{n}") for i, n in enumerate(("vlm", "object", "photo"))}
        clip = c[3].number_input("предел поправки", 0.5, 10.0, float((f or {}).get("clip_logit", 2.0)), 0.1, key="ds_fz_clip")
        st.caption("Слияние: поправка к логиту оценки классификатора по сигналам VLM/предмета/фото; все веса 0 — выключено.")
        return dict(weights={n: v for n, v in ws.items() if v > 0}, clip_logit=float(clip)) if any(v > 0 for v in ws.values()) else None
    if key == "pose.imgsz":
        opts = sorted(set(IMGSZ) | {int(st.session_state[wk])})
        return st.select_slider(label, opts, key=wk, help=help_)
    if ch is not None:
        if st.session_state[wk] not in ch:
            st.session_state[wk] = ch[0]
        fmt = (lambda x: "— нет —" if x is None else Path(str(x)).name) if key.startswith("options.") else str
        return st.selectbox(label, ch, key=wk, help=help_, format_func=fmt)
    if isinstance(cur, bool):
        return st.toggle(label, key=wk, help=help_)
    if isinstance(cur, int):
        st.session_state[wk] = int(st.session_state[wk])
        return int(st.number_input(label, step=1, key=wk, help=help_))
    if isinstance(cur, float):
        st.session_state[wk] = float(st.session_state[wk])
        return float(st.number_input(label, step=0.01 if abs(cur) < 1 else 0.1, format="%.2f" if abs(cur) >= 0.1 or cur == 0 else "%.3f", key=wk, help=help_))
    return st.text_input(label, key=wk, help=help_)


def _yaml(v) -> str:
    return json.dumps(v, ensure_ascii=False)


def _all_params(df: pd.DataFrame, vals: dict) -> dict:
    """Таблица всех параметров вне ключевых: правка значения в YAML. Возвращает {ключ: новое значение} для изменённых строк."""
    rest = df[~df.key.isin(CURATED)].copy()
    rest["этап"] = rest.stage.map(lambda s: A.STAGES[s][0])
    rest["в базе"] = [_yaml(vals.get(k)) for k in rest.key]
    rest["новое"] = rest["в базе"]
    preload = st.session_state.get("ds_load") or {}
    if preload:
        rest["новое"] = [_yaml(preload[k]) if k in preload else n for k, n in zip(rest.key, rest["новое"])]
    q = st.text_input("Поиск параметра", key="ds_q", placeholder="например stitch, smooth, crop")
    view = rest[rest.key.str.contains(q, case=False) | rest.comment.str.contains(q, case=False)] if q else rest
    ed = st.data_editor(view[["этап", "key", "в базе", "новое", "comment"]], hide_index=True, width="stretch", height=380, key="ds_table",
                        disabled=["этап", "key", "в базе", "comment"],
                        column_config={"key": st.column_config.TextColumn("параметр", width="medium"), "новое": st.column_config.TextColumn("новое значение (JSON/YAML)", width="medium"),
                                       "comment": st.column_config.TextColumn("описание", width="large")})
    out, bad = {}, []
    for r in ed.to_dict("records"):
        if r["новое"] != r["в базе"]:
            try:
                out[r["key"]] = yaml.safe_load(r["новое"])
            except yaml.YAMLError:
                bad.append(r["key"])
    for b in bad:
        st.error(f"{b}: значение не разбирается как YAML/JSON")
    return out


def _estimate_minutes(dirs: list[str]) -> float | None:
    from sd import data_catalog as DC

    try:
        vids = LIB.videos_in(dirs)
        pr = DC.probe_cached(vids)
        return sum((p.get("duration") or 0) for p in pr.values()) / 60
    except Exception:       # noqa: BLE001
        return None


def render() -> None:
    UC.style()
    names = PR.list_profiles()
    if not names:
        UC.header("Конструктор", "Нет сохранённых профилей: установите решатель («Выпуск») или положите YAML в configs/experiments.")
        return
    head = st.empty()                    # заголовок с метками и схема заполняются после редактора (зависят от правок)
    c = st.columns([2, 3, 2], vertical_alignment="bottom")
    bname = c[0].selectbox("Базовый профиль", names, index=names.index("mvp") if "mvp" in names else 0, key="ds_base_pick")
    base = PR.load_named(bname)
    if st.session_state.get("ds_ident") != base.name:
        _reset(base)
        st.rerun()
    runs = UC.runs_df()
    rlab = {r.id: UC.run_label(r) for r in runs.itertuples()} if len(runs) else {}
    with c[1]:
        rid = st.selectbox("Взять параметры из прогона журнала (необязательно)", [None, *rlab], key="ds_from_run", format_func=lambda i: "—" if i is None else rlab[i])
    if c[2].button("Применить параметры прогона", disabled=rid is None, help="подставить конфигурацию прогона как правки поверх базового профиля"):
        from sd import experiments as XP

        _, meta, _, _ = XP.load_run(rid)
        pr = meta.get("profile") or {}
        p = PR.Profile(bname, "", pr.get("base"), pr.get("config") or {}, pr.get("options") or {})
        _reset(base, _base_values(p))
        st.rerun()
    vals = _base_values(base)
    df = _defaults(UC.file_mtime(Path(A.CONFIGS) / "default.yaml"))
    desc = dict(zip(df.key, df.comment))
    desc.update({"options.objects": "детекторы сигареты на кропе кисть–рот", "options.photo_bundle": "пакет фото-модели: zero-shot CLIP и photo_* по кропам рта",
                 "options.vlm_model": "тег модели Ollama; пусто — VLM выключен", "options.vlm_mode": "grey — только серая зона оценки; all — все циклы",
                 "options.cycle_bundle": "пакет классификатора цикла (обязателен)"})

    graph = st.container(border=True)
    left, right = st.columns([3, 2], gap="large")
    edits: dict = {}
    with left:
        tabs = st.tabs([g for g, _ in GROUPS] + ["Все параметры"])
        for (gname, keys), tab in zip(GROUPS, tabs):
            with tab:
                cols = st.columns(2)
                for i, k in enumerate(keys):
                    with cols[i % 2] if k not in ("options.fusion", "options.objects") else st.container():
                        v = _widget(k, vals.get(k), desc.get(k, ""), vals)
                    if _yaml(v) != _yaml(vals.get(k)):
                        edits[k] = v
        with tabs[-1]:
            edits.update(_all_params(df, vals))
    st.session_state.pop("ds_load", None)

    sets = [f"{k}={_yaml(v)}" for k, v in edits.items()]
    new, err = base, None
    try:
        new = PR.with_overrides(base, sets)
    except Exception as e:      # noqa: BLE001
        err = str(e)
    dd = A.diff(base, new) if not err else pd.DataFrame(columns=["key", "было", "стало", "этап"])
    imp = A.impact(dd.key.tolist())
    chips = [(f"база: {base.name}", ""), (f"правок: {len(dd)}", "warn" if len(dd) else "ok"), (f"отпечаток {new.fingerprint() if not err else '—'}", "")]
    with head.container():
        UC.header("Конструктор", "Схема конвейера и правка любого его параметра с ценой правки; оценка без сохранения, сохранение профиля, серия вариантов.", chips)
    with graph:
        st.graphviz_chart(A.dot(new, set(imp.get("stages", [])), set(imp.get("recompute", []))), width="stretch")
        st.caption("Оранжевые — изменённые этапы, пунктир — пересчитываемые из-за правок, серые — выключенные. Описание каждого параметра — в подсказке «?».")
    with right:
        st.markdown("##### Сводка правок")
        if err:
            st.error(f"правки не применяются: {err}")
            return
        LEVEL_UI[imp["level"]](imp["text"])
        if len(dd):
            st.dataframe(dd.assign(было=dd["было"].map(_yaml), стало=dd["стало"].map(_yaml)).rename(columns={"key": "параметр"}), hide_index=True, width="stretch")
        items = []
        for i in SV.check(new, devices=True):
            items.append(("error" if i.level == "error" else "warning", i.text))
        try:
            arch = SV.detect_architecture(new)
            r = arch.get("resources_estimate") or SV.estimate_resources(arch)
            items.append(("info", f"память (оценка): пик {r.get('peak_gb')} ГБ, при последовательной загрузке ≈ {r.get('sequential_gb')} ГБ"))
        except Exception:       # noqa: BLE001
            pass
        gt = GT.load()
        labeled = sorted(set(gt.clip_id)) if len(gt) else []
        lk = RN.training_leak(new, labeled)
        if lk["clips"]:
            items.append(("warning", f"классификатор обучен на {len(lk['clips'])} из {len(labeled)} размеченных клипов — оценка на них оптимистична"))
        if items:
            UC.checklist(items)

        st.markdown("##### Действия")
        dirs = LIB.discover_dirs()
        dnames = dirs.name.tolist() if len(dirs) else []
        if "ds_dirs" not in st.session_state:
            st.session_state["ds_dirs"] = [n for n in dnames if Path(n).name in ("курение", "лжекурение")] or dnames[:1]
        pick = st.multiselect("Папки для оценки", dnames, key="ds_dirs")
        paths = [str(p) for p in dirs.set_index("name").loc[pick, "path"]] if pick else []
        mins = _estimate_minutes(paths) if imp["level"] >= 4 and paths else None
        if mins:
            st.caption(f"Пересчёт позы: видео {mins:.0f} мин; на этом ПК поза ≈ 0.7–1.3× реального времени → порядка {mins / 0.9:.0f} мин.")
        blocked = any(lvl == "error" for lvl, _ in items) or not paths
        t1, t2, t3 = st.tabs(["Оценить", "Сохранить профиль", "Серия вариантов"])
        with t1:
            st.caption("Фоновая задача `sd eval` с правками `-s` — файл профиля не меняется; результат — в журнале и «Аналитике».")
            UC.task_button("eval", dict(dirs="\n".join(paths), profile=base.name, sets="\n".join(sets), policy="fixed"), "▶ Оценить эту конфигурацию", key="ds_eval",
                           disabled=blocked)
        with t2:
            nm = st.text_input("Имя профиля", f"{base.name}_v2" if len(dd) else base.name, key="ds_name")
            dsc = st.text_area("Описание", base.description, key="ds_desc", height=90)
            ok_name = bool(nm.strip()) and all(ch.isalnum() or ch in "-_" for ch in nm.strip())
            cfg_e = {k: v for k, v in edits.items() if not k.startswith("options.")}
            opt_e = {k.split(".", 1)[1]: v for k, v in edits.items() if k.startswith("options.")}
            target = PR.Profile(nm.strip(), dsc, base.base, {**base.config, **cfg_e}, {**base.options, **opt_e})

            def _save():
                PR.save(target)
                return f"сохранён configs/experiments/{nm.strip()}.yaml, отпечаток {target.fingerprint()}"

            if nm.strip() in names:
                UC.confirm("Перезаписать профиль", "ds_save", f"Профиль «{nm.strip()}» будет перезаписан: прежние оценки в журнале перестанут соответствовать файлу.", _save,
                           disabled=not ok_name or (nm.strip() == base.name and dd.empty), help="правок нет — перезаписывать нечего" if dd.empty else None)
            elif st.button("Сохранить новый профиль", type="primary", disabled=not ok_name, key="ds_save_new"):
                st.success(_save())
            if not ok_name:
                st.caption("Имя: буквы, цифры, - и _.")
        with t3:
            allk = sorted(set(df.key) | CURATED)
            param = st.selectbox("Параметр", allk, key="ds_var_param", index=allk.index("video.process_fps") if "video.process_fps" in allk else 0)
            raw = st.text_input("Значения через «;» (YAML/JSON)", key="ds_var_vals", placeholder="например 5; 8; 10")
            values = [x.strip() for x in raw.split(";") if x.strip()]
            rows = []
            plain = lambda k: k.split(".", 1)[1] if k.startswith("options.") else k     # noqa: E731 — в строках серии опции профиля пишутся без «options.»
            for v in values:
                nm_v = f"{param.split('.')[-1]}={v}"
                rows += [dict(name=nm_v, param=plain(k), value=_yaml(val)) for k, val in edits.items() if k != param]
                rows.append(dict(name=nm_v, param=plain(param), value=v))
            if values:
                st.caption(f"Вариантов: {len(values)} + базовый профиль; в каждом — текущие правки ({len(edits)}) и своё значение «{param}». "
                           f"Цена: {A.impact([param]).get('text')}")
            if st.button("Запустить серию (фон)", disabled=not values or blocked, key="ds_var_go"):
                f = T.inputs_dir() / f"variants_{time.strftime('%Y%m%d_%H%M%S')}.json"
                f.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
                tid = T.start("variants", dict(dirs="\n".join(paths), base=base.name, rows=str(f), policy="fixed"))
                st.toast(f"серия: {T.status(tid)['state']}", icon="⏳")
        UC.task_panel(("eval", "variants"), key="ds_tp", limit=3, title="Задачи")
