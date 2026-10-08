"""Редактор профиля моделей: любой доступный компонент выбирается из списка (каталог `catalog`) или вводится вручную; результат — `profiles.Profile`."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from sd import catalog as CAT
from sd.profiles import DEFAULT_OPTIONS, Profile

NONE = "— нет —"
IMGSZ = [480, 640, 800, 960, 1088, 1280, 1600, 1920]


def _pick(label: str, options: list[str], current: str | None, key: str, fmt=None, help: str | None = None) -> str:
    opts = list(options)
    if current is not None and current not in opts:
        opts.append(current)
    return st.selectbox(label, opts, index=opts.index(current) if current in opts else 0, key=key, format_func=fmt or str, help=help)


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k_, v in d.items():
        kk = f"{prefix}{k_}"
        if isinstance(v, dict):
            out.update(_flatten(v, kk + "."))
        elif isinstance(v, (bool, int, float, str)) or v is None:
            out[kk] = v
    return out


def _parse(txt: str, like):
    """Строка из таблицы → значение того же типа, что и у образца (bool/int/float/str)."""
    t = str(txt).strip()
    if isinstance(like, bool):
        if t.lower() in ("true", "1", "да", "yes"):
            return True
        if t.lower() in ("false", "0", "нет", "no"):
            return False
        raise ValueError(f"ожидалось true/false, получено {t!r}")
    if isinstance(like, int):
        return int(float(t))
    if isinstance(like, float):
        return float(t)
    return t


def _all_params(base: Profile, conf: dict, k: str) -> dict:
    """Таблица ВСЕХ скалярных параметров конфигурации (configs/default.yaml): любое значение можно изменить; изменённые уходят в профиль как переопределения."""
    from sd.config import load_config

    default = _flatten(load_config(None).to_dict())
    cur = _flatten(base.cfg().to_dict())
    cur.update({a: b for a, b in conf.items() if a in default})          # то, что уже выбрано выше в этом же редакторе
    over: dict = {}
    with st.expander("Все параметры конфигурации (поза, трекинг, признаки, автомат, предмет, трубка, VLM, отрисовка …)"):
        q = st.text_input("Фильтр по имени", "", key=k + "pfilter", placeholder="например cycles. или evidence.")
        rows = [dict(параметр=a, значение="" if v is None else str(v), по_умолчанию="" if default[a] is None else str(default[a])) for a, v in cur.items() if q.lower() in a.lower()]
        st.caption("Редактируйте колонку «значение»; отличие от «по умолчанию» сохранится в профиле. Неверный тип значения отклоняется.")
        ed = st.data_editor(pd.DataFrame(rows), hide_index=True, use_container_width=True, disabled=["параметр", "по_умолчанию"], key=k + "ptable", height=360)
        for _, r in ed.iterrows():
            a = r["параметр"]
            if str(r["значение"]) == ("" if cur[a] is None else str(cur[a])):
                continue
            try:
                over[a] = _parse(r["значение"], default[a] if default[a] is not None else cur[a])
            except ValueError as e:
                st.error(f"{a}: {e}")
    return {a: b for a, b in over.items() if b != cur.get(a)}


def editor(base: Profile, key: str = "pe", classifier: str | None = None) -> Profile:
    """`classifier` — выбранный снаружи классификатор цикла (страница «Оценка»): тогда выбор здесь не показывается, чтобы не было двух источников правды."""
    cfg, o = base.cfg(), base.opts()
    k = f"{key}_{base.name}_"
    conf = dict(base.config)
    opt = dict(base.options)

    st.markdown("**Поза и трекинг**")
    pose = CAT.pose_models()
    ids = [p.id for p in pose]
    weights = _pick("Модель позы", ids, cfg["pose"]["weights"], k + "pose", fmt=lambda i: next((p.label for p in pose if p.id == i), f"{i} (нет весов)"))
    runtime = st.radio("Рантайм", list(CAT.RUNTIMES), index=list(CAT.RUNTIMES).index(cfg["pose"]["runtime"]), horizontal=True, key=k + "rt")
    devs = CAT.RUNTIMES[runtime]
    device = st.selectbox("Устройство", devs, index=devs.index(cfg["pose"]["device"]) if cfg["pose"]["device"] in devs else 0, key=k + "dev")
    imgs = sorted(set(IMGSZ) | {int(cfg["pose"]["imgsz"])})
    imgsz = st.select_slider("imgsz (длинная сторона)", imgs, value=int(cfg["pose"]["imgsz"]), key=k + "imgsz")
    refine = st.selectbox("Уточнение точек по кропу", ["выкл", *CAT.REFINE_METHODS], index=(1 + CAT.REFINE_METHODS.index(cfg["pose"]["refine"]["method"])
                                                                                          if cfg["pose"]["refine"]["enabled"] and cfg["pose"]["refine"]["method"] in CAT.REFINE_METHODS else 0), key=k + "ref")
    tracker = st.radio("Трекер", list(CAT.TRACKERS), index=list(CAT.TRACKERS).index(cfg["tracking"]["tracker"]), horizontal=True, key=k + "trk")
    fps = st.slider("Частота обработки, к/с", 4, 15, int(cfg["video"]["process_fps"]), key=k + "fps")
    conf.update({"pose.weights": weights, "pose.backend": "rtmlib" if str(weights).startswith("rtmlib") else "ultralytics", "pose.runtime": runtime, "pose.device": device,
                 "pose.imgsz": int(imgsz), "pose.refine.enabled": refine != "выкл", "tracking.tracker": tracker, "video.process_fps": int(fps)})
    if refine != "выкл":
        conf["pose.refine.method"] = refine

    st.markdown("**Оценка цикла и дополнительные модели**")
    cyc = [b.id for b in CAT.cycle_bundles()]
    allow_heur = bool(o["allow_heuristic"])
    if classifier:
        cb = classifier
    elif cyc or o["cycle_bundle"]:
        cb = _pick("Классификатор цикла (обязателен)", cyc, o["cycle_bundle"], k + "cb", fmt=lambda p: Path(p).name)
    else:
        st.error("В models/cycle/ нет ни одного классификатора цикла: обучите (`sd train-bundle`) или установите решатель (`sd solver install`).")
        cb = NONE
        allow_heur = st.checkbox("Разрешить эвристику по длительности паузы (только отладка)", value=allow_heur, key=k + "heur")
    pho = [b.id for b in CAT.photo_bundles()]
    pb = _pick("Фото-модель (признаки)", [NONE, *pho], o["photo_bundle"] or NONE, k + "pb", fmt=lambda p: Path(p).name if p != NONE else NONE)
    dets = [d.id for d in CAT.detector_models()]
    objs = st.multiselect("Детекторы предмета на кропах кисть–рот", sorted(set(dets) | set(o["objects"] or [])), default=list(o["objects"] or []), key=k + "obj")
    vlms = [v.id for v in CAT.vlm_models()]
    cur_v = o["vlm_model"] or NONE
    vm = _pick("VLM-верификатор (Ollama)", [NONE, *vlms], cur_v, k + "vlm", help="список — модели запущенного локального Ollama; другой тег можно ввести ниже")
    manual = st.text_input("…или тег VLM вручную", "", key=k + "vlmm", placeholder="например qwen3-vl:4b")
    vm = manual.strip() or vm
    vmode = st.radio("Режим VLM", ["grey", "all"], index=0 if o["vlm_mode"] != "all" else 1, horizontal=True, key=k + "vmode", disabled=vm == NONE,
                     format_func=lambda m: {"grey": "серая зона", "all": "все циклы"}[m])
    grey = st.slider("Серая зона оценки цикла", 0.0, 1.0, tuple(o["grey"]), 0.05, key=k + "grey", disabled=vm == NONE or vmode == "all")
    with st.expander("Слияние сигналов с оценкой классификатора (VLM, предмет, фото-модель)"):
        st.caption("Поправка к логиту оценки классификатора; вектор признаков классификатора не меняется. Вес 0 — выключено. Подбирайте на валидации («Эксперименты» → «Варианты профиля», ключ fusion).")
        fz = dict((o["fusion"] or {}).get("weights", {}))
        fw = {n: st.slider(lbl, 0.0, 5.0, float(fz.get(n, 0.0)), 0.1, key=k + "fz" + n) for n, lbl in (("vlm", "вес VLM"), ("object", "вес предмета"), ("photo", "вес фото-модели"))}
        fclip = st.slider("Максимум поправки, логиты", 0.5, 10.0, float((o["fusion"] or {}).get("clip_logit", 2.0)), 0.1, key=k + "fzclip")
    opt.update(cycle_bundle=None if cb == NONE else cb, photo_bundle=None if pb == NONE else pb, objects=list(objs), vlm_model=None if vm == NONE else vm,
               vlm_mode=vmode if vm != NONE else "off", grey=[float(grey[0]), float(grey[1])], allow_heuristic=allow_heur,
               fusion=dict(weights={n: w for n, w in fw.items() if w > 0}, clip_logit=float(fclip)) if any(w > 0 for w in fw.values()) else None)

    st.markdown("**Пороги**")
    th = st.slider("Порог confidence события (фиксируется до скрытого набора)", 0.0, 1.0, float(cfg["events"]["confidence_threshold"]), 0.01, key=k + "th")
    conf["events.confidence_threshold"] = float(th)
    with st.expander("Автомат циклов и сборка событий"):
        a, b = st.columns(2)
        th_in = a.slider("th_in — вход ко рту", 0.10, 1.50, float(cfg["cycles"]["th_in"]), 0.05, key=k + "thin")
        th_out = b.slider("th_out — выход", 0.20, 2.00, float(cfg["cycles"]["th_out"]), 0.05, key=k + "thout")
        hold = a.slider("Пауза у рта, с", 0.2, 8.0, (float(cfg["cycles"]["hold_min_sec"]), float(cfg["cycles"]["hold_max_sec"])), 0.1, key=k + "hold")
        cth = b.slider("Порог оценки цикла", 0.0, 1.0, float(cfg["events"]["cycle_th"]), 0.05, key=k + "cth")
        gap = a.slider("Склейка циклов, с", 5.0, 30.0, float(cfg["events"]["merge_gap_sec"]), 1.0, key=k + "gap")
        win = b.slider("Окно «два цикла», с", 10.0, 40.0, float(cfg["events"]["window_sec"]), 1.0, key=k + "win")
        conf.update({"cycles.th_in": float(th_in), "cycles.th_out": float(max(th_out, th_in + 0.05)), "cycles.hold_min_sec": float(hold[0]), "cycles.hold_max_sec": float(hold[1]),
                     "events.cycle_th": float(cth), "events.merge_gap_sec": float(gap), "events.window_sec": float(win)})
    conf.update(_all_params(base, conf, k))
    clean = {k_: v for k_, v in opt.items() if DEFAULT_OPTIONS.get(k_) != v or k_ in base.options}
    return Profile(name=base.name, description=base.description, base=base.base, config=conf, options=clean)
