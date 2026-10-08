"""Страница «Модели»: реестр и профили экспериментов, подключение своих моделей, проверка, перенос; фото-модель."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from sd import catalog as CAT
from sd import profiles as PR
from sd.ui import tabs_models as TM


def _profiles() -> None:
    names = PR.list_profiles()
    c = CAT.summary()
    st.caption(f"Доступно: поза {c['pose']} · детекторы {c['detector']} · классификаторы цикла {c['cycle']} · фото {c['photo']} · VLM {c['vlm']}")
    if not names:
        st.info("Профилей пока нет: соберите модели на странице «Оценка» и сохраните профиль.")
        return
    rows = []
    for n in names:
        try:
            rows.append(dict(профиль=n, **PR.load(n).describe()))
        except Exception as e:
            rows.append(dict(профиль=n, pose=f"ошибка: {e}"))
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    rm = st.selectbox("Удалить профиль", ["—"] + names)
    if rm != "—" and st.button("Удалить"):
        PR.delete(rm)
        st.rerun()


def _show_meta(meta: dict) -> None:
    a = meta["architecture"]
    st.markdown(f"**{meta['id']}** · архитектура `{a['mode']}` · создан {meta['created']} · отпечаток `{meta['fingerprint']}`")
    rows = []
    for s in a["stages"]:
        extra = s.get("weights", {}).get("id") or ",".join(x["id"] for x in s.get("detectors", [])) or s.get("bundle") or s.get("model") or ""
        rows.append(dict(стадия=s["id"], модель=extra, параметры="; ".join(f"{k}={v}" for k, v in s.items() if k not in ("id", "summary", "weights", "detectors", "bundle", "model"))[:110]))
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    cl = next((s.get("summary") for s in a["stages"] if s["id"] == "cycle_classifier"), None)
    if cl:
        st.caption(f"классификатор {cl['name']}: {cl['n_features']} признаков {cl['groups']}, члены {cl['members']}, AUC out-of-fold {cl['auc_oof']}")
    r = a["resources_estimate"]
    st.caption(f"память (оценка порядка величин): пик {r['peak_gb']} ГБ при всех моделях вместе, ≈ {r['sequential_gb']} ГБ при последовательной загрузке · режимы: {a['streaming']}")
    if meta.get("evaluation"):
        e = meta["evaluation"]["metrics"]
        st.caption(f"оценка в момент упаковки: F1 {e['f1']}, порог {e['threshold']}, клипов {e['clips']}")
    for n in meta.get("notes", []):
        st.caption("• " + n)


def _solver() -> None:
    import tempfile
    from pathlib import Path

    from sd import experiments as XP
    from sd import runner as RN
    from sd import solver as SV
    from sd.paths import OUTPUTS

    st.markdown("**Упаковать** профиль целиком (классификатор, метаданные, по желанию веса и набор разметки) для передачи на другое устройство.")
    names = PR.list_profiles()
    if names:
        c = st.columns([2, 2, 2])
        prof = c[0].selectbox("Профиль", names, key="sv_prof")
        cls = c[1].selectbox("Классификатор", ["как в профиле", *[b.id for b in CAT.cycle_bundles()]], key="sv_cls")
        runs = XP.list_runs()
        run = c[2].selectbox("Оценка в метаданные", ["—", *(runs["id"].tolist() if len(runs) else [])], key="sv_run")
        w = st.columns(3)
        ww = w[0].checkbox("Веса моделей", help="если лежат на диске (onnx/safetensors/pt)")
        wd = w[1].checkbox("Набор разметки", help="events_gt.csv, cycle_labels.csv, индекс клипов")
        nm = w[2].text_input("Имя", prof, key="sv_name")
        if st.button("Упаковать", type="primary"):
            try:
                p = RN.with_classifier(PR.load(prof), None if cls == "как в профиле" else cls)
                out = SV.pack(p, OUTPUTS / "solvers" / Path(nm).name, name=nm, with_weights=ww, with_dataset=wd, run_id=None if run == "—" else run)
                st.session_state["sv_packed"] = out["path"]
                st.success(f"{out['path']} · {out['size_mb']} МБ")
            except Exception as e:
                st.error(str(e))
        packed = st.session_state.get("sv_packed")
        if packed and Path(packed).exists():
            _show_meta(SV.inspect(packed)["meta"])
            st.download_button("Скачать архив", Path(packed).read_bytes(), file_name=Path(packed).name)
    else:
        st.info("Сохранённых профилей нет: соберите на странице «Оценка» и сохраните.")
    st.divider()
    st.markdown("**Установить** решатель из архива.")
    up = st.file_uploader("Архив *.sdsolver.zip", type=["zip"], key="sv_up")
    if up is not None:
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "in.sdsolver.zip"
            f.write_bytes(up.getvalue())
            try:
                info = SV.inspect(f)
            except Exception as e:
                st.error(f"не решатель: {e}")
                return
            _show_meta(info["meta"])
            if not info["ok"]:
                st.error("нарушена целостность: " + "; ".join(info["problems"]))
                return
            ow = st.checkbox("Заменить, если уже установлен")
            if st.button("Установить"):
                try:
                    out = SV.install(f, overwrite=ow)
                    st.success(f"установлен {out['name']}: " + ", ".join(out["installed"]))
                    for i in out["issues"]:
                        (st.error if i["level"] == "error" else st.warning)(i["text"])
                except Exception as e:
                    st.error(str(e))
    inst = SV.installed_solvers()
    if inst:
        st.markdown("**Установленные**")
        st.dataframe(pd.DataFrame(inst), hide_index=True)


def render() -> None:
    part = st.radio("Раздел", ["Профили", "Решатель", "Реестр", "Подключить модель", "Проверить модель", "Перенос", "Фото-модель"], horizontal=True, label_visibility="collapsed")
    if part == "Профили":
        _profiles()
    elif part == "Решатель":
        _solver()
    elif part == "Фото-модель":
        TM.tab_photo()
    else:
        TM.tab_models({"Реестр": "Реестр", "Подключить модель": "Подключить свою модель", "Проверить модель": "Проверить модель", "Перенос": "Перенос на другую машину"}[part])
