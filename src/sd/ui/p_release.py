"""Вкладка «Выпуск»: единственный путь модели в продукт — ворота качества → решатель + PDF + манифест выпуска; список выпусков с проверкой целостности;
установка решателя, пришедшего с другой машины."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from sd import profiles as PR
from sd import release as RL
from sd import solver as SV
from sd.ui import common as UC

SECTIONS = ["Новый выпуск", "Выпуски", "Установить решатель"]


def _new() -> None:
    names = PR.list_profiles()
    if not names:
        st.info("Нет профилей.")
        return
    c = st.columns([2, 4])
    prof = c[0].selectbox("Профиль", names, index=names.index("mvp") if "mvp" in names else 0, key="rl_prof")
    fp = PR.load_named(prof).fingerprint()
    runs = UC.runs_df()
    import json

    from sd import experiments as XP

    match = []
    for r in runs.itertuples() if len(runs) else []:
        try:
            if json.loads((XP.EXP_DIR / r.id / "meta.json").read_text(encoding="utf-8")).get("fingerprint") == fp and r.mode == "events":
                match.append(r.id)
        except Exception:       # noqa: BLE001
            continue
    with c[1]:
        if match:
            st.caption(f"прогонов с текущим отпечатком профиля ({fp}): {len(match)} — выбран последний")
        else:
            st.warning(f"текущий профиль (отпечаток {fp}) ещё не оценивался — ворота «тот же профиль» не пройдут: оцените его («Валидация»)")
        run = UC.run_select("Прогон оценки, подтверждающий качество", key="rl_run", mode="events", prefer=match[0] if match else None)
    if not run:
        return
    c = st.columns(3)
    min_f1 = c[0].number_input("Минимальная F1", 0.0, 1.0, 0.70, 0.01, key="rl_min")
    name = c[1].text_input("Имя выпуска", key="rl_name", placeholder=f"{prof}_дата")
    weights = c[2].checkbox("Положить веса моделей в архив", key="rl_w", help="для машины без интернета; архив станет большим")
    gs = RL.gates(prof, run, float(min_f1))
    st.markdown("**Ворота качества**")
    for g in gs:
        st.markdown(f"{ {True: '✅', False: '⛔', None: '⚠️'}[g['ok']]} **{g['title']}** — {g['detail']}")
    ok = RL.passed(gs)
    vals = dict(profile=prof, run=run, name=name.strip(), min_f1=float(min_f1), weights=weights, force=False)
    if ok:
        UC.task_button("release", vals, "Выпустить", key="rl_go")
    else:
        st.error("Ворота не пройдены: выпуск обычным путём невозможен. Исправьте причины выше (чаще всего — оцените текущий профиль на отложенных клипах).")

        def _force():
            from sd import tasks as T

            return f"задача {T.start('release', {**vals, 'force': True})} — выпуск будет помечен как принудительный"

        UC.confirm("Выпустить принудительно", "rl_force", "Выпуск при непройденных воротах помечается в манифесте (forced) и виден при установке. Используйте только для демонстрации, "
                                                          "не для поставки заказчику.", _force)
    UC.task_panel(("release",), key="rl_tp", limit=2, title=None)
    st.caption("Подпись манифеста: если задан ключ SD_RELEASE_KEY, release.json подписывается HMAC-SHA256 (проверка — раздел «Выпуски»). Без ключа — только sha256 архива.")


def _list() -> None:
    rel = RL.list_releases()
    if not rel:
        st.info("Выпусков нет.")
        return
    df = pd.DataFrame(rel)
    st.dataframe(df.drop(columns=["dir"]), hide_index=True)
    pick = st.selectbox("Выпуск", [r["name"] for r in rel], key="rl_pick")
    r = next(x for x in rel if x["name"] == pick)
    d = Path(r["dir"])
    v = RL.verify(d)
    st.markdown(f"{'✅' if v['sha_ok'] else '⛔'} sha256 архива {'совпадает' if v['sha_ok'] else 'НЕ совпадает — архив изменён после выпуска'} · "
                + ("подпись: " + ("верна" if v["signature_ok"] else ("неверна" if v["signature_ok"] is False else "есть, но ключ не задан — не проверить")) if v["signed"] else "без подписи"))
    c = st.columns(2)
    z = next(d.glob("*.sdsolver.zip"), None)
    if z:
        c[0].download_button("Скачать решатель", z.read_bytes(), file_name=z.name, key="rl_dl_z")
    if (d / "report.pdf").exists():
        c[1].download_button("Скачать PDF прогона", (d / "report.pdf").read_bytes(), file_name=f"{pick}.pdf", key="rl_dl_p")
    with st.expander("release.json"):
        st.code((d / "release.json").read_text(encoding="utf-8"), language="json")


def _install() -> None:
    up = st.file_uploader("Архив *.sdsolver.zip", type=["zip"], key="sv_up")
    if up is not None:
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "in.sdsolver.zip"
            f.write_bytes(up.getvalue())
            try:
                info = SV.inspect(f)
            except Exception as e:      # noqa: BLE001
                st.error(f"не решатель: {e}")
                return
            meta = info["meta"]
            st.markdown(f"**{meta['id']}** · создан {meta['created']} · отпечаток `{meta['fingerprint']}`")
            if meta.get("evaluation"):
                e = meta["evaluation"]["metrics"]
                st.caption(f"оценка при упаковке: F1 {e['f1']}, порог {e['threshold']}, клипов {e['clips']}")
            if not info["ok"]:
                st.error("нарушена целостность: " + "; ".join(info["problems"]))
                return
            ow = st.checkbox("Заменить, если уже установлен", key="sv_ow")
            if st.button("Установить", key="sv_inst"):
                try:
                    out = SV.install(f, overwrite=ow)
                    st.success(f"установлен {out['name']}: " + ", ".join(out["installed"]))
                    for i in out["issues"]:
                        (st.error if i["level"] == "error" else st.warning)(i["text"])
                except Exception as e:      # noqa: BLE001
                    st.error(str(e))
    inst = SV.installed_solvers()
    if inst:
        st.markdown("**Установленные решатели**")
        st.dataframe(pd.DataFrame(inst), hide_index=True)


def render() -> None:
    UC.style()
    UC.header("Выпуск", "Единственный путь модели в продукт: ворота качества → решатель + PDF + манифест; установка решателя.")
    sec = UC.sections(SECTIONS, "rel_section")
    {"Новый выпуск": _new, "Выпуски": _list, "Установить решатель": _install}[sec]()
