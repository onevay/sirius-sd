"""Вкладки «Фото-модель» (данные, оценка, проверка своего снимка) и «Модели» (реестр, подключение и проверка новых моделей, перенос на другую машину)."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from sd import doctor as D
from sd import feature_auc as FA
from sd import model_tools as MT
from sd import models as M
from sd.paths import MODELS, OUTPUTS, ROOT

NONE = "— нет —"
PHOTO_OUT = OUTPUTS / "photo"


def _csv(name: str) -> pd.DataFrame | None:
    f = PHOTO_OUT / name
    return pd.read_csv(f) if f.exists() else None


def tab_photo():
    st.markdown("**Фото-классификатор «курит / не курит»** на замороженных эмбеддингах (CLIP, ConvNeXt-B) + линейная голова; данные — открытые фото-наборы в `data/` и `data_external/`. "
                "Что и как проверялось — `docs/ANALYSIS.md`, раздел 9. Команды: `sd.cmd photos audit | embed | eval | train | eval-video`.")
    f = PHOTO_OUT / "audit.json"
    if f.exists():
        rep = json.loads(f.read_text(encoding="utf-8"))
        rows = [dict(набор=k, снимков=v["n"], классы=str(v["classes"]), групп_дублей=v["dup_groups"], дубли_между_сплитами=v["dup_groups_across_splits"],
                     конфликт_меток=v["dup_groups_label_conflict"], AUC_только_метаданные=v.get("shortcut", {}).get("auc_meta_only")) for k, v in rep.items() if not k.startswith("_")]
        st.subheader("Данные и аудит")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("«AUC только метаданные» — классификатор видит лишь размер/яркость/резкость: заметно выше 0.5 значит, что часть класса «угадывается» без содержимого (шорткат).")
    else:
        st.info("Аудит ещё не запускали: `sd.cmd photos audit`.")
    img = ROOT / "docs" / "img" / "photo_eval.png"
    if img.exists():
        st.subheader("Качество на фото")
        st.image(str(img), caption="AUC внутри наборов (CV по группам дублей) и при переносе между наборами")
    for nm, ttl in (("eval_per_negative.csv", "AUC против каждого «трудного» негатива"), ("eval_video.csv", "Перенос на кропы рта из видео (68 размеченных циклов)"),
                    ("eval_zero_shot.csv", "CLIP без обучения")):
        d = _csv(nm)
        if d is not None:
            st.subheader(ttl)
            st.dataframe(d.round(3), hide_index=True)
    st.subheader("Проверить свой снимок")
    bundles = [b["name"] for b in MT.list_bundles() if b["kind"] == "photo"]
    if not bundles:
        st.info("Пакетов фото-модели нет: `sd.cmd photos train photo_v1`.")
        return
    name = st.selectbox("Пакет", bundles)
    up = st.file_uploader("Снимок (jpg/png)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="photo_up")
    if up and st.button("Оценить", type="primary"):
        from sd import photo_clf as PC
        from sd import photo_feats as PF

        bundle = PC.load_bundle(MODELS / "photo" / name)
        imgs = [cv2.imdecode(np.frombuffer(u.read(), np.uint8), cv2.IMREAD_COLOR) for u in up]
        rgb = [PF.to_square_rgb(i) for i in imgs]
        X = {bb: PF.make_embedder(bb).embed(rgb) for bb in bundle["manifest"]["backbones"]}
        s = PC.score_bundle(bundle, X)
        cols = st.columns(min(4, len(up)))
        for i, (u, im, p) in enumerate(zip(up, imgs, s)):
            cols[i % len(cols)].image(cv2.cvtColor(cv2.resize(im, (260, int(260 * im.shape[0] / im.shape[1]))), cv2.COLOR_BGR2RGB), caption=f"{u.name}: P(курит) = {p:.2f}")


def _registry_table() -> pd.DataFrame:
    rows = []
    for s in M.load_registry().values():
        r = M.status(s)
        rows.append(dict(id=r["id"], тип=r["kind"], скачан="да" if r["present"] else "нет", МБ=r["size_mb"], доверие=r["trust"], лицензия=r["license"], скан=r["scan"]))
    return pd.DataFrame(rows)


def tab_models():
    st.markdown("**Модели: подключить, проверить, перенести.** Чужие веса не исполняются: `.pt` (pickle) сканируются, `.onnx`/`.safetensors` не содержат кода, пакеты проекта — только JSON/TXT/safetensors "
                "с проверкой sha256. Записи пользователя лежат в `models/user_models.yaml` (не затираются обновлением проекта).")
    part = st.radio("Раздел", ["Реестр", "Подключить свою модель", "Проверить модель", "Перенос на другую машину"], horizontal=True, key="models_part")
    if part == "Реестр":
        st.dataframe(_registry_table(), hide_index=True)
        b = MT.list_bundles()
        st.subheader("Пакеты моделей проекта")
        st.dataframe(pd.DataFrame(b), hide_index=True) if b else st.info("Пакетов пока нет.")
        users = MT.load_user_models()
        if users:
            st.subheader("Ваши модели")
            st.dataframe(pd.DataFrame(users), hide_index=True)
            rm = st.selectbox("Удалить запись", [NONE] + [m["id"] for m in users])
            if rm != NONE and st.button("Удалить"):
                MT.remove_user_model(rm)
                st.rerun()
    elif part == "Подключить свою модель":
        _add_model()
    elif part == "Проверить модель":
        _test_model()
    else:
        _transfer()


def _add_model():
    st.markdown("Файл весов (`.pt/.pth` — после сканирования, `.onnx`, `.safetensors`) или репозиторий Hugging Face (запись в реестр; скачивание — отдельной кнопкой).")
    c1, c2, c3 = st.columns(3)
    mid = c1.text_input("id (буквы, цифры, - _ .)", "my-detector")
    kind = c2.selectbox("Тип", list(MT.KINDS), help="detector — YOLO-детектор предмета; pose — поза; photo/vlm/tube — прочие")
    lic = c3.text_input("Лицензия", "?")
    up = st.file_uploader("Файл весов", type=["pt", "pth", "onnx", "safetensors", "bin"], key="model_up")
    repo = st.text_input("…или Hugging Face repo (например, user/name)", "")
    pats = st.text_input("Шаблоны файлов для скачивания через запятую", "*.json,model.safetensors") if repo else ""
    allow_unknown = st.checkbox("Разрешить «unknown» после сканирования pickle (осознанно)", value=False)
    if st.button("Подключить", type="primary"):
        try:
            if up is not None:
                with tempfile.TemporaryDirectory() as td:
                    p = Path(td) / up.name
                    p.write_bytes(up.getvalue())
                    r = MT.register_user_model(dict(id=mid, kind=kind, license=lic), copy_from=p, allow_unknown=allow_unknown)
                st.success(f"Подключено: {r['spec']['file']}. Проверка безопасности: {r['scan']['verdict']} — {'; '.join(r['scan']['notes'])}")
            elif repo:
                MT.register_user_model(dict(id=mid, kind=kind, license=lic, hf_repo=repo, hf_patterns=[x.strip() for x in pats.split(",") if x.strip()]))
                st.success("Запись добавлена. Скачать веса: кнопка ниже или `sd.cmd models get " + mid + "`.")
            else:
                st.warning("Укажите файл или репозиторий.")
        except Exception as e:
            st.error(str(e))
    users = [m for m in MT.load_user_models() if m.get("hf_repo")]
    if users:
        pick = st.selectbox("Скачать веса записи (HF)", [m["id"] for m in users])
        if st.button("Скачать"):
            with st.spinner("скачиваем…"):
                st.write(str(M.fetch(M.load_registry()[pick])))


def _labelled_cycles() -> pd.DataFrame:
    tab = FA.feature_table(FA.LABELS, with_dataset=True)
    return tab[tab.start.notna()].copy()


def _test_model():
    kind = st.radio("Что проверяем", ["пакет классификатора цикла", "пакет фото-модели", "детектор предмета (YOLO)", "VLM (Ollama)"], horizontal=True, key="test_kind")
    if kind.startswith("пакет классификатора"):
        names = [b["name"] for b in MT.list_bundles() if b["kind"] == "cycle"]
        if not names:
            st.info("Пакетов классификатора цикла нет: `sd.cmd train-bundle` или импорт из zip.")
            return
        name = st.selectbox("Пакет", names)
        if st.button("Оценить на размеченных циклах", type="primary"):
            tab = _labelled_cycles()
            res = MT.eval_cycle_bundle(MODELS / "cycle" / name, tab)
            c1, c2, c3 = st.columns(3)
            c1.metric("AUC", f"{res['auc']:.3f}")
            c2.metric("95% интервал (по видео)", f"{res['auc_lo']:.2f}–{res['auc_hi']:.2f}")
            c3.metric("циклов (курение / не курение)", f"{res['n_pos']} / {res['n'] - res['n_pos']}")
            if res["missing_features"]:
                st.warning(f"Нет признаков в таблице ({len(res['missing_features'])}): {', '.join(res['missing_features'][:12])} — они считаются пропусками.")
            man = json.loads((MODELS / "cycle" / name / "manifest.json").read_text(encoding="utf-8"))
            cv = man.get("cv", {})
            st.info(f"Честная оценка при обучении пакета (out-of-fold по видео, {man.get('n')} циклов, {man.get('n_groups')} видео): AUC ансамбля **{cv.get('auc_ensemble', '—')}** "
                    f"(члены: {cv.get('auc_members', {})}). AUC выше — на тех же циклах, на которых пакет обучался, поэтому завышен; для новой модели используйте циклы, которых не было в обучении.")
            st.caption(res["note"])
            st.dataframe(pd.DataFrame(dict(video=tab.video, tid=tab.tid, start=tab.start, метка=tab.label, y=tab.y, score=res["scores"])).round(3), hide_index=True)
    elif kind.startswith("пакет фото"):
        names = [b["name"] for b in MT.list_bundles() if b["kind"] == "photo"]
        if not names:
            st.info("Пакетов фото-модели нет.")
            return
        name = st.selectbox("Пакет", names)
        audit = PHOTO_OUT / "photos_audit.parquet"
        src = st.radio("Данные", ["кропы рта из видео (размеченные циклы)", "фото-наборы"], horizontal=True)
        if st.button("Оценить", type="primary"):
            if src.startswith("кропы"):
                from sd import photo_clf as PC
                from sd import photo_feats as PF
                from sd import photo_video as PV

                tab = _labelled_cycles()
                tab["start"] = tab.start.round(3)
                ft = pd.DataFrame([dict(video=r.video, tid=int(r.tid), start=float(r.start), path=str(p)) for r in tab.itertuples()
                                   for p in sorted(PV.cycle_dir(r.video, r.tid, r.start).glob("*.jpg"))])
                if ft.empty:
                    st.error("Кропов нет: `sd.cmd analyze-cycles --what photo --photo-bundle models/photo/<имя>`")
                    return
                bundle = PC.load_bundle(MODELS / "photo" / name)
                emb = {bb: PF.embed_files(PF.make_embedder(bb), ft.path.tolist()) for bb in bundle["manifest"]["backbones"]}
                agg = PV.aggregate(ft, {"p": PV.score_frames(bundle, emb)})
                d = tab.merge(agg, on=["video", "tid", "start"], how="left", suffixes=("_old", ""))
                rows = []
                for col in ("photo_p_mean", "photo_p_max", "photo_p_top2"):
                    a, lo, hi = MT.auc_with_ci(d.y.to_numpy(), d[col].to_numpy(float), d.video.to_numpy(), 300)
                    rows.append(dict(признак=col, AUC=a, нижняя=lo, верхняя=hi))
                st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
            elif audit.exists():
                df = pd.read_parquet(audit)
                n = st.session_state.get("photo_n", 400)
                df = df[df.label.notna()].groupby("source", group_keys=False).apply(lambda g: g.sample(min(len(g), n), random_state=0)).reset_index(drop=True)
                res = MT.eval_photo_bundle(MODELS / "photo" / name, df)
                st.json({k: v for k, v in res.items() if k != "scores"})
            else:
                st.info("Нет outputs/photo/photos_audit.parquet: `sd.cmd photos audit`.")
    elif kind.startswith("детектор"):
        _test_detector()
    else:
        _test_vlm()


def _test_detector():
    reg = M.load_registry()
    dets = {s.id: s.local_path for s in reg.values() if s.kind == "detector" and s.local_path and s.local_path.exists()}
    if not dets:
        st.info("Детекторов нет. Подключите свой на странице «Подключить свою модель».")
        return
    did = st.selectbox("Детектор", list(dets))
    conf = st.slider("Порог уверенности", 0.05, 0.9, 0.15, 0.05)
    src = st.radio("Картинки", ["загрузить", "кропы циклов (размеченные)"], horizontal=True)
    imgs, caps = [], []
    if src == "загрузить":
        for u in st.file_uploader("Картинки", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="det_up") or []:
            imgs.append(cv2.imdecode(np.frombuffer(u.read(), np.uint8), cv2.IMREAD_COLOR))
            caps.append(u.name)
    else:
        from sd import photo_video as PV

        tab = _labelled_cycles()
        tab["start"] = tab.start.round(3)
        n = st.slider("Циклов каждого класса", 2, 20, 6)
        for y in (1, 0):
            for r in tab[tab.y == y].sample(min(n, int((tab.y == y).sum())), random_state=1).itertuples():
                fr = sorted(PV.cycle_dir(r.video, r.tid, r.start).glob("*.jpg"))
                if fr:
                    imgs.append(cv2.imread(str(fr[len(fr) // 2])))
                    caps.append(f"{'курение' if y else r.label} · {str(r.video)[:20]}")
    if imgs and st.button("Прогнать детектор", type="primary"):
        t0 = time.perf_counter()
        with st.spinner("…"):
            res = MT.detect_images(dets[did], imgs, conf)
        st.caption(f"{len(imgs)} картинок за {time.perf_counter() - t0:.1f} с")
        cols = st.columns(4)
        for i, (im, ds, cp) in enumerate(zip(imgs, res, caps)):
            v = im.copy()
            for d in ds:
                x1, y1, x2, y2 = [int(t) for t in d["box"]]
                cv2.rectangle(v, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(v, f"{d['cls']} {d['conf']:.0%}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cols[i % 4].image(cv2.cvtColor(cv2.resize(v, (300, int(300 * v.shape[0] / v.shape[1]))), cv2.COLOR_BGR2RGB), caption=f"{cp} · найдено {len(ds)}")


def _test_vlm():
    from sd import ollama_ctl as O

    info = O.status()
    if not info["up"]:
        st.warning("Сервер Ollama не запущен (`sd.cmd ollama up --igpu`).")
        return
    st.write("Установленные модели:", [m["name"] for m in info["models"]] or "нет")
    tag = st.text_input("Тег модели для проверки", "qwen3.5:2b-q4_K_M")
    from sd import photo_video as PV
    from sd.vlm import OllamaVLM

    tab = _labelled_cycles()
    tab["start"] = tab.start.round(3)
    k = st.selectbox("Цикл", tab.index, format_func=lambda i: f"{tab.label[i]} · {str(tab.video[i])[:24]} · {tab.start[i]:.1f} с")
    if st.button("Спросить модель", type="primary"):
        r = tab.loc[k]
        fr = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in sorted(PV.cycle_dir(r.video, int(r.tid), float(r.start)).glob("*.jpg"))]
        if not fr:
            st.error("Нет кропов этого цикла.")
            return
        with st.spinner("модель думает…"):
            ans = OllamaVLM(tag, size=224, mode="yesno").verify(fr, 2.5)
        st.image(np.hstack([cv2.resize(f, (160, 160)) for f in fr]))
        st.metric("P(затяжка) по логитам", f"{ans['p_puff']:.2f}" if ans.get("p_puff") is not None else "—")
        st.caption(f"ответ «{ans['raw']}», {ans['ms'] / 1000:.1f} с; метка ассистента: {r.label}")


def _transfer():
    st.subheader("Экспорт пакетов для другой машины")
    b = MT.list_bundles()
    if not b:
        st.info("Пакетов нет.")
    else:
        pick = st.multiselect("Пакеты", [f"{x['kind']}/{x['name']}" for x in b])
        if pick and st.button("Собрать zip"):
            out = OUTPUTS / "export" / f"models_{time.strftime('%Y%m%d_%H%M%S')}.zip"
            man = MT.export_bundles([tuple(p.split("/")) for p in pick], out)
            st.session_state["export_zip"] = str(out)
            st.success(f"{out.name}: {len(man)} файлов, sha256 записаны в MANIFEST.sha256.json")
        z = st.session_state.get("export_zip")
        if z and Path(z).exists():
            st.download_button("Скачать zip", Path(z).read_bytes(), file_name=Path(z).name)
    st.subheader("Импорт пакетов")
    up = st.file_uploader("zip, собранный экспортом", type=["zip"], key="imp_zip")
    ow = st.checkbox("Перезаписать существующие пакеты", value=False)
    if up and st.button("Импортировать"):
        try:
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "in.zip"
                p.write_bytes(up.getvalue())
                st.success("Импортировано: " + ", ".join(MT.import_bundles(p, overwrite=ow)))
        except Exception as e:
            st.error(str(e))
    st.subheader("Проверка окружения (как `sd.cmd doctor`)")
    if st.button("Проверить окружение"):
        r = D.check()
        st.code(D.render_report(r), language="text")
