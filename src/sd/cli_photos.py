"""Команды `sd photos ...`: фото-датасеты курения — загрузка публичных, аудит, признаки, классификатор, перенос на видео."""
from __future__ import annotations

from . import _env  # noqa: F401

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from .paths import EXTERNAL, OUTPUTS, ROOT

photos_app = typer.Typer(no_args_is_help=True, help="Фото-датасеты «курит / не курит»: загрузка публичных, аудит (дубли, шорткаты), признаки, классификатор, перенос на кропы видео.")
console = Console(width=140)
PHOTO_OUT = OUTPUTS / "photo"
DEFAULT_ROOTS = [ROOT / "data" / "Smoker Detection"]


def _tbl(title: str, cols: list[str], rows: list[list]) -> None:
    t = Table(title=title, box=box.SIMPLE_HEAVY)
    for c in cols:
        t.add_column(c, overflow="fold")
    for r in rows:
        t.add_row(*[str(x) for x in r])
    console.print(t)


def photo_roots() -> list[Path]:
    """Все папки с фото-датасетами: data/Smoker Detection + всё, что fetch/пользователь положил в data_external с labels.csv."""
    roots = [p for p in DEFAULT_ROOTS if p.exists()]
    if EXTERNAL.exists():
        for src in sorted(p for p in EXTERNAL.iterdir() if p.is_dir()):
            for ds in sorted(p for p in src.iterdir() if p.is_dir()):
                if (ds / "labels.csv").exists():
                    roots.append(ds)
    return roots


@photos_app.command("fetch", help="Скачать публичный набор без входа в аккаунт (stanford40 | mendeley_smoker_2400 | smoking_img_final_test — фото «курит / не курит»; cigdet — рамки сигарет для проверки детектора) "
                                  "в data_external/. Остальные (Roboflow/Kaggle/Ultralytics) — вручную, ссылки в docs/NEXT_STEPS.md.")
def fetch_cmd(name: Annotated[str, typer.Argument(help="stanford40 | mendeley_smoker_2400 | smoking_img_final_test | cigdet")],
              easy_per_class: Annotated[int, typer.Option(help="stanford40: сколько случайных снимков брать из каждого «лёгкого» класса")] = 8,
              keep_zip: Annotated[bool, typer.Option(help="stanford40: не удалять zip после распаковки")] = False) -> None:
    from . import photo_fetch as PF

    if name not in PF.SOURCES:
        console.print(f"[red]неизвестный источник {name}[/]; доступны: {', '.join(PF.SOURCES)}")
        raise typer.Exit(1)
    kw = dict(easy_per_class=easy_per_class) if name == "stanford40" else {}
    res = PF.fetch(name, keep_zip=keep_zip, **kw)
    console.print(res)
    if "data_yaml" in res:
        console.print(f"лицензия и конфигурация YOLO: {Path(res['dest']) / 'LICENSE.txt'}, {res['data_yaml']}; проверка детектора: sd detector-eval <детектор> --data {res['data_yaml']} --split test")
    else:
        console.print(f"лицензия и метки: {Path(res['dest']) / 'LICENSE.txt'}, {Path(res['dest']) / 'labels.csv'}")


@photos_app.command("export-ov", help="Один раз конвертировать backbone в OpenVINO IR (models/ov): на iGPU ≈0.08–0.13 с на снимок вместо 1–3 с в torch. Нужен torch и веса (sd models get clip-vit-b32 / convnext-b-in22k).")
def export_ov_cmd(backbone: Annotated[str, typer.Argument(help="clip | convnext")], force: bool = False,
                  check: Annotated[bool, typer.Option(help="после экспорта сверить эмбеддинги OpenVINO и torch на 8 снимках (косинус; нужны веса и память для torch)")] = True) -> None:
    from . import photo_feats as PF

    console.print(f"IR: {PF.export_ov(backbone, force)}")
    if check:
        r = PF.ov_parity(backbone)
        ok = r["cos_mean"] >= 0.99
        console.print(f"сверка с torch на {r['n']} {'снимках' if r['real'] else 'синтетических изображениях'}: косинус средний {r['cos_mean']:.4f}, минимальный {r['cos_min']:.4f}"
                      + ("" if ok else " [yellow]— ниже 0.99: конвертация искажает признаки, не использовать этот IR (--force пересоздаст)[/]"))
        if not ok:
            raise typer.Exit(1)


def _audit_table() -> "pd.DataFrame":
    import pandas as pd

    f = PHOTO_OUT / "photos_audit.parquet"
    if not f.exists():
        console.print("[yellow]нет outputs/photo/photos_audit.parquet — сначала `sd photos audit`[/]")
        raise typer.Exit(1)
    return pd.read_parquet(f)


@photos_app.command("embed", help="Эмбеддинги всех фото-наборов (кэш по файлам в outputs/photo/emb): clip и/или convnext. Можно прервать и продолжить.")
def embed_cmd(backbones: Annotated[str, typer.Option(help="через запятую: clip,convnext")] = "clip,convnext",
              backend: Annotated[str, typer.Option(help="auto | ov | torch")] = "auto",
              batch: int = 16) -> None:
    import time

    from . import photo_feats as PF

    df = _audit_table()
    for bb in [b.strip() for b in backbones.split(",") if b.strip()]:
        t0 = time.perf_counter()
        emb = PF.make_embedder(bb, backend)
        console.print(f"{bb}: {type(emb).__name__}" + (f" на {emb.device}" if hasattr(emb, "device") else "") + f"; загрузка {time.perf_counter() - t0:.0f} с, снимков {len(df)}")
        last = [0.0]

        def prog(i, n, dt):
            if dt - last[0] > 30 or i == n:
                console.print(f"  {bb}: {i}/{n}, {dt:.0f} с")
                last[0] = dt

        X = PF.embed_files(emb, df.path.tolist(), batch, prog)
        console.print(f"{bb}: готово {np_nan_count(X)} без эмбеддинга; {time.perf_counter() - t0:.0f} с всего")


def np_nan_count(X) -> int:
    import numpy as np

    return int(np.isnan(X).any(1).sum())


@photos_app.command("embed-aug", help="Эмбеддинги «под камеру» (объект в кадре уменьшен, сжатие JPEG, размытие, шум): K вариантов каждого снимка; кэш по ключу <файл>|aug<i>. Для обучения, устойчивого к кропам видео.")
def embed_aug_cmd(backbones: Annotated[str, typer.Option(help="через запятую: clip,convnext")] = "clip", k: Annotated[int, typer.Option(help="вариантов на снимок")] = 2,
                  backend: str = "auto", sources: Annotated[Optional[str], typer.Option(help="наборы через запятую; по умолчанию все")] = None) -> None:
    import hashlib
    import time

    import cv2
    import numpy as np

    from . import photo_aug as AUG
    from . import photo_feats as PF

    df = _audit_table()
    if sources:
        df = df[df.source.isin([s.strip() for s in sources.split(",")])]
    paths = df.path.tolist()
    for bb in [b.strip() for b in backbones.split(",") if b.strip()]:
        t0 = time.perf_counter()
        emb = PF.make_embedder(bb, backend)
        cache = PF.load_cache(bb + "_aug")
        todo = [(p, i) for p in paths for i in range(k) if AUG.aug_key(PF._key(p), i) not in cache]
        console.print(f"{bb}: к расчёту {len(todo)} из {len(paths) * k}")
        for s in range(0, len(todo), 16):
            chunk = todo[s:s + 16]
            imgs, keys = [], []
            for p, i in chunk:
                im = cv2.imread(str(p))
                if im is None:
                    continue
                rng = np.random.default_rng(int(hashlib.md5(f"{PF._key(p)}|{i}".encode()).hexdigest()[:8], 16))
                imgs.append(PF.to_square_rgb(AUG.cctv_like(im, rng)))
                keys.append(AUG.aug_key(PF._key(p), i))
            if imgs:
                for key, v in zip(keys, emb.embed(imgs)):
                    cache[key] = v
            if (s // 16 + 1) % 20 == 0:
                PF.save_cache(bb + "_aug", cache)
                console.print(f"  {bb}: {s + len(chunk)}/{len(todo)}, {time.perf_counter() - t0:.0f} с")
        PF.save_cache(bb + "_aug", cache)
        console.print(f"{bb}: готово, {time.perf_counter() - t0:.0f} с")


@photos_app.command("eval-video", help="Перенос фото-пакета на кропы рта из видео: AUC против меток циклов (интервал по видео), проверка на смешение со сценой. Показывает, нужны ли аугментации/дообучение на кропах.")
def eval_video_cmd(bundles: Annotated[str, typer.Argument(help="имена пакетов models/photo/<имя> через запятую")] = "photo_v1",
                   backend: str = "auto", n_boot: int = 400) -> None:
    import numpy as np
    import pandas as pd

    from . import feature_auc as FA
    from . import model_tools as MT
    from . import photo_clf as PC
    from . import photo_feats as PF
    from . import photo_video as PV

    tab = FA.feature_table(FA.LABELS)
    tab = tab[tab.start.notna()].copy()
    tab["start"] = tab.start.round(3)
    rows = [dict(video=r.video, tid=int(r.tid), start=float(r.start), path=str(p)) for r in tab.itertuples() for p in sorted(PV.cycle_dir(r.video, r.tid, r.start).glob("*.jpg"))]
    ft = pd.DataFrame(rows)
    if ft.empty:
        console.print("[red]нет кропов циклов: `sd analyze-cycles --what photo --photo-bundle …`[/]")
        raise typer.Exit(1)
    fold = FA.folder_of(tab.video).to_numpy()
    out = []
    for name in [b.strip() for b in bundles.split(",")]:
        bundle = PC.load_bundle(ROOT / "models" / "photo" / name)
        emb = {bb: PF.embed_files(PF.make_embedder(bb, backend), ft.path.tolist()) for bb in bundle["manifest"]["backbones"]}
        s = PV.score_frames(bundle, emb)
        agg = PV.aggregate(ft, {"p": s})
        d = tab.merge(agg, on=["video", "tid", "start"], how="left", suffixes=("_old", ""))
        for col in ("photo_p_mean", "photo_p_max", "photo_p_top2"):
            a, lo, hi = MT.auc_with_ci(d.y.to_numpy(), d[col].to_numpy(float), d.video.to_numpy(), n_boot)
            neg = d.y.to_numpy() == 0
            scene = PC.auc((fold[neg] == "smoking").astype(int), d[col].to_numpy(float)[neg])
            sm = fold == "smoking"
            within = PC.auc(d.y.to_numpy()[sm], d[col].to_numpy(float)[sm])
            out.append(dict(пакет=name, признак=col, AUC=a, lo=lo, hi=hi, внутри_папки_курение=within, папка_среди_негативов=scene))
    res = pd.DataFrame(out)
    console.print(res.round(3).to_string(index=False))
    res.to_csv(PHOTO_OUT / "eval_video.csv", index=False)


@photos_app.command("eval", help="Качество фото-классификатора на эмбеддингах: CV по группам дублей внутри набора, перенос между наборами, официальные сплиты, «трудные» негативы, страты яркости, zero-shot CLIP. Пишет outputs/photo/eval_*.csv и docs/img/photo_eval.png.")
def eval_cmd(kinds: Annotated[str, typer.Option(help="через запятую: logreg,svm,et")] = "logreg,svm,et",
             backbones: Annotated[str, typer.Option(help="через запятую: clip,convnext")] = "clip,convnext",
             seed: int = 0, zero_shot: Annotated[bool, typer.Option(help="посчитать zero-shot CLIP (один раз загрузит текстовую башню через torch)")] = True) -> None:
    import numpy as np
    import pandas as pd

    from . import photo_clf as PC
    from . import photo_eval as PE
    from . import photo_feats as PF

    df = _audit_table()
    bbs, ks = tuple(b.strip() for b in backbones.split(",")), tuple(k.strip() for k in kinds.split(","))
    res = PE.run_all(df, bbs, ks, seed)
    d = res["table"]
    console.print(f"снимков с признаками: {len(d)} из {len(df)}; наборы: {d.groupby('source').size().to_dict()}")
    fmt = lambda v: "—" if v is None or v != v else f"{v:.3f}"   # noqa: E731
    w = res["within"].pivot(index="member", columns="source", values="auc")
    _tbl("AUC внутри набора (5-фолдовая CV по группам почти-дубликатов)", ["модель"] + list(w.columns), [[m] + [fmt(w.loc[m, c]) for c in w.columns] for m in w.sort_values(list(w.columns)[0]).index])
    t = res["transfer"]
    _tbl("Перенос между наборами (обучили на A → проверили на B)", ["обучение", "проверка", "модель", "AUC", "AP", "recall@0.5", "FPR@0.5"],
         [[r.train, r.test, r.member, fmt(r.auc), fmt(r.ap), fmt(r.recall), fmt(r.fpr)] for r in t.sort_values(["train", "auc"], ascending=[True, False]).itertuples()])
    o = res["official"]
    if len(o):
        _tbl("Официальные сплиты (train+val → test)", ["набор", "модель", "AUC", "recall@0.5", "FPR@0.5"],
             [[r.source, r.member, fmt(r.auc), fmt(r.recall), fmt(r.fpr)] for r in o.sort_values(["source", "auc"], ascending=[True, False]).itertuples()])
    out = PHOTO_OUT
    res["within"].to_csv(out / "eval_within.csv", index=False)
    t.to_csv(out / "eval_transfer.csv", index=False)
    o.to_csv(out / "eval_official.csv", index=False)

    # «трудные» негативы и страты яркости — на лучшем ансамбле каждого набора
    audit_meta = json.loads((out / "audit.json").read_text(encoding="utf-8")) if (out / "audit.json").exists() else {}
    neg_rows, strat_rows = [], []
    for src, g in d.groupby("source"):
        key = (src, "ens:all-single-backbone")
        if key not in res["store"]:
            continue
        idx, s = res["store"][key]
        gg = d.loc[idx].assign(_s=s)
        pn = PC.per_negative_auc(gg, s)
        for r in pn.itertuples():
            neg_rows.append(dict(source=src, neg_class=r.neg_class, n=r.n, auc=r.auc))
        sa = PC.stratified_auc(gg.label.to_numpy(int), s, gg.bright.to_numpy(float))
        strat_rows.append(dict(source=src, bright_strata_auc=sa["mean"], per_bin=sa["per_bin"]))
    neg = pd.DataFrame(neg_rows)
    if len(neg):
        _tbl("AUC «курит» против каждого негативного класса (ансамбль, OOF)", ["набор", "негативный класс", "n", "AUC"],
             [[r.source, r.neg_class, r.n, fmt(r.auc)] for r in neg.sort_values("auc").itertuples()])
        neg.to_csv(out / "eval_per_negative.csv", index=False)
    strat = pd.DataFrame(strat_rows)
    _tbl("AUC внутри страт яркости (если качество — шорткат по яркости, внутри страты оно падает)", ["набор", "AUC по стратам (среднее)", "по квартилям яркости"],
         [[r.source, fmt(r.bright_strata_auc), r.per_bin] for r in strat.itertuples()])
    strat.to_csv(out / "eval_brightness_strata.csv", index=False)

    if zero_shot and "clip" in bbs:
        text = PF.clip_text_embeddings()
        zs = PE.zero_shot_table(d, res["X"]["clip"], text)
        _tbl("CLIP без обучения (zero-shot): AUC по сходству с промптом", ["набор", "промпт", "AUC"], [[r.source, r.prompt, fmt(r.auc)] for r in zs.itertuples()])
        zs.to_csv(out / "eval_zero_shot.csv", index=False)
    meta = {s: v.get("shortcut", {}).get("auc_meta_only") for s, v in audit_meta.items() if not s.startswith("_")}
    fig = PE.plot_eval(res["within"], t, ROOT / "docs" / "img" / "photo_eval.png", meta, shared=PE.shared_duplicate_groups(d))
    console.print(f"график: {fig}; таблицы: {out}")


@photos_app.command("train", help="Обучить фото-модель на выбранных наборах и сохранить пакет БЕЗ pickle (logreg по каждому backbone в JSON + изотоническая калибровка по OOF) в models/photo/<имя>/.")
def train_cmd(name: Annotated[str, typer.Argument(help="имя пакета: models/photo/<имя>/")] = "photo_v1",
              backbones: Annotated[str, typer.Option(help="через запятую: clip,convnext")] = "clip,convnext",
              sources: Annotated[Optional[str], typer.Option(help="какие наборы брать (через запятую; по умолчанию все с эмбеддингами)")] = None,
              aug: Annotated[int, typer.Option(help="добавить K аугментированных «под камеру» вариантов каждого снимка (`sd photos embed-aug`)")] = 0,
              seed: int = 0) -> None:
    import numpy as np

    from . import photo_aug as AUG
    from . import photo_clf as PC
    from . import photo_eval as PE
    from . import photo_feats as PF

    df = _audit_table()
    bbs = tuple(b.strip() for b in backbones.split(","))
    if sources:
        df = df[df.source.isin([s.strip() for s in sources.split(",")])]
    df = df.reset_index(drop=True)
    X = PE.feature_sets(df, bbs)
    d = PE.usable(df, X)
    idx = d.index.to_numpy()
    y, g = d.label.to_numpy(int), d.xdup_group.to_numpy()
    Xb = {bb: X[bb][idx] for bb in bbs}
    n0 = len(y)
    if aug:
        caches = {bb: PF.load_cache(bb + "_aug") for bb in bbs}
        keys = [PF._key(p) for p in d.path]
        for i in range(aug):
            A = {bb: np.stack([caches[bb].get(AUG.aug_key(k, i), np.full(PF.DIM[bb], np.nan, np.float32)) for k in keys]) for bb in bbs}
            ok = ~np.any([np.isnan(a).any(1) for a in A.values()], axis=0)
            if not ok.all():
                console.print(f"[yellow]вариант {i}: нет эмбеддингов у {int((~ok).sum())} снимков — они пропущены[/]")
            Xb = {bb: np.concatenate([Xb[bb], A[bb][ok]]) for bb in bbs}
            y, g = np.concatenate([y, d.label.to_numpy(int)[ok]]), np.concatenate([g, d.xdup_group.to_numpy()[ok]])   # варианты снимка — в той же группе: в один фолд
    oof = np.mean([PC.oof_scores(Xb[bb], y, g, "logreg", seed=seed) for bb in bbs], axis=0)
    src = np.concatenate([d.source.to_numpy()] + [d.source.to_numpy()] * aug)[:len(y)] if aug else d.source.to_numpy()
    per_src = {s: PC.auc(y[:n0][(d.source == s).to_numpy()], oof[:n0][(d.source == s).to_numpy()]) for s in d.source.unique()}
    out = ROOT / "models" / "photo" / name
    meta = dict(sources={s: int((d.source == s).sum()) for s in d.source.unique()}, n=int(len(y)), n_originals=int(n0), n_pos=int(y.sum()), aug_variants=aug, seed=seed,
                oof_auc=PC.auc(y[:n0], oof[:n0]), oof_auc_by_source=per_src, created=__import__("time").strftime("%Y-%m-%d %H:%M:%S"),
                note="logreg(C=0.05, balanced) на нормированных эмбеддингах; калибровка — изотоническая по OOF среднего вероятностей; эмбеддинги считаются photo_feats (OpenVINO iGPU или torch)"
                     + ("; в обучении + аугментации «под камеру» (zoom-out, JPEG, размытие, шум)" if aug else ""))
    PC.save_bundle(out, list(bbs), Xb, y, oof, meta, seed)
    console.print(f"пакет: {out}  (снимков {n0}, обучающих строк {len(y)}, курящих {int(y.sum())})")
    console.print(f"OOF AUC (оригиналы): все {meta['oof_auc']:.3f}; по наборам " + ", ".join(f"{s} {a:.3f}" for s, a in per_src.items()))


@photos_app.command("from-yolo", help="Подключить скачанный YOLO-датасет (Roboflow «YOLOv8», Ultralytics) к фото-конвейеру: пишет labels.csv (снимок с рамкой нужного класса = «курит», без рамок = фон) в каталог набора. Дальше: `sd photos audit`, `embed`, `eval`.")
def from_yolo_cmd(path: Annotated[Path, typer.Argument(help="каталог набора с data.yaml, например data_external/roboflow/cigarrette_detection")],
                  positive: Annotated[Optional[str], typer.Option(help="имена классов-«курит» через запятую (по умолчанию любые рамки)")] = None) -> None:
    from . import photo_fetch as PF

    root = path if path.is_absolute() else ROOT / path
    if not (root / "data.yaml").exists():
        console.print(f"[red]нет {root / 'data.yaml'}[/]")
        raise typer.Exit(1)
    rows = PF.labels_from_yolo(root, {x.strip() for x in positive.split(",")} if positive else None)
    n1 = sum(r["label"] for r in rows)
    console.print(f"{root.name}: снимков {len(rows)}, «курит» (есть рамка) {n1}, фон {len(rows) - n1}; labels.csv записан")
    if not (root / "LICENSE.txt").exists():
        console.print("[yellow]нет LICENSE.txt — положите рядом строку лицензии со страницы датасета (требование кейса)[/]")


@photos_app.command("montage", help="Сетка случайных снимков набора для проверки разметки глазами: `sd photos montage --source stanford40_actions --cls smoking`. Файл — outputs/photo/montage_*.jpg.")
def montage_cmd(source: Annotated[Optional[str], typer.Option(help="имя набора из `sd photos audit`")] = None, cls: Annotated[Optional[str], typer.Option(help="исходный класс (колонка cls)")] = None,
                label: Annotated[Optional[int], typer.Option(help="1 — курит, 0 — нет")] = None, n: int = 24, seed: int = 0) -> None:
    import cv2

    from . import photo_data as PD

    df = _audit_table()
    if source:
        df = df[df.source == source]
    if cls:
        df = df[df.cls == cls]
    if label is not None:
        df = df[df.label == label]
    if df.empty:
        console.print("[red]по этим фильтрам снимков нет[/]")
        raise typer.Exit(1)
    out = PHOTO_OUT / f"montage_{(source or 'all')}_{(cls or 'any')}_{label if label is not None else 'x'}.jpg"
    cv2.imwrite(str(out), PD.montage(df, n, seed=seed))
    console.print(f"{len(df)} снимков в выборке, показано {min(n, len(df))}: {out}")


@photos_app.command("audit", help="Аудит фото-датасетов: баланс, размеры, почти-дубликаты (в т.ч. между сплитами), шорткаты по метаданным; пишет outputs/photo/audit.json и таблицу.")
def audit_cmd(root: Annotated[Optional[list[Path]], typer.Option("--root", help="папка набора (можно несколько); по умолчанию все известные")] = None,
              max_dist: Annotated[int, typer.Option(help="порог Хэмминга dHash для «почти дубликата»")] = 4) -> None:
    import pandas as pd

    from . import photo_data as PD

    roots = root or photo_roots()
    frames, report = [], {}
    for r in roots:
        r = r if r.is_absolute() else ROOT / r
        df = PD.scan_photos(r, source=r.name)
        df, rep = PD.audit(df, max_dist, progress=lambda i, n, r=r: console.print(f"  {r.name}: {i}/{n}") if i else None)
        frames.append(df)
        report[r.name] = rep
    allp = pd.concat(frames, ignore_index=True)
    # дубликаты между разными наборами — тоже утечка (один снимок в train одного и в test другого)
    allp["xdup_group"] = PD.duplicate_groups([int(h, 16) for h in allp.dhash], max_dist)
    g = allp.groupby("xdup_group").source.nunique()
    report["_across_sources"] = dict(groups_in_several_sources=int((g > 1).sum()))
    PHOTO_OUT.mkdir(parents=True, exist_ok=True)
    allp.to_parquet(PHOTO_OUT / "photos_audit.parquet", index=False)
    (PHOTO_OUT / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    rows = []
    for name, rep in report.items():
        if name.startswith("_"):
            continue
        sc = rep.get("shortcut", {})
        rows.append([name, rep["n"], rep["classes"], rep["dup_groups"], rep["dup_groups_across_splits"], rep["dup_groups_label_conflict"],
                     sc.get("auc_meta_only", "—")])
    _tbl("Аудит фото-наборов", ["набор", "снимков", "классы", "групп дублей", "из них между сплитами", "дубли с разными метками", "AUC «только метаданные»"], rows)
    console.print(f"между наборами: {report['_across_sources']}; таблица: {PHOTO_OUT / 'photos_audit.parquet'}; отчёт: {PHOTO_OUT / 'audit.json'}")
