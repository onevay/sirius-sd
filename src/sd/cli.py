"""CLI: каждый этап пайплайна запускается и проверяется отдельно, промежуточные результаты рисуются на видео/кадрах.

    python -m sd --help            (или sd.cmd --help в корне проекта)
Подробности и примеры — docs/CLI.md.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import json
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import box
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from .config import load_config
from .paths import OUTPUTS, ROOT, list_videos, resolve_video, video_id, weak_label

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="markdown",
                  help="**smoking-detection** — детекция курения по видео: поза+трекинг → циклы жеста → события по регламенту.")
models_app = typer.Typer(no_args_is_help=True, help="Реестр весов: список, загрузка, проверка безопасности.")
app.add_typer(models_app, name="models")
ollama_app = typer.Typer(no_args_is_help=True, help="Локальный Ollama для VLM: сервер только на 127.0.0.1 и без облака, статус, загрузка моделей.")
app.add_typer(ollama_app, name="ollama")
from .cli_photos import photos_app  # noqa: E402
from .cli_solver import solver_app  # noqa: E402

app.add_typer(photos_app, name="photos")
app.add_typer(solver_app, name="solver")
import shutil as _shutil

# при перенаправлении вывода rich берёт ширину 80 и ломает таблицы — задаём разумный минимум
console = Console(width=max(130, _shutil.get_terminal_size((160, 40)).columns))

# ---------------------------------------------------------------------------------------------- общие опции
VideoArg = Annotated[str, typer.Argument(help="Путь к видео или имя/stem внутри data/ (например `sm_3`, `курение__5`)")]
Start = Annotated[float, typer.Option("--start", help="Начало интервала, с")]
End = Annotated[Optional[float], typer.Option("--end", help="Конец интервала, с (по умолчанию — до конца)")]
SetOpt = Annotated[Optional[list[str]], typer.Option("--set", "-s", help="Переопределение конфига: секция.ключ=значение (можно несколько)")]
CfgOpt = Annotated[Optional[Path], typer.Option("--config", help="Профиль-накладка поверх configs/default.yaml (напр. configs/profiles/fast.yaml); достаточно указать изменяемые ключи")]
Force = Annotated[bool, typer.Option("--force", help="Пересчитать этап, игнорируя кэш")]
FrameAt = Annotated[Optional[str], typer.Option("--frame-at", help="Сохранить PNG-скриншоты с разметкой на моменты (с), через запятую: 3.3,12.5")]
NoVideo = Annotated[bool, typer.Option("--video/--no-video", help="Рендерить видео с разметкой")]


def _rel(path: Path) -> str:
    """Путь относительно корня проекта для сообщений (вне корня — как есть)."""
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _progress(desc: str):
    p = Progress(TextColumn("[bold]{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"),
                 TimeElapsedColumn(), TimeRemainingColumn(), console=console, transient=True)
    p.start()
    tid = p.add_task(desc, total=1)

    def cb(i: int, n: int) -> None:
        p.update(tid, completed=i, total=max(n, i))

    return p, cb


def _setup(video: str, start: float, end: float | None, set_: list[str] | None, config: Path | None):
    vid = resolve_video(video)
    cfg = load_config(config, set_ or [])
    return vid, cfg


def _frames_at(spec: str | None) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()] if spec else []


def _tbl(title: str, cols: list[str], rows: list[list], **kw) -> None:
    t = Table(title=title, box=box.SIMPLE_HEAVY, **kw)
    for c in cols:
        t.add_column(c, overflow="fold")
    for r in rows:
        t.add_row(*[str(x) for x in r])
    console.print(t)


# ---------------------------------------------------------------------------------------------- hw / models / info
@app.command(help="Характеристики ПК и оценка применимости VLM. `--bench` — реальные GFLOPS процессора.")
def hw(bench: Annotated[bool, typer.Option("--bench", help="Замерить GFLOPS matmul (≈6 с)")] = False,
       json_out: Annotated[Optional[Path], typer.Option("--json", help="Сохранить отчёт в JSON")] = None) -> None:
    from . import hw as hwm

    info = hwm.collect(bench=bench)
    console.print(hwm.render_report(info))
    rows = [r for r in hwm.vlm_budget(info) if r["quant"] in ("Q4", "Q8")]
    _tbl("VLM: помещается ли в память (оценка: веса + 1 ГБ на KV/рантайм; «после чистки» = всего − 3.5 ГБ на ОС)",
         ["модель", "квант", "веса, ГБ", "нужно, ГБ", "после чистки, ГБ", "свободно сейчас, ГБ", "память", "влезает после чистки", "влезает сейчас"],
         [[r["model"], r["quant"], r["weights_gb"], r["need_gb"], r["usable_gb"], r["avail_now_gb"], r["memory"], "да" if r["fits"] else "нет",
           "да" if r["fits_now"] else "нет"] for r in rows])
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(dict(hw=info, vlm=hwm.vlm_budget(info)), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        console.print(f"JSON: {json_out}")


@models_app.command("list", help="Список весов из configs/models.yaml: скачаны ли, размер, sha256, результат сканирования.")
def models_list() -> None:
    from . import models as M

    rows = []
    for s in M.load_registry().values():
        st = M.status(s)
        rows.append([s.id, s.kind, "да" if st["present"] else "нет", st["size_mb"], s.trust, s.license, st["scan"], st["sha256"]])
    _tbl("Реестр весов", ["id", "тип", "скачан", "МБ", "доверие", "лицензия", "скан .pt", "sha256"], rows)


@models_app.command("get", help="Скачать веса по id (несколько). Community-.pt сразу сканируются на опасные вызовы.")
def models_get(ids: Annotated[list[str], typer.Argument(help="id из `sd models list` или `all`")]) -> None:
    from . import models as M

    reg = M.load_registry()
    todo = list(reg) if ids == ["all"] else ids
    for i in todo:
        if i not in reg:
            console.print(f"[red]нет такого id: {i}[/]")
            raise typer.Exit(1)
        p, cb = _progress(f"{i}")
        try:
            path = M.fetch(reg[i], cb)
        finally:
            p.stop()
        console.print(f"[green]OK[/] {i}: {path}  scan={M.status(reg[i])['scan']}")


@models_app.command("scan", help="Проверка безопасности .pt: id из реестра или путь к файлу. Код не исполняется.")
def models_scan(target: Annotated[str, typer.Argument(help="id или путь")], details: bool = False) -> None:
    from . import models as M
    from .safety import scan_torch_archive

    reg = M.load_registry()
    path = reg[target].local_path if target in reg else Path(target)
    rep = scan_torch_archive(path)
    console.print(rep.summary())
    if details:
        for g in rep.globals_found:
            console.print("  ", g)
    raise typer.Exit(0 if rep.ok else 2)


@app.command(help="Метаданные видео: разрешение, fps, кодек, длительность, проверка VFR (равномерности меток времени).")
def info(videos: Annotated[Optional[list[str]], typer.Argument(help="Видео; без аргументов — все из data/")] = None,
         vfr: Annotated[bool, typer.Option("--vfr/--no-vfr", help="Проверить метки времени на первых 400 кадрах")] = True) -> None:
    from .video_io import probe, timestamp_stats

    paths = [resolve_video(v) for v in videos] if videos else list_videos()
    rows = []
    for p in paths:
        i = probe(p)
        ts = timestamp_stats(p) if vfr else {}
        rows.append([weak_label(p) or "-", p.name[:34], f"{i.width}x{i.height}", i.fps, i.codec, f"{i.duration:.0f}", i.size_mb,
                     ts.get("fps_measured", "-"), ts.get("gaps_gt_2x", "-")])
    _tbl(f"{len(rows)} видео", ["метка", "файл", "WxH", "fps", "кодек", "сек", "МБ", "fps изм.", "пропусков"], rows)


@app.command(help="EDA датасета: размеры людей (px), качество ключевых точек по размеру, присутствие, яркость. Пишет таблицы и графики.")
def eda(every_sec: Annotated[float, typer.Option(help="Шаг выборки кадров, с")] = 2.0,
        max_samples: Annotated[int, typer.Option(help="Максимум кадров на видео (шаг увеличивается для длинных)")] = 250,
        model: Annotated[str, typer.Option(help="id весов позы")] = "yolo26n-pose",
        imgsz: Annotated[int, typer.Option()] = 960,
        root: Annotated[Optional[Path], typer.Option(help="Папка с видео (по умолчанию data/)")] = None,
        only: Annotated[Optional[list[str]], typer.Option(help="Только эти видео (имя/stem)")] = None,
        out: Annotated[Path, typer.Option(help="Куда писать таблицы")] = OUTPUTS / "analysis",
        img_dir: Annotated[Path, typer.Option(help="Куда писать графики")] = ROOT / "docs" / "img",
        save_frames: Annotated[bool, typer.Option(help="Сохранить до 3 кадров с людьми на видео (набор для бенчмарков)")] = True) -> None:
    from . import eda as E

    vids = [resolve_video(o) for o in only] if only else list_videos(root)
    cfg = load_config()
    p = Progress(TextColumn("[bold]EDA"), BarColumn(), TextColumn("{task.fields[v]}"), TimeElapsedColumn(), console=console, transient=True)
    p.start()
    task = p.add_task("eda", total=len(vids), v="")

    def cb(i, n, vid, frac):
        p.update(task, completed=i + frac, v=f"{vid[:40]} {frac:.0%}")

    try:
        fr, pr = E.run_eda(vids, cfg, model, imgsz, every_sec, max_samples, save_frames=(out / "frames") if save_frames else None, progress=cb)
    finally:
        p.stop()
    out.mkdir(parents=True, exist_ok=True)
    fr.to_parquet(out / "eda_frames.parquet", index=False)
    pr.to_parquet(out / "eda_persons.parquet", index=False)
    s = E.summarize(fr, pr)
    (out / "eda_summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    paths = E.plots(fr, pr, img_dir)
    console.print(f"кадров {s.get('n_frames')}, людей {s.get('n_persons')}; кадров с людьми {s.get('frames_with_person', 0):.0%}; время {fr.attrs.get('wall_sec', 0):.0f} с")
    console.print(f"высота людей, px (5/25/50/75/95%): {s.get('height_quantiles')}; <80px: {s.get('share_h_lt80', 0):.0%}; <150px: {s.get('share_h_lt150', 0):.0%}")
    _tbl("Уверенность ключевых точек по высоте человека", list(s["kp_conf_by_height"][0].keys()),
         [[(round(v, 2) if isinstance(v, float) else v) for v in r.values()] for r in s["kp_conf_by_height"]])
    for pth in paths:
        console.print(f"график: {pth}")


# ---------------------------------------------------------------------------------------------- этапы пайплайна
def _render_outputs(vid: Path, tr, cfg, rd: Path, stage: str, renderer, video: bool, frames: list[float]) -> None:
    from .render import render_frame, render_video

    out = rd / "render"
    out.mkdir(exist_ok=True)
    for t in frames:
        console.print(f"скриншот: {render_frame(vid, tr, cfg, t, out / f'{stage}_t{t:g}.png', renderer=renderer)}")
    if video:
        p, cb = _progress("рендер видео")
        try:
            path = render_video(vid, tr, cfg, out / f"{stage}.mp4", renderer=renderer, progress=cb)
        finally:
            p.stop()
        console.print(f"видео: {path} ({path.stat().st_size / 1e6:.1f} МБ)")


def _print_pose(tr) -> None:
    m = tr.meta
    ref = f" + {m['pose']['refine_method']}" if m["pose"].get("refine") else ""
    console.print(f"[bold]поза+трекинг[/] {m['pose']['weights']}{ref} {m['pose']['runtime']}/{m['pose']['device']} imgsz={m['pose']['imgsz']} "
                  f"| кадров {m['frames_processed']} | установившийся {m.get('fps_steady')} к/с (с декодированием; прогрев {m.get('warmup_sec')} с; "
                  f"всего {m['fps_wall']} к/с) | мс/шаг (поза+трекер): медиана {m['ms']['total'].get('median')}, p95 {m['ms']['total'].get('p95')}")
    st = m.get("stitch", {})
    console.print(f"треков до склейки {st.get('tracks_before')} -> после {st.get('tracks_after')} (склеено {len(st.get('merged', {}))}, "
                  f"коротких удалено {st.get('dropped_short')})")
    s = tr.summary()
    _tbl("Треки", ["tid", "с", "по", "длит.", "детекций", "покрытие", "высота мед./мин/макс, px", "score", "IGNORE(<80px)"],
         [[int(r.tid), f"{r.t0:.1f}", f"{r.t1:.1f}", f"{r.dur:.1f}", int(r.n_det), f"{r.coverage:.0%}",
           f"{r.h_med:.0f}/{r.h_min:.0f}/{r.h_max:.0f}", f"{r.score:.2f}", "да" if r.ignore_small else ""] for r in s.itertuples()])


@app.command(help="Этап 1 — поза + трекинг людей. Рисует рамки с ID, ключевые точки (цвет = уверенность) и проценты.")
def pose(video: VideoArg, start: Start = 0.0, end: End = None, set_: SetOpt = None, config: CfgOpt = None, force: Force = False,
         render: NoVideo = True, frame_at: FrameAt = None,
         model: Annotated[Optional[str], typer.Option(help="id весов позы (sd models list), напр. yolo26s-pose")] = None,
         imgsz: Annotated[Optional[int], typer.Option(help="Размер входа, длинная сторона")] = None,
         runtime: Annotated[Optional[str], typer.Option(help="torch | openvino")] = None,
         device: Annotated[Optional[str], typer.Option(help="cpu | intel:cpu | intel:gpu")] = None,
         tracker: Annotated[Optional[str], typer.Option(help="botsort | bytetrack")] = None,
         refine: Annotated[Optional[bool], typer.Option("--refine/--no-refine", help="Уточнение ключевых точек по кропу человека (top-down)")] = None,
         refine_method: Annotated[Optional[str], typer.Option(help="метод уточнения: yolo | rtmpose-s | rtmpose-m (rtmpose — лучший по бенчмарку)")] = None) -> None:
    from . import stages
    from .render import Renderer

    ov = list(set_ or [])
    for k, v in (("pose.weights", model), ("pose.imgsz", imgsz), ("pose.runtime", runtime), ("pose.device", device),
                 ("tracking.tracker", tracker), ("pose.refine.enabled", refine), ("pose.refine.method", refine_method)):
        if v is not None:
            ov.append(f"{k}={str(v).lower() if isinstance(v, bool) else v}")
    vid, cfg = _setup(video, start, end, ov, config)
    p, cb = _progress("поза+трекинг")
    try:
        tr, rd = stages.stage_pose(vid, cfg, start, end, force=force, progress=cb)
    finally:
        p.stop()
    _print_pose(tr)
    console.print(f"каталог запуска: {rd}")
    _render_outputs(vid, tr, cfg, rd, "pose", Renderer(tr, cfg), render, _frames_at(frame_at))


def _load_chain(video, start, end, set_, config, force=False):
    from . import stages

    vid, cfg = _setup(video, start, end, set_, config)
    p, cb = _progress("поза+трекинг (кэш, если есть)")
    try:
        tr, rd = stages.stage_pose(vid, cfg, start, end, force=False, progress=cb)
    finally:
        p.stop()
    return vid, cfg, tr, rd


@app.command(help="Этап 2 — признаки по трекам: d_t (запястье–рот в ширинах плеч), скорость, углы. Рисует отрезок рука–рот и d_t.")
def features(video: VideoArg, start: Start = 0.0, end: End = None, set_: SetOpt = None, config: CfgOpt = None, force: Force = False,
             render: NoVideo = False, frame_at: FrameAt = None) -> None:
    from . import stages
    from .render import Renderer

    vid, cfg, tr, rd = _load_chain(video, start, end, set_, config)
    ser = stages.stage_features(tr, cfg, rd, force=force)
    rows = []
    for tid, s in ser.groupby("tid"):
        rows.append([int(tid), len(s), f"{s.d.notna().mean():.0%}", f"{s.d.min():.2f}" if s.d.notna().any() else "-",
                     f"{s.d.median():.2f}" if s.d.notna().any() else "-", f"{s.s.median():.0f}", f"{s.wrist_conf.mean():.2f}", f"{s.nose_conf.mean():.2f}"])
    _tbl("Признаки по трекам", ["tid", "отсчётов", "d валиден", "d мин", "d медиана", "масштаб s, px", "conf запястья", "conf носа"], rows)
    _render_outputs(vid, tr, cfg, rd, "features", Renderer(tr, cfg, series=ser), render, _frames_at(frame_at))


@app.command(help="Этап 3 — автомат циклов жеста REST→APPROACH→AT_MOUTH→RETRACT. Рисует состояния и таймлайн.")
def cycles(video: VideoArg, start: Start = 0.0, end: End = None, set_: SetOpt = None, config: CfgOpt = None, force: Force = False,
           render: NoVideo = True, frame_at: FrameAt = None) -> None:
    from . import stages
    from .render import Renderer

    vid, cfg, tr, rd = _load_chain(video, start, end, set_, config)
    ser = stages.stage_features(tr, cfg, rd)
    cyc, rej, st = stages.stage_cycles(ser, cfg, rd, force=True if force else False)
    console.print(f"циклов: {len(cyc)}; отброшено жестов: {len(rej)} ({rej.reason.value_counts().to_dict() if len(rej) else {}})")
    if len(cyc):
        _tbl("Циклы", ["tid", "начало", "у рта с", "у рта по", "конец", "пауза, с", "рука", "d мин"],
             [[int(r.tid), f"{r.start:.2f}", f"{r.mouth_in:.2f}", f"{r.mouth_out:.2f}", f"{r.end:.2f}", f"{r.hold:.2f}", "LR"[int(r.hand)] if r.hand in (0, 1) else "?",
               f"{r.d_min:.2f}"] for r in cyc.itertuples()])
    _render_outputs(vid, tr, cfg, rd, "cycles", Renderer(tr, cfg, series=ser, cycles=cyc, states=st), render, _frames_at(frame_at))


@app.command(help="Этап 6 — сборка событий по регламенту (2 цикла за 20 с или цикл+предмет, склейка 15 с). Пишет events.csv.")
def events(video: VideoArg, start: Start = 0.0, end: End = None, set_: SetOpt = None, config: CfgOpt = None,
           render: NoVideo = True, frame_at: FrameAt = None, camera_id: str = "cam_local") -> None:
    from . import stages
    from .render import Renderer

    vid, cfg, tr, rd = _load_chain(video, start, end, set_, config)
    ser = stages.stage_features(tr, cfg, rd)
    cyc, rej, st = stages.stage_cycles(ser, cfg, rd)
    df, evs = stages.stage_events(tr, ser, cyc, cfg, rd, camera_id, vid.stem)
    console.print(f"событий: {len(df)}")
    if len(df):
        console.print(df.to_string(index=False))
        for e in evs:
            console.print(f"  ID{e.tid}: {e.explain()} (правило {e.rule})")
    _render_outputs(vid, tr, cfg, rd, "events", Renderer(tr, cfg, series=ser, cycles=cyc, states=st, events=df), render, _frames_at(frame_at))


@app.command(help="Весь пайплайн: поза → признаки → циклы → события; пишет CSV и видео с разметкой.")
def run(video: VideoArg, start: Start = 0.0, end: End = None, set_: SetOpt = None, config: CfgOpt = None, force: Force = False,
        render: NoVideo = True, frame_at: FrameAt = None, camera_id: str = "cam_local") -> None:
    from . import stages
    from .render import Renderer

    vid, cfg = _setup(video, start, end, set_, config)
    p, cb = _progress("поза+трекинг")
    try:
        tr, rd = stages.stage_pose(vid, cfg, start, end, force=force, progress=cb)
    finally:
        p.stop()
    _print_pose(tr)
    ser = stages.stage_features(tr, cfg, rd, force=force)
    cyc, rej, st = stages.stage_cycles(ser, cfg, rd, force=force)
    df, evs = stages.stage_events(tr, ser, cyc, cfg, rd, camera_id, vid.stem, force=force)
    console.print(f"циклов {len(cyc)}, событий {len(df)}; CSV: {rd / 'events' }")
    _render_outputs(vid, tr, cfg, rd, "run", Renderer(tr, cfg, series=ser, cycles=cyc, states=st, events=df), render, _frames_at(frame_at))


@app.command("object", help="Этап 4 — прямой признак: сигарета/вейп на кропах кисть–рот вокруг каждого цикла. Рисует рамки и проценты.")
def object_cmd(video: VideoArg, start: Start = 0.0, end: End = None, set_: SetOpt = None, config: CfgOpt = None,
           detector: Annotated[Optional[list[str]], typer.Option(help="id детекторов из реестра (можно несколько)")] = None,
           conf: Annotated[Optional[float], typer.Option(help="порог уверенности детектора")] = None,
           render: NoVideo = False, frame_at: FrameAt = None) -> None:
    import pandas as pd

    from . import stages
    from .evidence import cycle_object_evidence, run_evidence
    from .render import Renderer

    ov = list(set_ or []) + ([f"evidence.conf={conf}"] if conf is not None else [])
    vid, cfg, tr, rd = _load_chain(video, start, end, ov, config)
    ser = stages.stage_features(tr, cfg, rd)
    cyc, rej, st = stages.stage_cycles(ser, cfg, rd)
    if cyc.empty:
        console.print("[yellow]циклов нет — искать предмет не на чем (ослабьте пороги: -s cycles.th_in=0.8 -s cycles.th_out=1.1)[/]")
        raise typer.Exit(0)
    p, cb = _progress("детектор на кропах")
    out = rd / "evidence"
    out.mkdir(exist_ok=True)
    try:
        det = run_evidence(vid, tr, ser, cyc, cfg, detector, progress=cb, save_crops=out / "crops")
    finally:
        p.stop()
    det.to_parquet(out / "detections.parquet", index=False)
    ev = cycle_object_evidence(det, cfg)
    console.print(f"кадров с кропами: {det.groupby(['detector', 'frame']).ngroups}; кропы с рамками: {out / 'crops'}")
    _tbl("Предмет по циклам (≥k из n соседних кадров)", ["детектор", "цикл", "кадров", "с предметом", "макс. conf", "has_object"],
         [[r.detector, r.cycle, r.frames, r.hit_frames, f"{r.max_conf:.0%}", "ДА" if r.has_object else ""] for r in ev.itertuples()])
    hits = det[det.cls.notna()]
    _render_outputs(vid, tr, cfg, rd, "object", Renderer(tr, cfg, series=ser, cycles=cyc, states=st, extras={"objects": hits}), render, _frames_at(frame_at))


@app.command(help="Этап 5 — видео-модели по «трубке» человека: X-CLIP (zero-shot) и/или VideoMAE (Kinetics, класс smoking).")
def tube(video: VideoArg, start: Start = 0.0, end: End = None, set_: SetOpt = None, config: CfgOpt = None,
         models: Annotated[str, typer.Option(help="xclip, videomae или xclip,videomae")] = "xclip",
         tid: Annotated[Optional[int], typer.Option(help="Трек для режима --t0/--t1")] = None,
         t0: Annotated[Optional[float], typer.Option(help="Начало окна, с (иначе окна вокруг циклов)")] = None,
         t1: Annotated[Optional[float], typer.Option(help="Конец окна, с")] = None,
         render: NoVideo = False, frame_at: FrameAt = None) -> None:
    from . import stages
    from .render import Renderer
    from .tube import run_tube

    vid, cfg, tr, rd = _load_chain(video, start, end, set_, config)
    ser = stages.stage_features(tr, cfg, rd)
    cyc, rej, st = stages.stage_cycles(ser, cfg, rd)
    if t0 is not None:
        wins = [(0, tid if tid is not None else tr.tids[0], t0, t1 if t1 is not None else t0 + cfg["tube"]["window_sec"])]
    else:
        wins = [(i, int(r.tid), float(r.peak_t - cfg["tube"]["window_sec"] / 2), float(r.peak_t + cfg["tube"]["window_sec"] / 2)) for i, r in enumerate(cyc.itertuples())]
    if not wins:
        console.print("[yellow]нет окон (циклов нет); задайте --t0/--t1[/]")
        raise typer.Exit(0)
    p, cb = _progress("видео-модели")
    out = rd / "tube"
    try:
        df = run_tube(vid, tr, wins, cfg, tuple(m.strip() for m in models.split(",")), progress=cb, save_dir=out / "strips")
    finally:
        p.stop()
    out.mkdir(exist_ok=True)
    df.to_parquet(out / "tube.parquet", index=False)
    console.print(f"окон: {len(df)}; мс/окно: {df.attrs.get('ms')}; ленты кадров: {out / 'strips'}")
    cols = ["window", "tid", "t0", "t1"] + [c for c in df.columns if c.startswith("p_")][:7] + [c for c in df.columns if c.startswith("videomae_k_smoking") or c == "videomae_top3"]
    show = df[[c for c in cols if c in df.columns]].copy()
    show.columns = [c.replace("p_a person ", "").replace("p_", "")[:18] for c in show.columns]
    console.print(show.round(2).to_string(index=False))
    _render_outputs(vid, tr, cfg, rd, "tube", Renderer(tr, cfg, series=ser, cycles=cyc, states=st, extras={"tube": df}), render, _frames_at(frame_at))


@app.command(help="Подобрать окна анализа (где больше людей) по результатам EDA — для пакетного прогона `sd batch`.")
def windows(length: Annotated[float, typer.Option(help="Длина окна, с")] = 30.0,
            out: Annotated[Path, typer.Option()] = OUTPUTS / "analysis" / "windows.csv") -> None:
    import pandas as pd

    from . import eda as E

    a = OUTPUTS / "analysis"
    fr, pr = pd.read_parquet(a / "eda_frames.parquet"), pd.read_parquet(a / "eda_persons.parquet")
    w = E.pick_windows(fr, pr, length)
    w.to_csv(out, index=False)
    console.print(w.to_string(index=False))
    console.print(f"-> {out}")


@app.command(help="Пакетный прогон: поза → признаки → циклы → события по окнам из windows.csv; сводка и таблица циклов для обучения.")
def batch(windows_csv: Annotated[Path, typer.Option("--windows")] = OUTPUTS / "analysis" / "windows.csv",
          set_: SetOpt = None, config: CfgOpt = None, only: Annotated[Optional[list[str]], typer.Option(help="Только эти видео (video_id)")] = None,
          force: Force = False) -> None:
    import pandas as pd

    from . import stages
    from .dataset import build_cycle_table
    from .paths import DATA

    cfg = load_config(config, set_ or [])
    w = pd.read_csv(windows_csv)
    all_v = {video_id(v): v for v in list_videos()}
    rows, tables = [], []
    for i, r in enumerate(w.itertuples()):
        if only and r.video not in only:
            continue
        vp = all_v[r.video]
        end = None if pd.isna(r.end) else float(r.end)
        t0 = time.perf_counter()
        try:
            tr, rd = stages.stage_pose(vp, cfg, float(r.start), end, force=force)
            ser = stages.stage_features(tr, cfg, rd)
            cyc, rej, st = stages.stage_cycles(ser, cfg, rd)
            df, evs = stages.stage_events(tr, ser, cyc, cfg, rd, "cam_local", vp.stem)
            s = tr.summary(cfg["video"]["min_person_height_px"])
            rows.append(dict(video=r.video, folder=weak_label(vp), start=r.start, end=end, tracks=len(s), tracks_ge80=int((~s.ignore_small).sum()),
                             h_med=float(s.h_med.median()) if len(s) else None, fps_wall=tr.meta["fps_wall"], infer_ms=tr.meta["ms"]["infer"].get("median"),
                             cycles=len(cyc), rejected=len(rej), events=len(df), wall_s=round(time.perf_counter() - t0, 1)))
            tab = build_cycle_table(vp, tr, ser, cyc)
            if len(tab):
                tables.append(tab)
            console.print(f"[{i + 1}/{len(w)}] {r.video}: треков {len(s)}, циклов {len(cyc)}, событий {len(df)}, {rows[-1]['wall_s']} с")
        except Exception as e:  # один сломанный ролик не должен останавливать пакет
            console.print(f"[red][{i + 1}/{len(w)}] {r.video}: {type(e).__name__}: {e}[/]")
            rows.append(dict(video=r.video, error=f"{type(e).__name__}: {e}"))
    out = OUTPUTS / "analysis"
    pd.DataFrame(rows).to_csv(out / "batch_summary.csv", index=False)
    if tables:
        (OUTPUTS / "dataset").mkdir(parents=True, exist_ok=True)
        pd.concat(tables, ignore_index=True).to_parquet(OUTPUTS / "dataset" / "cycles.parquet", index=False)
    console.print(f"сводка: {out / 'batch_summary.csv'}; таблица циклов: {OUTPUTS / 'dataset' / 'cycles.parquet'}")


@app.command("bench-pose", help="Бенчмарк моделей позы на кадрах из EDA: скорость и согласие с учителем (YOLO26x) на трёх «дальностях».")
def bench_pose_cmd(preset: Annotated[str, typer.Option(help="quick (torch+openvino+rtmlib) | full | torch | openvino | rtmlib | refine | hybrid | hybrid_ov | heavy")] = "quick",
                   n_frames: Annotated[int, typer.Option(help="Сколько кадров взять из outputs/analysis/frames")] = 20) -> None:
    import cv2  # noqa: F401
    import numpy as np
    import pandas as pd

    from . import bench as B

    fdir = OUTPUTS / "analysis" / "frames"
    frames = sorted(fdir.glob("*.jpg"))

    def mh(p: Path) -> float:  # медианная высота людей в имени файла: ..._n{число людей}_h{высота}.jpg
        return float(p.stem.split("_h")[-1])

    mid = [p for p in frames if 80 <= mh(p) <= 450] or frames       # «средние» люди (80–450 px): именно они важны
    pick = [mid[i] for i in np.linspace(0, len(mid) - 1, min(n_frames, len(mid))).astype(int)]

    def ux(w, sz, rt="torch", dev="cpu", refine=None):
        nm = f"{w.replace('-pose', '')}@{sz}{f'+ref{refine}' if refine else ''} {rt}{'' if rt == 'torch' else '-' + dev.split(':')[-1]}"
        return dict(name=nm, backend="ultralytics", weights=w, runtime=rt, device=dev, imgsz=sz, refine=refine)

    groups = {
        "torch": [ux("yolo26n-pose", 640), ux("yolo26n-pose", 960), ux("yolo26n-pose", 1280), ux("yolo26s-pose", 640), ux("yolo26s-pose", 960),
                  ux("yolo11n-pose", 960), ux("yolo11s-pose", 960)],
        "openvino": [ux("yolo26n-pose", 960, "openvino", "intel:cpu"), ux("yolo26n-pose", 960, "openvino", "intel:gpu"),
                     ux("yolo26s-pose", 960, "openvino", "intel:gpu"), ux("yolo26n-pose", 1280, "openvino", "intel:gpu")],
        "rtmlib": [dict(name="rtmlib lightweight (YOLOX-tiny+RTMPose-s)", backend="rtmlib", weights="rtmlib-body-lightweight", runtime="onnx", device="cpu", imgsz=""),
                   dict(name="rtmlib balanced (YOLOX-m+RTMPose-m)", backend="rtmlib", weights="rtmlib-body-balanced", runtime="onnx", device="cpu", imgsz="")],
        "refine": [ux("yolo26n-pose", 640, refine=320), ux("yolo26n-pose", 960, refine=320), ux("yolo26n-pose", 640, refine=384), ux("yolo26s-pose", 640, refine=320)],
        # гибрид: YOLO — детектор (+трекер), ключевые точки — RTMPose по рамкам (top-down)
        "hybrid": [ux("yolo26n-pose", 640, refine="rtmpose-s"), ux("yolo26n-pose", 960, refine="rtmpose-s"), ux("yolo26n-pose", 960, refine="rtmpose-m")],
        "hybrid_ov": [ux("yolo26n-pose", 960, "openvino", "intel:gpu", refine="rtmpose-s"), ux("yolo26s-pose", 960, "openvino", "intel:gpu", refine="rtmpose-s"),
                      ux("yolo26n-pose", 1280, "openvino", "intel:gpu", refine="rtmpose-s"), ux("yolo26n-pose", 960, "openvino", "intel:gpu", refine="rtmpose-m")],
        "heavy": [ux("yolo26m-pose", 960), ux("yolo11m-pose", 960), ux("yolo26s-pose", 1280), ux("yolo26m-pose", 960, "openvino", "intel:gpu"),
                  ux("yolo26s-pose", 960, "openvino", "intel:cpu")],
    }
    names = {"quick": ["torch", "openvino", "rtmlib"], "full": ["torch", "openvino", "rtmlib", "refine", "hybrid", "hybrid_ov", "heavy"]}.get(preset, [preset])
    specs = [s for n in names for s in groups[n]]
    teacher = dict(name="teacher", backend="ultralytics", weights="yolo26x-pose", runtime="torch", device="cpu", imgsz=1280)
    a = OUTPUTS / "analysis"
    out_csv = a / ("bench_pose.csv" if preset in ("quick", "full") else f"bench_pose_{preset}.csv")
    p, cb = _progress("бенчмарк поз-моделей")
    try:
        B.bench_pose(pick, specs, teacher, out_csv, progress=cb)
    finally:
        p.stop()
    parts = [pd.read_csv(f) for f in sorted(a.glob("bench_pose*.csv")) if f.name != "bench_pose_all.csv"]
    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["name", "scale"], keep="last")
    df.to_csv(a / "bench_pose_all.csv", index=False)
    B.plot_pose_bench(df, ROOT / "docs" / "img" / "bench_pose.png")
    cols = ["name", "scale", "fps", "person_recall", "extra_per_frame", "wrist_ok35", "wrist_err_med", "nose_err_med", "wrist_conf_mean", "error"]
    console.print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    console.print(f"{out_csv}\n{a / 'bench_pose_all.csv'}\n{ROOT / 'docs' / 'img' / 'bench_pose.png'}")


@app.command(help="Проверить внешние датасеты в data_external/ (скачанные вручную: Roboflow, Kaggle, HMDB51 ...): структура, число файлов, классы, лицензия.")
def datasets() -> None:
    from .external import EXTERNAL, scan_external

    rows = scan_external()
    if not rows:
        console.print(f"[yellow]{EXTERNAL} пуст или не существует.[/] Что и как положить — docs/DATA_AND_WEIGHTS.md")
        return
    _tbl("Внешние датасеты", ["источник", "имя", "картинок", "видео", "МБ", "YOLO", "классы", "сплиты", "лицензия", "замечания"],
         [[r["source"], r["name"], r["images"], r["videos"], r["size_mb"], "да" if r["yolo"] else "", r["classes"], r["splits"], r["license"], "; ".join(r["issues"])] for r in rows])


@app.command(help="Калибровка автомата циклов: перебор th_in/th_out/hand_extend по готовым запускам (поза не пересчитывается).")
def calibrate(th_in: Annotated[str, typer.Option(help="значения th_in через запятую")] = "0.35,0.6,0.9,1.2",
              gap: Annotated[str, typer.Option(help="th_out = th_in + gap, значения через запятую")] = "0.2,0.35",
              k: Annotated[str, typer.Option(help="hand_extend, значения через запятую")] = "0,0.5",
              pose_sig: Annotated[Optional[str], typer.Option(help="подстрока в имени запуска (напр. rtmpose-s)")] = None) -> None:
    import itertools

    import pandas as pd

    from . import calibrate as C

    grid = [(float(a), round(float(a) + float(g), 3), float(kk)) for a, g, kk in itertools.product(th_in.split(","), gap.split(","), k.split(","))]
    p, cb = _progress("перебор порогов")
    try:
        df = C.sweep(grid, pose_sig, cb)
    finally:
        p.stop()
    out = OUTPUTS / "analysis"
    df.to_csv(out / "calibrate_sweep.csv", index=False)
    pv = C.summarize_sweep(df)
    pv.to_csv(out / "calibrate_summary.csv")
    console.print("циклы и циклы/мин (на человека ≥80 px) по папкам; `smoking` — папка «курение», `fake` — «лжекурение»:")
    console.print(pv.to_string())
    ev = C.eval_vs_intervals(grid, pose_sig=pose_sig)
    if len(ev):
        ev.to_csv(out / "calibrate_vs_intervals.csv", index=False)
        agg = ev.groupby(["target", "th_in", "th_out", "k"]).agg(интервалов=("intervals", "sum"), найдено=("hit", "sum"), циклов=("cycles", "sum"),
                                                                  циклов_вне=("cycles_outside", "sum")).reset_index()
        agg["recall"] = (agg["найдено"] / agg["интервалов"]).round(2)
        console.print("против визуальных интервалов (docs/review/visual_intervals.csv):")
        console.print(agg.to_string(index=False))


@app.command("bench-track", help="Сравнение трекеров переигрыванием по сохранённым сырым детекциям (без повторного инференса позы).")
def bench_track_cmd(min_people: Annotated[float, typer.Option(help="брать окна, где 95-й перцентиль числа людей ≥ этого значения")] = 2.0,
                    set_: SetOpt = None) -> None:
    import pandas as pd

    from . import bench as B
    from .tracks import Tracks

    cfg = load_config(None, set_ or [])
    variants = [("botsort буфер 3 с", dict(tracker="botsort", buffer_sec=3.0)), ("botsort буфер 1 с", dict(tracker="botsort", buffer_sec=1.0)),
                ("botsort буфер 6 с", dict(tracker="botsort", buffer_sec=6.0)), ("bytetrack буфер 3 с", dict(tracker="bytetrack", buffer_sec=3.0)),
                ("bytetrack буфер 1 с", dict(tracker="bytetrack", buffer_sec=1.0)), ("ocsort буфер 3 с", dict(tracker="ocsort", buffer_sec=3.0)),
                ("fasttrack буфер 3 с", dict(tracker="fasttrack", buffer_sec=3.0))]
    rows = []
    runs = sorted((OUTPUTS / "runs").glob("*/*/pose/raw_dets.parquet"))
    console.print(f"окон с сырыми детекциями: {len(runs)}")
    for f in runs:
        raw, ft = pd.read_parquet(f), pd.read_parquet(f.parent / "frame_t.parquet")
        big = raw[(raw.conf >= 0.4) & ((raw.y2 - raw.y1) >= 80)]
        if len(big) == 0 or big.groupby("frame").size().quantile(0.95) < min_people:
            continue
        vid = f.parents[2].name
        for name, kw in variants:
            try:
                df = B.replay_tracking(raw, ft, cfg, **kw)
                m = B.track_metrics(df, raw, ft, cfg)
                rows.append(dict(video=vid, variant=name, **m))
            except Exception as e:
                rows.append(dict(video=vid, variant=name, error=f"{type(e).__name__}: {str(e)[:80]}"))
    d = pd.DataFrame(rows)
    d.to_csv(OUTPUTS / "analysis" / "bench_track_windows.csv", index=False)
    if d.empty:
        console.print("[yellow]нет подходящих окон (запустите sd batch)[/]")
        raise typer.Exit(0)
    ok = d[d.get("error").isna()] if "error" in d else d
    agg = ok.groupby("variant").agg(окон=("video", "count"), треков=("tracks", "sum"), людей=("n_people", "sum"), треков_на_человека=("tracks_per_person", "mean"),
                                    склеек=("stitched", "sum"), дубликатов=("dup_frames", "sum"), медиана_длины_с=("median_len_s", "mean"),
                                    покрытие=("coverage", "mean")).round(2)
    agg.to_csv(OUTPUTS / "analysis" / "bench_track.csv")
    console.print(agg.to_string())
    if "error" in d and d.error.notna().any():
        console.print("ошибки:", d[d.error.notna()][["variant", "error"]].drop_duplicates().to_string(index=False))


@app.command(help="Собрать manifest.json (руководство §10.2): версии, веса с sha256, лицензии, пороги, железо.")
def manifest(out: Annotated[Path, typer.Option()] = ROOT / "manifest.json", set_: SetOpt = None) -> None:
    from . import __version__
    from . import hw as hwm
    from . import models as M

    cfg = load_config(None, set_ or [])
    lock = M._read_lock()
    reg = M.load_registry()
    used = [cfg["pose"]["weights"]] + list(cfg["evidence"]["detectors"])
    comps = []
    for i in used:
        s = reg.get(i)
        if s is None:
            continue
        comps.append(dict(name=s.kind, id=i, weights=s.file or s.hf_repo or s.rtmlib, sha256=lock.get(i, {}).get("sha256"), license=s.license,
                          trust=s.trust, source=s.url or s.hf_repo))
    if cfg["pose"]["refine"]["enabled"] and str(cfg["pose"]["refine"]["method"]).startswith("rtmpose"):
        tag = "rtmpose-s" if cfg["pose"]["refine"]["method"].endswith("s") else "rtmpose-m"
        ck = Path.home() / ".cache" / "rtmlib" / "hub" / "checkpoints"
        for f in sorted(ck.glob(f"{tag}*.onnx")):
            comps.append(dict(name="keypoints (top-down, rtmlib)", id=tag, weights=f.name, sha256=M.sha256_file(f), license="Apache-2.0 (код rtmlib); веса OpenMMLab — лицензия в репозитории mmpose",
                              trust="official", source="download.openmmlab.com / hf Tau-J/RTMPose"))
    info = hwm.collect()
    m = dict(team="<название команды>", model_version=f"sd-{__version__}", confidence_threshold=cfg["events"]["confidence_threshold"],
             merge_rules=dict(same_person_gap_sec=cfg["events"]["merge_gap_sec"], positive_window_sec=cfg["events"]["window_sec"],
                              hold_sec=[cfg["cycles"]["hold_min_sec"], cfg["cycles"]["hold_max_sec"]], min_person_height_px=cfg["video"]["min_person_height_px"],
                              window_mode=cfg["events"]["window_mode"]),
             cycle_thresholds=dict(th_in=cfg["cycles"]["th_in"], th_out=cfg["cycles"]["th_out"], cycle_th=cfg["events"]["cycle_th"]),
             components=comps, hardware=f"{info['cpu']['name']}, {info['ram']['total_gb']} GB RAM, GPU: {[g.get('Name') for g in info['gpu_windows']]}",
             processing_fps=cfg["video"]["process_fps"], created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), config=cfg.to_dict(),
             note="ЧЕРНОВИК: порог confidence_threshold и cycle_th фиксируются после выбора на своей валидации и ДО скрытого набора")
    out.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    console.print(f"manifest -> {out}")


@app.command(help="Таблица циклов с признаками по всем запускам + метки (ручные и слабые). Печатает статистику.")
def dataset(use_weak: Annotated[bool, typer.Option(help="Включать слабые метки по папке видео")] = True) -> None:
    import pandas as pd

    from .dataset import attach_labels, load_labels

    f = OUTPUTS / "dataset" / "cycles.parquet"
    if not f.exists():
        console.print("[red]нет outputs/dataset/cycles.parquet — сначала `sd batch` или вкладка «Обучение» в UI[/]")
        raise typer.Exit(1)
    t = attach_labels(pd.read_parquet(f), use_weak=use_weak)
    console.print(f"циклов {len(t)} в {t.group.nunique()} видео; ручных меток: {len(load_labels())}")
    _tbl("Метки", ["папка", "источник", "метка", "n"], [[a, b, c, n] for (a, b, c), n in t.groupby(["weak_label", "label_source", "label"], dropna=False).size().items()])


@app.command(help="Обучить классификатор циклов (LightGBM + групповая CV по видео + изотоническая калибровка + абляция групп признаков).")
def train(use_weak: Annotated[bool, typer.Option(help="Слабые метки по папке (только для проверки конвейера!)")] = True,
          labels: Annotated[Optional[Path], typer.Option(help="CSV меток: формат проекта (labels/cycle_labels.csv) или визуальный (video,tid,start,label); по умолчанию labels/cycle_labels.csv")] = None,
          models: Annotated[bool, typer.Option("--models/--no-models", help="присоединить признаки D/E из outputs/analysis (предмет, X-CLIP, VideoMAE)")] = False,
          vlm: Annotated[Optional[Path], typer.Option(help="parquet результата `sd vlm-eval` — добавить признаки группы G (оценка VLM)")] = None,
          out: Annotated[Path, typer.Option()] = ROOT / "models" / "cycle") -> None:
    import pandas as pd

    from . import dataset as D
    from .train import train_cycle_model

    t = pd.read_parquet(OUTPUTS / "dataset" / "cycles.parquet")
    an = OUTPUTS / "analysis"
    if models:
        for fn, kw in (("evidence_cycles.parquet", "evidence"), ("tube_xclip_cycles.parquet", "tube"), ("tube_videomae_cycles.parquet", "tube")):
            if (an / fn).exists():
                t = D.attach_model_features(t, **{kw: pd.read_parquet(an / fn)})
    if vlm:
        t = D.attach_model_features(t, vlm=pd.read_parquet(vlm))
    lab = None
    if labels:
        head = pd.read_csv(labels, nrows=1)
        lab = pd.read_csv(labels) if {"cx", "cy"} <= set(head.columns) else D.visual_labels_to_project(labels, t)   # формат проекта хранит положение рта (cx, cy)
    t = D.attach_labels(t, use_weak=use_weak, labels=lab)
    res = train_cycle_model(t, out)
    if "error" in res:
        console.print(f"[red]{res['error']}[/]")
        raise typer.Exit(1)
    console.print(f"циклов с метками {res['n_samples']} (затяжек {res['n_pos']}), видео {res['n_groups']}; оценка на: {res['eval_on']}; признаков {len(res['features'])}; "
                  f"{'ТОЛЬКО слабые метки — для сдачи не годится' if res['weak_only'] else 'есть ручные метки'}")
    fmt = lambda v: "—" if v is None else f"{v:.3f}"   # noqa: E731
    _tbl("Качество out-of-fold (фолды по видео)", ["модель", "n", "AUC", "PR-AUC", "лучший F1", "порог"],
         [[k, m.get("n"), fmt(m.get("auc")), fmt(m.get("pr_auc")), fmt(m.get("f1")), fmt(m.get("th"))] for k, m in res["metrics"].items()])
    if res.get("ablation"):
        _tbl("Абляция: AUC без группы признаков", ["без группы", "AUC", "PR-AUC"], [[k, fmt(m.get("auc")), fmt(m.get("pr_auc"))] for k, m in res["ablation"].items()])
    console.print("важность признаков (gain): " + ", ".join(f"{k} {v:g}" for k, v in list(res["importance_gain"].items())[:8]))
    console.print(f"модель и отчёт: {out}")


@app.command(help="Визуальная проверка: монтажи кадров по циклам (--what cycles) или ленты кропа человека по времени (--what timeline).")
def review(video: VideoArg, start: Start = 0.0, end: End = None, set_: SetOpt = None, config: CfgOpt = None,
           what: Annotated[str, typer.Option(help="cycles | timeline")] = "cycles",
           tid: Annotated[Optional[int], typer.Option(help="Трек (для timeline; по умолчанию самый длинный)")] = None,
           step: Annotated[float, typer.Option(help="Шаг ленты, с")] = 1.0,
           max_cycles: Annotated[int, typer.Option(help="Сколько циклов показать")] = 16) -> None:
    from . import stages
    from .review import cycle_montage, timeline_sheet

    vid, cfg, tr, rd = _load_chain(video, start, end, set_, config)
    ser = stages.stage_features(tr, cfg, rd)
    cyc, rej, st = stages.stage_cycles(ser, cfg, rd)
    out = rd / "review"
    if what == "timeline":
        s = tr.summary()
        t_id = tid if tid is not None else int(s.sort_values("dur", ascending=False).tid.iloc[0])
        p = timeline_sheet(vid, tr, t_id, start, end if end is not None else float(tr.frame_t.t.max()), out / f"timeline_tid{t_id}.png", step=step,
                           series=ser, cycles=cyc)
        console.print(f"лента: {p}")
    else:
        if cyc.empty:
            console.print("[yellow]циклов нет[/]")
            raise typer.Exit(0)
        sub = cyc.sort_values("hold", ascending=False).head(max_cycles) if len(cyc) > max_cycles else cyc
        paths = cycle_montage(vid, tr, sub.sort_values("start"), out / "cycles", series=ser, title=vid.stem)
        for p in paths:
            console.print(f"монтаж: {p}")


@app.command("vlm-bench", help="Замер VLM-верификатора: задержка на кандидата (6 кадров кропа человека) и ответ в формате JSON.")
def vlm_bench(video: VideoArg, start: Start = 0.0, end: End = None, set_: SetOpt = None, config: CfgOpt = None,
              backend: Annotated[str, typer.Option(help="ollama | transformers | openai (llama.cpp/LM Studio сервер)")] = "transformers",
              repo: Annotated[str, typer.Option(help="HF-репозиторий для transformers")] = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
              model: Annotated[str, typer.Option(help="тег модели Ollama, например qwen3.5:2b-q4_K_M")] = "qwen3.5:2b-q4_K_M",
              mode: Annotated[str, typer.Option(help="Ollama: yesno | letter | json (ответ одним токеном с оценкой по логитам / по схеме)")] = "yesno",
              base_url: Annotated[Optional[str], typer.Option(help="URL сервера (Ollama: http://127.0.0.1:11434, openai: http://127.0.0.1:8080/v1)")] = None,
              tid: Annotated[Optional[int], typer.Option()] = None, t0: Annotated[Optional[float], typer.Option(help="начало окна, с")] = None,
              t1: Annotated[Optional[float], typer.Option()] = None, frames: int = 6, size: int = 336, n: Annotated[int, typer.Option(help="сколько окон")] = 3) -> None:
    import cv2
    import psutil

    from . import stages
    from .tube import tube_frames
    from .vlm import smoke_score

    vid, cfg, tr, rd = _load_chain(video, start, end, set_, config)
    ser = stages.stage_features(tr, cfg, rd)
    cyc, _, _ = stages.stage_cycles(ser, cfg, rd)
    if t0 is not None:
        wins = [(tid if tid is not None else tr.tids[0], t0, t1 if t1 is not None else t0 + 2.5)]
    else:
        wins = [(int(r.tid), float(r.peak_t - 1.25), float(r.peak_t + 1.25)) for r in cyc.head(n).itertuples()]
    if not wins:
        console.print("[yellow]нет окон: циклов нет (ослабьте пороги) — задайте --t0/--t1[/]")
        raise typer.Exit(0)
    cfg_v = load_config(config, (set_ or []) + [f"tube.size={size}"])
    proc = psutil.Process()
    t_load = time.perf_counter()
    vlm = _make_vlm(backend, repo, model, base_url, size, mode=mode)
    console.print(f"модель загружена за {time.perf_counter() - t_load:.1f} с; RSS процесса {proc.memory_info().rss / 2**30:.2f} ГБ")
    rows, secs = [], []
    for (t_id, a, b) in wins:
        fr, ts, raw = tube_frames(vid, tr, t_id, a, b, cfg_v, n=frames)
        if not fr:
            console.print(f"[yellow]окно ID{t_id} {a:.1f}–{b:.1f}: трек не покрывает окно[/]")
            continue
        ans = vlm.verify(fr, b - a)
        secs.append(ans['ms'] / 1000)
        sc = smoke_score(ans)
        rows.append([t_id, f"{a:.1f}–{b:.1f}", ans["action"], ans.get("object_visible"), ans.get("smoke_visible"), "—" if sc != sc else f"{sc:.2f}",
                     f"{ans['ms'] / 1000:.1f}" + (f" (prefill {ans['prefill_ms'] / 1000:.1f})" if ans.get("prefill_ms") else ""), ans.get("prompt_tokens", "-"),
                     ans.get("new_tokens", "-"), ans["raw"][:60].replace("\n", " ")])
    _tbl(f"VLM {repo if backend == 'transformers' else model if backend == 'ollama' else base_url}: {frames} кадров {size}px на кандидата",
         ["ID", "окно, с", "action", "предмет", "дым", "p_smoke", "время, с", "токенов промпта", "новых", "сырой ответ"], rows)
    if rows:
        console.print(f"среднее время на кандидата: {sum(secs) / len(secs):.1f} с; пик RSS {proc.memory_info().peak_wset / 2**30 if hasattr(proc.memory_info(), 'peak_wset') else proc.memory_info().rss / 2**30:.2f} ГБ")


def _make_vlm(backend: str, repo: str, model: str, base_url: str | None, size: int, num_ctx: int = 4096, mode: str = "yesno"):
    from .vlm import OllamaVLM, OpenAICompatVLM, TransformersVLM

    if backend == "ollama":
        from .ollama_ctl import BASE as OLLAMA_BASE

        return OllamaVLM(model, base_url or OLLAMA_BASE, size=size, mode=mode, num_ctx=num_ctx)
    if backend == "openai":
        return OpenAICompatVLM(base_url or "http://127.0.0.1:8080/v1", model=model, size=size)
    return TransformersVLM(repo, size)


@app.command("vlm-eval", help="VLM-верификатор на размеченных циклах: AUC оценки «затяжка» против меток, решение при 0.5, время на цикл. Результат дописывается по циклам — прогон можно прервать и продолжить.")
def vlm_eval_cmd(backend: Annotated[str, typer.Option(help="ollama | transformers | openai")] = "ollama",
                 model: Annotated[str, typer.Option(help="тег модели Ollama, например qwen3.5:2b-q4_K_M")] = "qwen3.5:2b-q4_K_M",
                 repo: Annotated[str, typer.Option(help="HF-репозиторий для transformers")] = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                 base_url: Annotated[Optional[str], typer.Option(help="URL сервера (по умолчанию Ollama: http://127.0.0.1:11434)")] = None,
                 labels: Annotated[Path, typer.Option(help="CSV меток циклов")] = ROOT / "docs" / "review" / "assistant_visual_labels.csv",
                 modes: Annotated[str, typer.Option(help="Ollama: режимы через запятую (yesno, letter, json); каждый режим на окне стоит полный prefill (кэш между разными вопросами не работает); letter не лучше yesno")] = "yesno",
                 frames: Annotated[int, typer.Option(help="кадров на окно")] = 6, size: Annotated[int, typer.Option(help="сторона кадра, px (кропы «трубки» нативно 224)")] = 224,
                 window: Annotated[float, typer.Option(help="длина окна вокруг пика цикла, с")] = 2.5,
                 per_class: Annotated[Optional[int], typer.Option(help="скрининг: взять по N циклов каждого класса (по кругу по видео); без опции — все")] = None,
                 num_ctx: Annotated[int, typer.Option(help="контекст Ollama, токенов")] = 4096,
                 crop: Annotated[str, typer.Option(help="какие кадры показывать модели: mouth (крупно рот и активная кисть, лучше) | tube (верх тела человека целиком)")] = "mouth",
                 mouth_scale: Annotated[float, typer.Option(help="для crop=mouth: сторона кропа в ширинах плеч")] = 2.5,
                 all_cycles: Annotated[bool, typer.Option("--all-cycles", help="прогнать ВСЕ найденные циклы, включая неразмеченные («неясно»): нужно для `sd event-check --vlm`; метрики считаются по размеченным")] = False,
                 tag: Annotated[Optional[str], typer.Option(help="суффикс имени файла результата")] = None,
                 n_boot: int = 500) -> None:
    import re

    from . import feature_auc as FA
    from . import vlm_eval as VE

    tab = FA.feature_table(labels)
    if tab.empty or "start" not in tab:
        console.print("[red]нет размеченных циклов, найденных в таблице циклов[/]")
        raise typer.Exit(1)
    tab = tab[tab["start"].notna()].copy()
    if all_cycles:
        tab = FA.all_cycles_features().merge(tab[["video", "tid", "start", "label", "y"]], on=["video", "tid", "start"], how="left")
    sub = VE.pick_subset(tab, per_class).sort_values(["run", "start"]) if "run" in tab else VE.pick_subset(tab, per_class)
    name = re.sub(r"[^0-9A-Za-z._-]+", "_", model if backend != "transformers" else repo.split("/")[-1])
    out = OUTPUTS / "analysis" / f"vlm_{name}_f{frames}_s{size}{'_mouth' if crop == 'mouth' else ''}{('_' + tag) if tag else ''}.parquet"
    console.print(f"циклов к проверке: {len(sub)} (затяжек {int((sub.y == 1).sum())}, не-затяжек {int((sub.y == 0).sum())}, без метки {int(sub.y.isna().sum())}); результат: {out}")
    names = [m.strip() for m in modes.split(",") if m.strip()] if backend == "ollama" else ["single"]
    vlms = {n: _make_vlm(backend, repo, model, base_url, size, num_ctx, n if backend == "ollama" else "yesno") for n in names}
    p, cb = _progress("VLM по циклам")
    try:
        res = VE.run_vlm(sub, vlms, out, frames, window, cb, crop, mouth_scale, size)
    finally:
        p.stop()
    evs = {n: VE.evaluate(sub[sub.y.notna()], res, n_boot, n) for n in VE.modes_in(res)}
    keys = list(dict.fromkeys(k for e in evs.values() for k in e))
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)   # noqa: E731
    _tbl(f"VLM {model if backend != 'transformers' else repo}: качество на размеченных циклах ({frames} кадров {size} px)", ["показатель"] + list(evs),
         [[k] + [fmt(e.get(k, "")) for e in evs.values()] for k in keys])


@app.command("clips-run", help="Конвейер по папке клипов «папка = класс» (data/ или data_external/<источник>/<датасет>): сводка по классам, таблица циклов для обучения.")
def clips_run(root: Annotated[Path, typer.Argument(help="папка вида <root>/<класс>/*.mp4, например data или data_external/hmdb51/hmdb51_org")],
              per_class: Annotated[Optional[int], typer.Option(help="не больше N клипов на класс (детерминированная выборка)")] = None,
              max_sec: Annotated[float, typer.Option(help="анализировать первые N секунд каждого клипа")] = 15.0, start: Start = 0.0,
              only_class: Annotated[Optional[str], typer.Option(help="только эти классы-папки, через запятую")] = None,
              positive: Annotated[Optional[str], typer.Option(help="имена папок «курение» через запятую (по умолчанию smoking, smoke, smoking_hookah, курение)")] = None,
              tag: Annotated[Optional[str], typer.Option(help="имя набора: результаты в outputs/external/<tag>")] = None,
              cycle_bundle: Annotated[Optional[Path], typer.Option(help="пакет классификатора цикла (models/cycle/<имя>): события по его оценкам; колонка «по правилам» — по длительности паузы, для сравнения")] = None,
              set_: SetOpt = None, config: CfgOpt = None, force: Force = False) -> None:
    from . import clips as C
    from .bundle import Bundle

    cfg = load_config(config, set_ or [])
    root = root if root.is_absolute() else (ROOT / root)
    pos = {x.strip().lower() for x in positive.split(",")} if positive else None
    only = {x.strip() for x in only_class.split(",")} if only_class else None
    n = len(C.pick_clips(root, per_class, only))
    if not n:
        console.print(f"[red]в {root} нет видео вида <класс>/*.mp4[/]")
        raise typer.Exit(1)
    console.print(f"клипов к прогону: {n} (первые {max_sec:g} с каждого)")
    t0 = time.perf_counter()
    bundle = Bundle(cycle_bundle if cycle_bundle.is_absolute() else ROOT / cycle_bundle) if cycle_bundle else None
    df, cyc = C.run_clips(root, cfg, per_class, max_sec, start, only, pos, tag, force,
                          lambda i, k, name: console.print(f"[{i}/{k}] {name} ({time.perf_counter() - t0:.0f} с)"), bundle=bundle)
    summ = C.summarize(df, cfg["cycles"]["th_in"])
    has_rules = "with_event_rules" in summ
    has_reach = "reach_th_in" in summ
    _tbl("Сводка по классам (метка клипа = имя папки)" + ("; события — по классификатору, «по правилам» — по длительности паузы" if has_rules else ""),
         ["класс", "метка", "клипов", "доля с циклом", "доля с событием"] + (["…по правилам"] if has_rules else []) + ["циклов/мин на человека", "рост человека, px"]
         + (["рука достаёт до рта (d < th_in)", "достоверность запястий"] if has_reach else []) + ["ошибок"],
         [[r.cls, r.label, r.clips, f"{r.with_cycle:.0%}", f"{r.with_event:.0%}"] + ([f"{r.with_event_rules:.0%}"] if has_rules else []) + [f"{r.cycles_per_person_min:.2f}", f"{r.h_med_px:.0f}"]
          + ([f"{r.reach_th_in:.0%}", f"{r.wrist_conf_med:.2f}"] if has_reach else []) + [r.errors] for r in summ.itertuples()])
    out = OUTPUTS / "external" / (tag or root.name)
    if bundle is not None:
        ca = C.class_auc(df, cyc)
        console.print(f"оценка клипа = максимальная оценка цикла (нет циклов = 0); «курение» против остальных: AUC {ca['clip_auc']:.3f} [{ca['lo']:.2f}–{ca['hi']:.2f}] (95% по клипам; клипов {ca['clips']}, "
                      f"из них курение {ca['positives']}); оценка ≥ 0.5 у {ca.get('positive_share_ge_05', float('nan')):.0%} клипов «курение»; AUC по циклам (метка клипа, шум меток): {ca['cycle_auc']:.3f}")
        if ca["per_class"]:
            _tbl("«Курение» против каждого негативного класса (уровень клипа)", ["класс", "клипов", "AUC", "медиана оценки", "доля клипов с оценкой ≥ 0.5"],
                 [[r["cls"], r["n"], f"{r['auc']:.3f}", f"{r['score_median']:.2f}", f"{r['share_ge_05']:.0%}"] for r in sorted(ca["per_class"], key=lambda r: r["auc"])])
        (out / "class_auc.json").write_text(json.dumps(ca, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    console.print(f"клипы: {out / 'clips.csv'}; циклы для обучения: {out / 'cycles.parquet'}")


@app.command("fetch-videos", help="Скачать выборку открытых видео БЕЗ входа в аккаунт в data_external/<источник>/<набор>/<класс>/клип.mp4 (формат `sd clips-run`). Пока hmdb51: зеркало Hugging Face divm/hmdb51, "
                                  "по умолчанию классы smoke, drink, eat, chew, talk, smile (≈ 83 МБ целиком). Kinetics/AVA/CDAD — вручную, см. docs/NEXT_STEPS.md.")
def fetch_videos_cmd(name: Annotated[str, typer.Argument(help="hmdb51")] = "hmdb51",
                     classes: Annotated[Optional[str], typer.Option(help="классы через запятую (имена HMDB51: smoke, drink, eat, chew, talk, smile, laugh, kiss, pour…)")] = None,
                     per_class: Annotated[Optional[int], typer.Option(help="не больше N клипов на класс (детерминированная выборка)")] = None) -> None:
    from . import video_fetch as VF

    if name not in VF.VIDEO_SOURCES:
        console.print(f"[red]неизвестный источник {name}[/]; доступны: {', '.join(VF.VIDEO_SOURCES)}")
        raise typer.Exit(1)
    res = VF.fetch(name, [c.strip() for c in classes.split(",")] if classes else None, per_class)
    console.print(res)
    console.print(f"лицензия: {Path(res['dest']) / 'LICENSE.txt'}; дальше: sd clips-run {_rel(Path(res['dest']))} --max-sec 10 --cycle-bundle models/cycle/cycle_fast --tag {name}")


@app.command("event-check", help="Сквозная проверка «циклы → оценка цикла → события → протокол оценки» на текущих данных по эталону из визуальных меток (механика и порядок величин, не качество).")
def event_check_cmd(labels: Annotated[Path, typer.Option(help="CSV меток циклов (эталон строится из них)")] = ROOT / "docs" / "review" / "assistant_visual_labels.csv",
                    vlm: Annotated[Optional[Path], typer.Option(help="parquet из `sd vlm-eval`: добавить оценку VLM к признакам модели")] = None) -> None:
    from . import event_check as EC

    df, info = EC.run_check(labels, vlm)
    console.print(f"клипов {info['clips']} (негативных {info['negative_clips']}, {info['negative_hours'] * 60:.1f} мин); циклов {info['cycles']}, размечено {info['labelled']}; "
                  f"эталонных событий {info['gt_events']}, интервалов IGNORE {info['ignore_intervals']}")
    console.print("признаки модели: " + ", ".join(info["features"]))
    fmt = lambda v: "—" if v is None or v != v else f"{v:.2f}"   # noqa: E731
    _tbl("События против эталона из меток (метод × порог оценки цикла; «rules» — только правило длительности паузы, «classifier» — логрегрессия leave-one-video-out)",
         ["метод", "порог", "событий", "TP", "FP", "FN", "precision", "recall", "F1", "FP в негативных клипах", "FP/час негатива"],
         [[r.method, r.cycle_th, r.events, r.tp, r.fp, r.fn, fmt(r.precision), fmt(r.recall), fmt(r.f1), r.fp_in_negative_clips, fmt(r.fp_per_negative_hour)] for r in df.itertuples()])
    out = OUTPUTS / "analysis" / "event_check.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    console.print(f"таблица: {out}")


@ollama_app.command("status", help="Версия, установленные и загруженные в память модели, свободная ОЗУ.")
def ollama_status() -> None:
    from . import ollama_ctl as O

    st = O.status()
    console.print(f"ollama: {st['exe'] or 'не найден'}; сервер {'работает, версия ' + str(st.get('version')) if st['up'] else 'НЕ запущен (sd.cmd ollama up)'}; "
                  f"ОЗУ свободно {st['free_ram_gb']} из {st['total_ram_gb']} ГБ")
    if st["models"]:
        _tbl("Установленные модели", ["модель", "размер, ГБ", "параметры", "квантизация"], [[m["name"], m["size_gb"], m["params"], m["quant"]] for m in st["models"]])
    if st["loaded"]:
        _tbl("В памяти сейчас", ["модель", "размер, ГБ", "из них в видеопамяти", "выгрузится"], [[m["name"], m["size_gb"], m["in_vram_gb"], m["expires"]] for m in st["loaded"]])


@ollama_app.command("up", help="Запустить сервер Ollama (127.0.0.1:11434, без облака, одна модель в памяти). --igpu: пробовать встроенную графику Intel (Vulkan, экспериментально).")
def ollama_up(igpu: Annotated[bool, typer.Option("--igpu", help="OLLAMA_IGPU_ENABLE=1: считать на встроенной графике (по умолчанию Ollama её отключает)")] = False) -> None:
    from . import ollama_ctl as O

    console.print(O.start({"OLLAMA_IGPU_ENABLE": "1"} if igpu else None))


@ollama_app.command("down", help="Остановить сервер и значок Ollama в трее.")
def ollama_down() -> None:
    from . import ollama_ctl as O

    console.print(f"остановлено процессов: {O.stop()}")


@ollama_app.command("pull", help="Загрузить модель, например: sd.cmd ollama pull qwen3.5:2b (сервер должен быть запущен).")
def ollama_pull(model: Annotated[str, typer.Argument(help="тег модели, например qwen3.5:2b-q4_K_M")]) -> None:
    from . import ollama_ctl as O

    p, cb = _progress(f"загрузка {model}")
    last = {"s": ""}

    def on(status: str, done, total) -> None:
        if total:
            cb(int(done or 0), int(total))
        if status != last["s"]:
            last["s"] = status
            p.update(0, description=f"{model}: {status[:40]}")

    try:
        O.pull(model, on)
    finally:
        p.stop()
    console.print(f"готово: {model}")


@ollama_app.command("unload", help="Выгрузить модели из памяти (освободить ОЗУ).")
def ollama_unload(model: Annotated[Optional[str], typer.Argument(help="тег модели; без аргумента — все загруженные")] = None) -> None:
    from . import ollama_ctl as O

    console.print("выгружено: " + (", ".join(O.unload(model)) or "ничего не было загружено"))


@app.command("resources", help="Что занимает память и процессор перед тяжёлым запуском (VLM, эмбеддинги): ОЗУ, подкачка, диск, самые тяжёлые процессы с пометками «подозрительно» (эвристики). "
                               "`--free-own` выгружает модели из Ollama (освобождает свои ресурсы); ЧУЖИЕ процессы команда не останавливает.")
def resources_cmd(free_own: Annotated[bool, typer.Option("--free-own", help="выгрузить модели из Ollama")] = False,
                  stop_ollama: Annotated[bool, typer.Option("--stop-ollama", help="вместе с --free-own остановить локальный сервер Ollama")] = False,
                  top: int = 10) -> None:
    from . import resources as RS

    snap = RS.snapshot(top)
    console.print(f"ОЗУ: свободно {snap['ram_available_gb']} из {snap['ram_total_gb']} ГБ; подкачка занята {snap['swap_used_gb']} ГБ; CPU {snap['cpu_percent']:.0f}%; диск свободно {snap['disk_free_gb']} ГБ; "
                  f"VLM {'помещается' if snap['enough_for_vlm'] else 'НЕ помещается без закрытия приложений'}")
    _tbl("Самые тяжёлые процессы", ["PID", "процесс", "МБ", "проекта", "подозрительно"], [[r["pid"], r["name"], r["mb"], "да" if r["own"] else "", "; ".join(r["suspicious"])[:90]] for r in snap["top"]])
    for a in RS.advice(snap):
        console.print(f"[yellow]! {a}[/]")
    if free_own:
        for line in RS.free_own(stop_ollama):
            console.print(line)


@app.command("doctor", help="Проверка окружения перед запуском на другой машине/в контейнере: пакеты против lock-файла, ускорители, веса, пакеты моделей, Ollama, данные. Код возврата 1, если критичное не в порядке.")
def doctor_cmd(json_out: Annotated[Optional[Path], typer.Option("--json", help="сохранить отчёт в JSON")] = None) -> None:
    from . import doctor as D

    r = D.check()
    console.print(D.render_report(r))
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        console.print(f"отчёт: {json_out}")
    raise typer.Exit(0 if r["critical_ok"] else 1)


@app.command("recognize", help="Полный цикл распознавания: видео → поза+трекинг → циклы → признаки → оценка цикла (пакеты моделей) → события по регламенту. Пишет outputs/recognize/<видео>/<окно>_<время>/ (result.json, cycles_scored.csv, events.csv/json, overlay.mp4).")
def recognize_cmd(video: VideoArg, start: Start = 0.0, end: End = None,
                  cycle_bundle: Annotated[Optional[Path], typer.Option(help="пакет классификатора цикла (models/cycle/<имя>); без него — эвристика-заглушка")] = None,
                  cycle_bundle_full: Annotated[Optional[Path], typer.Option(help="пакет с VLM-признаком (для --vlm-mode grey|all)")] = None,
                  photo_bundle: Annotated[Optional[Path], typer.Option(help="пакет фото-модели (models/photo/<имя>), даёт признаки photo_*")] = None,
                  objects: Annotated[Optional[str], typer.Option(help="детекторы предмета через запятую (smoking_yolo11m_beehzod,…)")] = None,
                  vlm: Annotated[Optional[str], typer.Option(help="тег Ollama (qwen3.5:2b-q4_K_M); сервер должен быть запущен: sd.cmd ollama up --igpu")] = None,
                  vlm_mode: Annotated[str, typer.Option(help="off | grey (только серая зона оценки дешёвого пакета) | all")] = "grey",
                  grey: Annotated[str, typer.Option(help="серая зона оценки: нижняя,верхняя")] = "0.3,0.8",
                  render: Annotated[bool, typer.Option("--render/--no-render", help="видео с разметкой overlay.mp4")] = True,
                  backend: Annotated[str, typer.Option(help="рантайм фото-модели: auto | ov | torch")] = "auto",
                  camera_id: str = "cam_local", set_: SetOpt = None, config: CfgOpt = None, force_pose: Annotated[bool, typer.Option("--force-pose", help="пересчитать позу")] = False) -> None:
    from . import pipeline as P

    vid, cfg = _setup(video, start, end, set_, config)
    lo, hi = (float(x) for x in grey.split(","))
    opts = P.Options(cycle_bundle=str(cycle_bundle) if cycle_bundle else None, cycle_bundle_full=str(cycle_bundle_full) if cycle_bundle_full else None,
                     photo_bundle=str(photo_bundle) if photo_bundle else None, objects=tuple(x.strip() for x in objects.split(",")) if objects else (),
                     vlm_model=vlm, vlm_mode=vlm_mode if vlm else "off", grey=(lo, hi), render=render, camera_id=camera_id, backend=backend)
    t0 = time.perf_counter()
    last: dict = {}

    def prog(stage: str, i: int, n: int) -> None:
        if stage != last.get("s") or i == n or time.perf_counter() - last.get("t", 0) > 20:
            console.print(f"  {stage}: {i}/{n} ({time.perf_counter() - t0:.0f} с)")
            last.update(s=stage, t=time.perf_counter())

    res = P.recognize(vid, start, end, cfg, opts, progress=prog, force_pose=force_pose)
    m = res.meta
    console.print(f"циклов {m['counts']['cycles']}, событий {m['counts']['events']}, вызовов VLM {m['counts']['vlm_calls']}; время этапов, с: {m['timing_sec']}")
    for w in m["warnings"]:
        console.print(f"[yellow]! {w}[/]")
    if len(res.events):
        _tbl("События", ["id", "начало", "конец", "confidence", "ID человека", "пик"],
             [[r.event_id, f"{r.start_sec:.1f}", f"{r.end_sec:.1f}", f"{r.confidence:.2f}", r.person_track_id, f"{r.peak_sec:.1f}"] for r in res.events.itertuples()])
    console.print(f"результаты: {res.out_dir}")


@app.command("analyze-cycles", help="Массовый анализ ВСЕХ циклов (текущие пороги): предмет на кропах (evidence), X-CLIP, VideoMAE, положение точек (pose), фото-классификатор (photo); пишет outputs/analysis/*_cycles.parquet.")
def analyze_cycles(what: Annotated[str, typer.Option(help="через запятую: evidence,xclip,videomae,pose,photo (по умолчанию первые три; pose дёшево, photo нужен --photo-bundle)")] = "evidence,xclip,videomae",
                   detectors: Annotated[str, typer.Option(help="id детекторов для evidence")] = "smoking_yolo11m_beehzod,smoking_yolo26s_basant18",
                   photo_bundle: Annotated[Optional[Path], typer.Option(help="папка пакета фото-модели (sd photos train), для --what photo")] = None,
                   backend: Annotated[str, typer.Option(help="photo: auto | ov | torch")] = "auto") -> None:
    from . import analysis as A

    cycles = A.collect_cycles()
    cycles.to_parquet(OUTPUTS / "analysis" / "cycles_all.parquet", index=False)
    console.print(f"циклов: {len(cycles)} в {cycles.video.nunique()} видео")
    todo = [w.strip() for w in what.split(",")]

    def pr(tag):
        t0 = time.perf_counter()
        return lambda i, n: console.print(f"{tag}: окно {i}/{n} ({time.perf_counter() - t0:.0f} с)")

    if "evidence" in todo:
        ev = A.evidence_all(cycles, [d.strip() for d in detectors.split(",")], progress=pr("evidence"))
        console.print(f"evidence: {len(ev)} циклов -> {OUTPUTS / 'analysis' / 'evidence_cycles.parquet'}")
    if "xclip" in todo:
        tb = A.tube_all(cycles, ("xclip",), out=OUTPUTS / "analysis" / "tube_xclip_cycles.parquet", progress=pr("xclip"))
        console.print(f"xclip: {len(tb)} окон; мс/окно {tb.attrs.get('ms') if hasattr(tb, 'attrs') else ''}")
    if "videomae" in todo:
        tb = A.tube_all(cycles, ("videomae",), out=OUTPUTS / "analysis" / "tube_videomae_cycles.parquet", progress=pr("videomae"))
        console.print(f"videomae: {len(tb)} окон")
    if "pose" in todo:
        pt = A.pose_all(progress=pr("pose"))
        console.print(f"pose: {len(pt)} циклов -> {OUTPUTS / 'analysis' / 'pose_cycles.parquet'}")
    if "photo" in todo:
        if photo_bundle is None:
            console.print("[red]--what photo: укажите --photo-bundle (папка пакета после `sd photos train`)[/]")
            raise typer.Exit(1)
        t0 = time.perf_counter()
        ph = A.photo_all(cycles, photo_bundle if photo_bundle.is_absolute() else ROOT / photo_bundle, backend,
                         progress=lambda tag, i, n: console.print(f"photo {tag}: {i}/{n} ({time.perf_counter() - t0:.0f} с)") if i == n or i % 20 == 0 else None)
        console.print(f"photo: {len(ph)} циклов -> {OUTPUTS / 'analysis' / 'photo_cycles.parquet'}")


@app.command("feature-auc", help="AUC признаков цикла (кинематика, предмет, X-CLIP, VideoMAE) на разметке глазами + грубая абляция групп; пишет outputs/analysis/feature_auc*.csv и docs/img/feature_auc.png.")
def feature_auc(labels: Annotated[Path, typer.Option(help="CSV разметки циклов (video,tid,start,label)")] = ROOT / "docs" / "review" / "assistant_visual_labels.csv",
                n_boot: Annotated[int, typer.Option(help="бутстрэп-перевыборок (по видео)")] = 500,
                top: Annotated[int, typer.Option(help="сколько признаков показать в таблице на группу")] = 8,
                vlm: Annotated[Optional[Path], typer.Option(help="parquet из `sd vlm-eval`: добавить группу G (оценка VLM); таблица сузится до циклов, которые VLM обработал")] = None,
                tag: Annotated[Optional[str], typer.Option(help="суффикс имён файлов результата (по умолчанию `vlm` при --vlm, иначе пусто — основные файлы)")] = None) -> None:
    from . import feature_auc as FA

    sfx = f"_{tag}" if tag else ("_vlm" if vlm else "")   # с VLM таблица уже (только обработанные циклы) — не затираем основные файлы
    tab = FA.feature_table(labels, vlm=vlm)
    n_lab = len(FA.load_labelled(labels))
    if tab.empty or "start" not in tab:
        console.print(f"[red]ни одна метка из {labels} не нашла цикл[/] (размечено {n_lab}); проверьте `sd.cmd analyze-cycles` и `sd.cmd dataset` для тех же порогов")
        raise typer.Exit(1)
    matched = int(tab["start"].notna().sum())
    console.print(f"размеченных циклов (без unsure): {n_lab}; нашли пару в таблице циклов: {matched}; курение {int(tab.y.sum())}, не курение {int((1 - tab.y).sum())}; видео {tab.video.nunique()}")
    tab = tab[tab["start"].notna()]
    res = FA.per_feature_auc(tab, n_boot)
    if res.empty:
        console.print("[red]нет признаков для оценки (нет parquet-файлов в outputs/analysis?)[/]")
        raise typer.Exit(1)
    cols = [c for c in res.columns if c.startswith("auc_vs_")]
    for g, sub in res.groupby("group", sort=False):
        _tbl(f"AUC признаков группы «{g}» (курение vs не курение; +/− = интервал по видео целиком выше/ниже 0.5)",
             ["признак", "AUC", "95% (по видео)", "знак", "n", "не нули"] + [c.replace("auc_vs_", "vs ") for c in cols],
             [[r.feature, f"{r.auc:.2f}", f"{r.lo:.2f}–{r.hi:.2f}", r.signal or "", r.n, f"{r.nonzero:.0%}"] + [f"{getattr(r, c):.2f}" for c in cols]
              for r in sub.head(top).itertuples()])
    out = OUTPUTS / "analysis" / f"feature_auc{sfx}.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    rc = FA.object_recall(tab)
    if len(rc):
        rcols = [c for c in rc.columns if c != "detector"]
        _tbl("Детектор предмета на кропе кисть–рот: доля циклов с «устойчиво виденным» предметом (≥ k из n кадров) по меткам; recall по затяжкам < 50% → только бонус (§3.4)",
             ["детектор"] + rcols, [[str(r["detector"]).replace("obj_", "").replace("_hit", "")] + [f"{r[c]:.0%}" for c in rcols] for _, r in rc.iterrows()])
        rc.to_csv(OUTPUTS / "analysis" / f"feature_object_recall{sfx}.csv", index=False, encoding="utf-8-sig")
    cc_all = FA.confound_check(tab)
    cc_all.to_csv(OUTPUTS / "analysis" / f"feature_confound{sfx}.csv", index=False, encoding="utf-8-sig")
    best = res.assign(dev=(res.auc - 0.5).abs()).sort_values("dev", ascending=False).groupby("group").head(3).feature   # 3 самых «сильных» признака каждой группы
    cc = cc_all[cc_all.feature.isin(best)]
    _tbl("Не выучил ли признак сцену вместо жеста: все затяжки из папки «курение», негативы в основном из «лжекурения». "
         "«папка среди негативов» заметно выше 0.5 → признак видит сцену, и его AUC в таблицах выше завышен",
         ["признак", "AUC все", "AUC внутри папки «курение» (затяжек / негативов)", "папка среди негативов (0.5 = не видит)"],
         [[r.feature, f"{r.auc_all:.2f}", f"{r.auc_in_smoking_dir:.2f} ({r.n_pos_dir} / {r.n_neg_dir})", f"{r.auc_dir_among_neg:.2f}"] for r in cc.itertuples()])
    ab = FA.group_ablation(tab)
    if len(ab):
        _tbl("Грубая абляция групп: логистическая регрессия, фолды по видео (AUC на отложенных циклах, среднее по перемешиваниям)",
             ["набор", "признаки", "AUC среднее", "мин–макс", "повторов"],
             [[r.combo, r.features, f"{r.auc_mean:.2f}", f"{r.auc_min:.2f}–{r.auc_max:.2f}", r.repeats] for r in ab.itertuples()])
        ab.to_csv(OUTPUTS / "analysis" / f"feature_auc_ablation{sfx}.csv", index=False, encoding="utf-8-sig")
    info = (f"{len(tab)} циклов: {int(tab.y.sum())} курение / {int((1 - tab.y).sum())} не курение, {tab.video.nunique()} видео; метки — оценка ассистента по кадрам, не эталон; "
            f"негативы в основном из другой папки видео")
    img = FA.plot_auc(res, ROOT / "docs" / "img" / f"feature_auc{sfx}.png", info=info)
    tab.to_csv(OUTPUTS / "analysis" / f"feature_table_labelled{sfx}.csv", index=False, encoding="utf-8-sig")
    console.print(f"таблицы: {out}; график: {img}")


@app.command("detector-eval", help="Проверка детектора предмета (сигарета/вейп): на YOLO-датасете (--data data.yaml: P/R/mAP по рамкам и AUC уровня картинки) и/или на фото-наборах «курит / не курит» (--photo-sets). "
                                   "Детектор — id из реестра (smoking_yolo11m_beehzod…) или путь к весам; `.pt` сканируются перед загрузкой.")
def detector_eval_cmd(detector: Annotated[str, typer.Argument(help="id из `sd models list` или путь к .pt/.onnx")] = "smoking_yolo11m_beehzod",
                      data: Annotated[Optional[Path], typer.Option(help="data.yaml YOLO-датасета (data_external/<источник>/<имя>/data.yaml)")] = None,
                      split: Annotated[str, typer.Option(help="val | test | train")] = "val",
                      photo_sets: Annotated[bool, typer.Option("--photo-sets", help="фото-наборы из outputs/photo/photos_audit.parquet (`sd photos audit`)")] = False,
                      per_source: Annotated[int, typer.Option(help="--photo-sets: снимков на набор (поровну курящих/нет)")] = 300,
                      sources: Annotated[Optional[str], typer.Option(help="--photo-sets: только эти наборы (через запятую), например stanford40_actions")] = None,
                      imgsz: int = 640, no_box: Annotated[bool, typer.Option("--no-box", help="только AUC уровня картинки (быстрее)")] = False) -> None:
    import pandas as pd

    from . import detector_eval as DE
    from .models import local_weights

    w = Path(detector) if Path(detector).exists() else local_weights(detector, allow_unknown=False)
    res: dict = dict(detector=str(detector))
    if data is not None:
        data = data if data.is_absolute() else ROOT / data
        im, lb = DE.split_dirs(data, split)
        lab = DE.image_labels(im, lb)
        console.print(f"{data.parent.name}/{split}: {len(lab)} картинок, с предметом {int(lab.positive.sum())}")
        fn = DE.yolo_predict_fn(w, imgsz)
        il = DE.image_level(lab, fn)
        res["image_level"] = {k: v for k, v in il.items() if k != "scores"}
        _tbl("Уровень картинки: «есть ли предмет»", ["порог", "recall", "FPR"], [[p["conf"], f"{p['recall']:.2f}", f"{p['fpr']:.2f}"] for p in il["points"]])
        console.print(f"AUC по максимальной уверенности: {il['auc']:.3f}")
        if not no_box:
            try:
                res["box"] = DE.box_metrics(w, data, split, imgsz)
                b = res["box"]
                console.print(f"по рамкам: P {b['precision']:.3f}  R {b['recall']:.3f}  mAP50 {b['map50']:.3f}  mAP50-95 {b['map50_95']:.3f}; классы детектора {b['detector_classes']}")
            except Exception as e:   # несовпадение классов/форматов — не повод терять результат уровня картинки
                console.print(f"[yellow]метрики по рамкам не посчитаны: {e}[/]")
    if photo_sets:
        f = OUTPUTS / "photo" / "photos_audit.parquet"
        if not f.exists():
            console.print("[red]нет outputs/photo/photos_audit.parquet — сначала `sd photos audit`[/]")
            raise typer.Exit(1)
        tbl = pd.read_parquet(f)
        if sources:
            tbl = tbl[tbl.source.isin([x.strip() for x in sources.split(",")])]
        pt = DE.photo_table(tbl, DE.yolo_predict_fn(w, imgsz), per_source)
        _tbl("Детектор на фото-наборах (максимальная уверенность на снимке)", ["набор", "снимков", "AUC", "recall @0.15", "FPR @0.15"],
             [[s, v["n"], f"{v['auc']:.3f}", f"{v['recall_015']:.2f}", f"{v['fpr_015']:.2f}"] for s, v in pt["per_source"].items()])
        _tbl("AUC против каждого негативного класса", ["набор", "класс", "n", "AUC", "FPR @0.15"], [[r["source"], r["neg_class"], r["n"], f"{r['auc']:.3f}", f"{r['fpr_015']:.2f}"] for r in sorted(pt["per_negative"], key=lambda r: r["auc"])])
        res["photo_sets"] = {k: v for k, v in pt.items() if k != "table"}
    # имя файла различает, на чём считали: прогон на YOLO-датасете не должен затирать результат на фото-наборах того же детектора
    tag = ((f"_{data.parent.name}_{split}" if data is not None else "") + (f"_{sources.replace(',', '+')}" if (photo_sets and sources) else "")
           + (f"_imgsz{imgsz}" if imgsz != 640 else ""))
    res["imgsz"] = imgsz
    out = OUTPUTS / "analysis" / f"detector_eval_{Path(str(detector)).stem}{tag}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    console.print(f"результат: {out}")


@app.command("start-eval", help="Качество детекции НАЧАЛА цикла (кандидатов-жестов): recall/precision генераторов (автомат при разных порогах и их объединения), точность старта, аудит случайных окон. "
                                "Эталон — метки жестов глазами по пулу кандидатов (docs/review/gesture_gt.csv, не эталон организаторов). Пишет outputs/analysis/start_eval_*.csv и docs/img/start_eval.png.")
def start_eval_cmd(gt: Annotated[Path, typer.Option(help="CSV меток жестов (video,tid,peak_t,label,start_quality)")] = ROOT / "docs" / "review" / "gesture_gt.csv",
                   audit: Annotated[Path, typer.Option(help="CSV меток случайных окон вне кандидатов")] = ROOT / "docs" / "review" / "gesture_audit_windows.csv",
                   rebuild: Annotated[bool, typer.Option(help="пересобрать пул кандидатов по запускам (иначе берётся outputs/analysis/gesture_pool.parquet)")] = False,
                   montages: Annotated[Optional[Path], typer.Option(help="каталог для монтажей НЕразмеченных кандидатов (по 6 на картинку, 7 кадров каждый) + gesture_gt_todo.csv для заполнения меток")] = None,
                   n_boot: int = 500) -> None:
    import pandas as pd

    from . import start_eval as SE

    if rebuild or not SE.POOL.exists():
        cyc = SE.generator_cycles()
        SE.build_pool(cyc).to_parquet(SE.POOL, index=False)
    pool = pd.read_parquet(SE.POOL)
    labels = pd.read_csv(gt)
    p = SE.attach_gt(pool, labels)
    console.print(f"кандидатов в пуле {len(p)}; размечено {int(p.label.notna().sum())}; метки: {p.label.value_counts().to_dict()}")
    if montages is not None:
        todo = p[p.label.isna()].sort_values(["video", "start"]).reset_index(drop=True)
        if todo.empty:
            console.print("все кандидаты пула размечены")
        else:
            out_dir = montages if montages.is_absolute() else ROOT / montages
            idx = SE.candidate_montage(todo, out_dir, per_image=6, ids=list(range(len(todo))))
            blank = idx.assign(label="", start_quality="", note="", source="pool", reviewer="")[["id", "video", "tid", "peak_t", "start", "found_by", "png", "label", "start_quality", "note", "source", "reviewer"]]
            blank.to_csv(out_dir / "gesture_gt_todo.csv", index=False, encoding="utf-8-sig")
            console.print(f"к разметке {len(todo)} кандидатов: монтажи {out_dir / 'cand_NN.png'}, шаблон {out_dir / 'gesture_gt_todo.csv'}. Заполните label "
                          f"({', '.join(SE.GESTURE_LABELS)}, unsure, no_gesture) и при возможности start_quality (ok/late/early), затем добавьте строки (video, tid, peak_t, start, label, start_quality, note, source, reviewer) в {_rel(SE.GT_CSV)}")
    gens = list(SE.GENERATORS) + ["main+lenient_k05", "main+guide_k05", "main+lenient"]
    t = SE.generator_table(p, gens, n_boot=n_boot)
    fmt = lambda v: "—" if v != v else f"{v:.2f}"   # noqa: E731
    ci = lambda r, k: f"{fmt(r[k])} [{fmt(r[k + '_lo'])}–{fmt(r[k + '_hi'])}]"   # noqa: E731
    _tbl("Генераторы кандидатов-циклов (метки «неясно» не учтены; 95% интервал по видео)", ["генератор", "кандидатов", "recall затяжек", "recall жестов", "precision затяжек", "precision жестов"],
         [[r["generator"], int(r["cand"]), ci(r, "recall_puff"), ci(r, "recall_gesture"), ci(r, "precision_puff"), ci(r, "precision_gesture")] for _, r in t.iterrows()])
    sq = SE.start_quality_table(p, "main")
    _tbl("Точность старта цикла у рабочего генератора (оценка по кадрам, допуск ±0.5 с; NA — не определить)", ["старт", "циклов", "доля"], [[r.start_quality, r.n, f"{r.share:.0%}"] for r in sq.itertuples()])
    out = OUTPUTS / "analysis"
    t.to_csv(out / "start_eval_generators.csv", index=False, encoding="utf-8-sig")
    sq.to_csv(out / "start_eval_start_quality.csv", index=False, encoding="utf-8-sig")
    if audit.exists():
        a = pd.read_csv(audit)
        k, n = int((a.label == "touch_face").sum() + (a.label == "smoke").sum()), int((a.label != "unsure").sum())
        lo, hi = SE.wilson(k, n)
        console.print(f"аудит случайных окон вне кандидатов: {len(a)} окон ({int((a.label == 'unsure').sum())} неясных), настоящих жестов пропущено {k} из {n} = {k / max(n, 1):.0%} [{lo:.0%}–{hi:.0%}] (Уилсон); "
                      f"затяжек среди пропущенных {int((a.label == 'smoke').sum())}")
    img = SE.plot_generators(t[t.generator.isin(["main", "guide", "guide_k05", "lenient", "lenient_k05", "main+lenient_k05"])], ROOT / "docs" / "img" / "start_eval.png",
                             f"{len(p)} кандидатов пула, из них размечено «жест/не жест» {int((p.label.notna() & (p.label != 'unsure')).sum())}; генераторы — пороги th_in/th_out и продление кисти k (SE.GENERATORS)")
    console.print(f"таблицы: {out / 'start_eval_generators.csv'}; график: {img}")


@app.command("pool-features", help="Признаки (кинематика, ритм, положение точек, предмет) для ВСЕХ кандидатов пула → outputs/analysis_pool/*.parquet. Нужны для `compare-sets --pool` и "
                                   "`train-bundle --pool`. Пул — outputs/analysis/gesture_pool.parquet (`sd start-eval --rebuild`). Предмет — самая долгая часть (десятки секунд на кандидата на "
                                   "этом ноутбуке): `--objects \"\"` пропускает его (наборы с признаками предмета тогда недоступны).")
def pool_features_cmd(objects: Annotated[str, typer.Option(help="детекторы предмета через запятую; пустая строка — без предмета")] = "smoking_yolo11m_beehzod,smoking_yolo26s_basant18",
                      out: Annotated[Optional[Path], typer.Option(help="куда писать (по умолчанию outputs/analysis_pool; существующие таблицы будут перезаписаны)")] = None) -> None:
    from . import pool_features as PFt
    from . import start_eval as SE

    if not SE.POOL.exists():
        console.print(f"[red]нет {_rel(SE.POOL)} — сначала `sd start-eval --rebuild` (собрать пул кандидатов по запускам)[/]")
        raise typer.Exit(1)
    import pandas as pd

    pool = pd.read_parquet(SE.POOL)
    dets = tuple(x.strip() for x in objects.split(",") if x.strip())
    target = (out if (out is None or out.is_absolute()) else ROOT / out) or PFt.POOL_AN
    console.print(f"кандидатов в пуле {len(pool)}, запусков {pool.run.nunique()}; предмет: {', '.join(dets) if dets else 'не считается'}; пишу в {target}")
    p, cb = _progress("признаки пула")
    try:
        res = PFt.build(pool, out=target, detectors=dets, progress=cb)
    finally:
        p.stop()
    console.print(f"готово: {', '.join(sorted(f.name for f in res.glob('*.parquet')))} в {res}")


@app.command("smoke-cues", help="«Дым» как признак цикла: простые признаки «дымки» (падение резкости, контраста и насыщенности ПОСЛЕ затяжки) и CLIP zero-shot «виден дым» по кропам рта; AUC с интервалом по видео, "
                                "проверка на смешение со сценой и вклад в набор `cheap`. Кропы — после `sd analyze-cycles --what photo`; пишет outputs/analysis/smoke_cycles.parquet и smoke_cues.csv.")
def smoke_cues_cmd(labels: Annotated[Path, typer.Option(help="CSV меток циклов")] = ROOT / "docs" / "review" / "assistant_visual_labels.csv",
                   clip: Annotated[bool, typer.Option("--clip/--no-clip", help="CLIP zero-shot «виден дым» (один раз загружает текстовую башню через torch)")] = True,
                   repeats: int = 3, n_boot: int = 400) -> None:
    from . import cycle_models as CM
    from . import feature_auc as FA
    from . import smoke_cues as SC

    cyc = FA._cycles_with_durations(FA.AN)
    p, cb = _progress("признаки дыма")
    try:
        cues = SC.cycle_cues(cyc, with_clip=clip, progress=cb)
    finally:
        p.stop()
    if cues.empty:
        console.print("[red]нет кропов рта: сначала `sd analyze-cycles --what photo --photo-bundle models/photo/photo_v2`[/]")
        raise typer.Exit(1)
    out = OUTPUTS / "analysis"
    cues.to_parquet(out / "smoke_cycles.parquet", index=False)
    tab = FA.feature_table(labels, with_dataset=True)
    tab = tab[tab.start.notna()]
    tab = tab.assign(start=tab.start.round(3)).merge(cues, on=SC.KEY, how="left").reset_index(drop=True)
    res = SC.evaluate(tab, n_boot=n_boot)
    fmt = lambda v: "—" if v != v else f"{v:.2f}"   # noqa: E731
    _tbl(f"Признаки «дыма»: {len(tab)} размеченных циклов (курение {int(tab.y.sum())}), метки — оценка ассистента; направления заданы заранее (0.5 = нет сигнала)",
         ["признак", "AUC [95% по видео]", "внутри папки «курение»", "папка среди негативов (0.5 = не видит)"],
         [[r.признак, f"{fmt(r.AUC)} [{fmt(r.lo)}–{fmt(r.hi)}]", fmt(r.внутри_папки_курение), fmt(r.папка_среди_негативов)] for r in res.itertuples()])
    res.to_csv(out / "smoke_cues.csv", index=False, encoding="utf-8-sig")
    cols = [c for c in SC.ALL_COLS if c in tab.columns and tab[c].notna().any()]
    cmp = CM.compare(tab, {"cheap": CM.SETS["cheap"], "cheap + дым": CM.SETS["cheap"] + cols}, repeats=repeats)
    acols = [c for c in cmp.columns if c.startswith("AUC")]
    _tbl("Вклад в классификатор цикла (out-of-fold, фолды по видео)", ["набор", "признаков"] + [c.replace("AUC ", "") for c in acols] + ["95% по видео (ансамбль)"],
         [[r["набор"], r["признаков"]] + [fmt(r[c]) for c in acols] + [f"{fmt(r['ансамбль lo'])}–{fmt(r['ансамбль hi'])}"] for _, r in cmp.iterrows()])
    cmp.to_csv(out / "smoke_cues_compare.csv", index=False, encoding="utf-8-sig")
    console.print(f"таблицы: {out / 'smoke_cues.csv'}, {out / 'smoke_cues_compare.csv'}; признаки по циклам: {out / 'smoke_cycles.parquet'}")


@app.command("enrich", help="Дорогие признаки цикла (предмет на кропах, VLM) по готовым запускам `sd recognize` → outputs/enrich/<клип>/; кэш по циклам, прерванный прогон продолжается. "
                            "Нужны для классификатора `cheap`/`full` и для `sd oof-eval --enriched`.")
def enrich_cmd(objects: Annotated[str, typer.Option(help="детекторы предмета через запятую; пусто — не считать")] = "smoking_yolo11m_beehzod,smoking_yolo26s_basant18",
               vlm: Annotated[str, typer.Option(help="тег Ollama (например qwen3.5:2b-q4_K_M); пусто — без VLM; сервер: sd.cmd ollama up --igpu")] = "",
               clips: Annotated[Optional[str], typer.Option(help="только эти клипы (video_id через запятую)")] = None,
               photo_bundle: Annotated[Optional[Path], typer.Option(help="пакет фото-модели (models/photo/<имя>): признаки photo_* и zero-shot CLIP по кропам рта")] = None) -> None:
    from . import enrich as EN
    from . import oof_eval as O

    cl = O.collect(videos=[c.strip() for c in clips.split(",")] if clips else None)
    if not cl:
        console.print("[red]нет готовых запусков `sd recognize` (outputs/recognize/<клип>/0-end*/analysis)[/]")
        raise typer.Exit(1)
    dets = [x.strip() for x in objects.split(",") if x.strip()]
    console.print(f"клипов {len(cl)}; предмет: {', '.join(dets) or 'нет'}; VLM: {vlm or 'нет'}")
    t0 = time.perf_counter()
    res = EN.enrich_clips(cl, dets, vlm or None, photo_bundle=photo_bundle, progress=lambda tag, i, n, name: console.print(f"  [{i}/{n}] {tag} {name} ({time.perf_counter() - t0:.0f} с)") if tag == "клип" else None)
    console.print(res.to_string(index=False))
    console.print(f"каталог: {EN.ROOT}")


@app.command("gt-from-gestures", help="Эталон событий (`labels/events_gt.csv`) из меток жестов `docs/review/gesture_gt.csv`: затяжки одного человека → эпизоды (POSITIVE), одиночные и неясные → IGNORE, "
                                      "папка без курения → «весь клип без курения». Строки разметчика `assistant` по этим клипам заменяются, ваши — нет. Нужны готовые запуски (`sd recognize`/`sd eval`).")
def gt_from_gestures_cmd(labeler: Annotated[str, typer.Option(help="имя разметчика в файле эталона")] = "assistant") -> None:
    import pandas as pd

    from . import gt as GT
    from . import gt_pool as GP
    from . import oof_eval as O
    from . import start_eval as SE
    from .paths import list_videos, video_id
    from .tracks import Tracks
    from .video_io import probe

    clips = O.collect()
    if not clips:
        console.print("[red]нет готовых запусков `sd recognize` по полному видео[/]")
        raise typer.Exit(1)
    runs = [(n, c.run_dir) for n, c in clips.items()]
    pool = SE.build_pool(SE.generator_cycles(runs))
    pool = O.match_gestures(pool, pd.read_csv(SE.GT_CSV, encoding="utf-8-sig"))
    vids = {video_id(v): v for v in list_videos()}
    dur = {k: float(probe(vids[k]).duration) for k in clips}
    cache: dict = {}

    def box_at(v, tid, t):
        if v not in cache:
            cache[v] = Tracks.load(clips[v].run_dir / "pose")
        return cache[v].box_at(int(tid), t)

    r = GP.rebuild_gt(pool, dur, box_at, labeler=labeler)
    df = GT.load()
    console.print(r)
    console.print(f"эталон: {GT.default_path()} · строк {len(df)}, клипов {df.clip_id.nunique()}, замечания проверки: {GT.validate(df) or 'нет'}")


@app.command("oof-eval", help="Честная оценка классификатора цикла и событий: оценка каждого цикла моделью, не видевшей его видео (фолды по видео), AUC по классам жестов, F1 события по эталону "
                              "(лучший на всех клипах — оптимистичен — и с вложенным подбором порогов). Данные: готовые запуски `sd recognize` по полным видео + метки жестов + `sd enrich`. "
                              "Пишет outputs/analysis/oof_study.csv.")
def oof_eval_cmd(sets: Annotated[str, typer.Option(help="наборы признаков через запятую (см. cycle_models.SETS)")] = "fast,obj_hold,obj_hold_zsd,obj_hold_vlm",
                 kinds: Annotated[str, typer.Option(help="члены ансамбля: lr, gb, nn")] = "lr,gb", repeats: int = 3, nested_seeds: int = 3,
                 by_scene: Annotated[bool, typer.Option("--by-scene", help="фолды по СЦЕНАМ (клипы одной камеры/человека вместе: noyabrsk, Jar, 2025_11_06_*, распитие), а не по клипам — строже")] = False,
                 tag: Annotated[str, typer.Option(help="суффикс файла результата oof_study<tag>.csv")] = "") -> None:
    import numpy as np
    import pandas as pd

    from . import enrich as EN
    from . import gt as GT
    from . import oof_eval as O
    from . import start_eval as SE
    from .paths import list_videos, video_id
    from .video_io import probe

    clips = O.collect()
    if not clips:
        console.print("[red]нет готовых запусков `sd recognize` по полным видео[/]")
        raise typer.Exit(1)
    tab = EN.attach(O.attach_labels(clips, pd.read_csv(SE.GT_CSV, encoding="utf-8-sig")), list(clips))
    vids = {video_id(v): v for v in list_videos()}
    dur = {k: float(probe(vids[k]).duration) for k in clips}
    names = [x.strip() for x in sets.split(",") if x.strip()]
    from . import cycle_models as CM

    bad = [n for n in names if n not in CM.SETS]
    if bad:
        console.print(f"[red]неизвестные наборы {bad}; доступны: {', '.join(CM.SETS)}[/]")
        raise typer.Exit(1)
    console.print(f"клипов {len(clips)}, циклов {len(tab)}, размечено {int(tab.y.notna().sum())} (затяжек {int(np.nansum(tab.y))}); эталон событий: {len(GT.load())} строк")
    res, _ = O.study(clips, tab, dur, names, tuple(k.strip() for k in kinds.split(",")), repeats, nested_seeds=nested_seeds, by_scene=by_scene, progress=lambda i, n, nm: console.print(f"  [{i}/{n}] {nm}"))
    f = lambda v: "—" if v != v else f"{v:.2f}"   # noqa: E731
    _tbl("Классификатор цикла (oof) и события по эталону; «вложенный» — честная оценка подбора порогов (мин–макс по разбиениям)", ["набор", "признаков", "AUC [95% по видео]", "F1 лучший*", "порог цикла / события", "P / R", "F1 вложенный"],
         [[r["набор"], r["признаков"], f"{f(r['auc'])} [{f(r['auc_lo'])}–{f(r['auc_hi'])}]", f(r["f1_лучший"]), f"{r['порог_цикла']:.2f} / {r['порог_события']:.2f}", f"{f(r['P'])} / {f(r['R'])}",
           f"{f(r['f1_вложенный_медиана'])} ({f(r['f1_вложенный_мин'])}–{f(r['f1_вложенный_макс'])})"] for _, r in res.iterrows()])
    console.print("* пороги подобраны на тех же клипах — оптимистично.")
    out = OUTPUTS / "analysis" / f"oof_study{tag or ('_scene' if by_scene else '')}.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    console.print(f"таблица: {out}")


@app.command("ladder", help="«Лестница вычислений»: что даёт более тяжёлая поза (модель, размер входа, уточнение точек) на клипах с метками затяжек — recall затяжек циклами автомата, "
                            "лишние циклы, мс на кадр, достоверность запястий. Оценка выигрыша от более мощного ПК; поза каждой конфигурации кэшируется. Пишет outputs/analysis/ladder.csv.")
def ladder_cmd(configs: Annotated[str, typer.Option(help="через запятую: n960, n1280, n1600, n1920, s960, s1920, m1280, x1280, x1280_rtmm")] = "n960,m1280",
               clips: Annotated[str, typer.Option(help="video_id[:начало-конец][@масштаб] через запятую; по умолчанию клипы с затяжками до 66 с (@масштаб — уменьшенная копия именно этого клипа)")] = "курение__1:0-66,курение__sm_6,курение__4,курение__sm_2",
               scale: Annotated[float, typer.Option(help="во сколько раз уменьшить содержимое кадра при прежнем размере кадра (серые поля): имитация дальней камеры, 80–200 px скрытого набора")] = 1.0,
               tag: Annotated[str, typer.Option(help="суффикс файла результата ladder<tag>.csv")] = "") -> None:
    import pandas as pd

    from . import ladder as LD
    from . import start_eval as SE

    names = [c.strip() for c in configs.split(",") if c.strip()]
    bad = [n for n in names if n not in LD.CONFIGS]
    if bad:
        console.print(f"[red]неизвестные конфигурации {bad}; доступны: {', '.join(LD.CONFIGS)}[/]")
        raise typer.Exit(1)
    spec = []
    for c in clips.split(","):
        c, _, sc = c.strip().partition("@")                          # video_id[:начало-конец][@масштаб]
        v, _, w = c.partition(":")
        a, _, b = w.partition("-")
        spec.append((v, float(a) if a else 0.0, float(b) if b else None, float(sc) if sc else None))
    gest = pd.read_csv(SE.GT_CSV, encoding="utf-8-sig")
    t0 = time.perf_counter()
    df = LD.run(spec, gest, names, progress=lambda i, n, nm: console.print(f"  [{i}/{n}] {nm} ({time.perf_counter() - t0:.0f} с)"), scale=scale)
    out = OUTPUTS / "analysis" / f"ladder{tag or ('_x%03d' % round(scale * 100) if scale != 1.0 else '')}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    sm = LD.summarize(df)
    f = lambda v: "—" if v != v else f"{v:.2f}"   # noqa: E731
    _tbl("Поза: затяжки, найденные циклами автомата (метки — мои, не эталон)", ["конфигурация", "клипов", "затяжек", "найдено", "recall", "лишних циклов", "достоверность запястий", "мс/кадр", "ошибок"],
         [[r.config, r.clips, r.smoke, r.found, f(r.recall), r.extra, f(r.wrist_conf), f"{r.ms_per_frame:.0f}", r.errors] for r in sm.itertuples()])
    if (df.error != "").any():
        console.print("[yellow]ошибки: " + "; ".join(sorted(set(f"{r.config} {r.clip}: {r.error}" for r in df[df.error != ''].itertuples())))[:600] + "[/]")
    console.print(f"таблица: {out}")


def _need_pool_tables() -> None:
    from . import pool_features as PFt

    if not (PFt.POOL_AN / "dataset_cycles.parquet").exists():
        console.print(f"[red]нет {_rel(PFt.POOL_AN)}/dataset_cycles.parquet — сначала `sd pool-features`[/]")
        raise typer.Exit(1)


DEFAULT_VLM_PARQUET = OUTPUTS / "analysis" / "vlm_qwen3.5_2b-q4_K_M_f6_s224_mouth_full.parquet"


@app.command("compare-sets", help="Сравнение наборов признаков и ансамблей для классификатора цикла (логрегрессия / LightGBM / MLP и их среднее), фолды по видео, повторные перемешивания. Пишет outputs/analysis/compare_sets.csv.")
def compare_sets_cmd(labels: Annotated[Path, typer.Option(help="CSV меток циклов")] = ROOT / "docs" / "review" / "assistant_visual_labels.csv",
                     vlm: Annotated[Optional[Path], typer.Option(help="parquet из `sd vlm-eval` (VLM-признак); по умолчанию — полный прогон 2B, если он есть")] = None,
                     repeats: int = 5, sets: Annotated[Optional[str], typer.Option(help="наборы через запятую (по умолчанию все)")] = None,
                     tag: Annotated[Optional[str], typer.Option(help="суффикс имени файла")] = None,
                     pool: Annotated[bool, typer.Option("--pool", help="таблица ВСЕХ кандидатов пула (160 жестов от нескольких генераторов, метки docs/review/gesture_gt.csv; «неясно» исключены): `sd pool-features` должен быть выполнен")] = False) -> None:
    import pandas as pd

    from . import cycle_models as CM
    from . import feature_auc as FA

    vlm = vlm or (DEFAULT_VLM_PARQUET if DEFAULT_VLM_PARQUET.exists() else None)
    if pool:
        from . import pool_features as PFt
        from . import start_eval as SE

        _need_pool_tables()
        tab = PFt.labelled_table(pd.read_csv(SE.GT_CSV), pd.read_parquet(SE.POOL), vlm=None)
        tag = tag or "pool"
    else:
        tab = FA.feature_table(labels, vlm=vlm, with_dataset=True)
        tab = tab[tab.start.notna()]
    tab = tab.reset_index(drop=True)
    console.print(f"размеченных циклов: {len(tab)} (курение {int(tab.y.sum())}), видео {tab.video.nunique()}; VLM: {vlm.name if vlm else 'нет'}")
    chosen = {k: CM.SETS[k] for k in sets.split(",")} if sets else CM.SETS
    res = CM.compare(tab, chosen, repeats=repeats)
    fmt = lambda v: "—" if v != v else f"{v:.2f}"   # noqa: E731
    cols = [c for c in res.columns if c.startswith("AUC")]
    _tbl("AUC out-of-fold (фолды по видео); метки — оценка ассистента, не эталон", ["набор", "признаков"] + [c.replace("AUC ", "") for c in cols] + ["95% по видео (ансамбль)"],
         [[r["набор"], r["признаков"]] + [fmt(r[c]) for c in cols] + [f"{fmt(r['ансамбль lo'])}–{fmt(r['ансамбль hi'])}"] for _, r in res.iterrows()])
    out = OUTPUTS / "analysis" / f"compare_sets{('_' + tag) if tag else ''}.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    console.print(f"таблица: {out}")


@app.command("train-bundle", help="Обучить пакет классификатора цикла БЕЗ pickle (ансамбль: логрегрессии + LightGBM + MLP; калибровка по out-of-fold) в models/cycle/<имя>/ с manifest.json.")
def train_bundle_cmd(name: Annotated[str, typer.Argument(help="имя пакета: models/cycle/<имя>/")] = "cycle_v1",
                     feature_set: Annotated[str, typer.Option("--set", help="набор признаков: kin | kin+rhythm | kin+pose | kin+obj | cheap | cheap+photo | cheap+video | cheap+vlm | full")] = "cheap",
                     labels: Annotated[Path, typer.Option(help="CSV меток циклов")] = ROOT / "docs" / "review" / "assistant_visual_labels.csv",
                     vlm: Annotated[Optional[Path], typer.Option(help="parquet VLM (обязателен для наборов с vlm)")] = None, repeats: int = 5,
                     pool: Annotated[bool, typer.Option("--pool", help="обучать на ВСЕХ кандидатах пула (≈ 130 с метками, включая «нет жеста»): для широкого генератора кандидатов; VLM/видео-признаков в пуле нет")] = False,
                     enriched: Annotated[bool, typer.Option("--enriched", help="циклы ПОЛНЫХ запусков `sd recognize` + метки жестов + признаки `sd enrich` (предмет, фото/zero-shot, VLM): набор `obj_hold_zsd` и др.")] = False,
                     calibrate: Annotated[bool, typer.Option("--calibrate/--no-calibrate", help="изотоническая калибровка оценки; без неё оценка = среднее членов (пороги по `sd oof-eval` переносятся как есть)")] = True,
                     kinds: Annotated[str, typer.Option(help="члены ансамбля через запятую: lr (логрегрессия), gb (LightGBM), nn (MLP)")] = "lr,gb,nn") -> None:
    import pandas as pd

    from . import cycle_models as CM
    from . import feature_auc as FA

    if feature_set not in CM.SETS:
        console.print(f"[red]набор «{feature_set}» не найден; доступны: {', '.join(CM.SETS)}[/]")
        raise typer.Exit(1)
    vlm = vlm or (DEFAULT_VLM_PARQUET if DEFAULT_VLM_PARQUET.exists() else None)
    if enriched:
        from . import enrich as EN
        from . import oof_eval as O
        from . import start_eval as SE

        clips = O.collect()
        tab = EN.attach(O.attach_labels(clips, pd.read_csv(SE.GT_CSV, encoding="utf-8-sig")), list(clips))
        tab = tab[tab.y.notna()].reset_index(drop=True)
        labels = SE.GT_CSV
    elif pool:
        from . import pool_features as PFt
        from . import start_eval as SE

        _need_pool_tables()
        tab = PFt.labelled_table(pd.read_csv(SE.GT_CSV), pd.read_parquet(SE.POOL), vlm=None)
        labels = SE.GT_CSV
    else:
        tab = FA.feature_table(labels, vlm=vlm, with_dataset=True)
        tab = tab[tab.start.notna()]
    tab = tab.reset_index(drop=True)
    meta = dict(labels=str(labels.relative_to(ROOT)) if labels.is_relative_to(ROOT) else str(labels), label_source="assistant_visual (суждения по кадрам, не эталон)" if ("assistant" in labels.name or "gesture_gt" in labels.name) else "ручные",
                candidates="циклы рабочего автомата полных запусков + метки жестов пула" if enriched else ("пул (несколько генераторов, включая «нет жеста»)" if pool else "рабочий автомат (th_in 0.65 / th_out 0.90)"), vlm=vlm.name if (vlm and "vlm" in feature_set) else None, note="пакет собран для проверки механики и сравнения наборов; метрики — OOF по видео на малой выборке, не итоговое качество")
    man = CM.train(tab, feature_set, ROOT / "models" / "cycle" / name, meta, repeats, calibrate=calibrate, kinds=tuple(k.strip() for k in kinds.split(",") if k.strip()))
    console.print(f"пакет: {ROOT / 'models' / 'cycle' / name}; циклов {man['n']} (курение {man['n_pos']}), видео {man['n_groups']}, признаков {len(man['features'])}")
    console.print(f"AUC out-of-fold: члены {man['cv']['auc_members']}, ансамбль {man['cv']['auc_ensemble']}, после калибровки {man['cv']['auc_ensemble_calibrated']}")


@app.command("review-all", help="Монтажи кадров циклов по ВСЕМ готовым запускам с текущими порогами (для разметки глазами); outputs/analysis/review/.")
def review_all(per_video: Annotated[int, typer.Option(help="сколько циклов показать на видео (равномерно по времени)")] = 8,
               set_: SetOpt = None) -> None:
    from . import review_batch as RB

    p, cb = _progress("монтажи")
    try:
        df = RB.make_all(per_video=per_video, progress=cb)
    finally:
        p.stop()
    console.print(df.to_string(index=False))
    console.print(f"каталог: {OUTPUTS / 'analysis' / 'review'}")


@app.command("eval", help="Оценка на выбранных папках: события → метрики с опорой на целевую F1 (интервал, бюджет ошибок, порог, причины ошибок); запись в журнал outputs/experiments/. "
                          "Эталон — labels/events_gt.csv (страница «Разметка») или имя папки (--mode clips).")
def eval_cmd(dirs: Annotated[list[Path], typer.Option("--dir", "-d", help="папка с видео (можно несколько раз)")],
             profile: Annotated[list[str], typer.Option("--profile", "-p", help="профиль из configs/experiments (можно несколько — сравнение)")] = None,
             mode: Annotated[str, typer.Option(help="events — по ручному эталону | clips — по имени папки")] = "events",
             role: Annotated[str, typer.Option(help="validation | hidden (порог из профиля, подбор запрещён)")] = "validation",
             policy: Annotated[str, typer.Option(help="plateau | best | fixed — как выбрать порог (только validation)")] = "plateau",
             target_f1: Annotated[float, typer.Option(help="целевая Event F1")] = 0.80,
             max_sec: Annotated[Optional[float], typer.Option(help="только первые N секунд каждого клипа")] = None,
             roi: Annotated[Optional[Path], typer.Option(help="roi.json")] = None,
             solver: Annotated[Optional[Path], typer.Option(help="оценить установленный/упакованный решатель (*.sdsolver.zip или каталог) вместо профиля")] = None,
             classifier: Annotated[Optional[str], typer.Option(help="классификатор цикла (models/cycle/<имя>); подменяет выбранный в профиле. ОБЯЗАТЕЛЕН: без него оценка не запускается")] = None,
             no_cache: Annotated[bool, typer.Option("--no-cache")] = False) -> None:
    from . import profiles as PR
    from . import runner as RN
    from . import solver as SV

    profs = [SV.resolve_profile(solver)] if solver else ([PR.load(p) for p in profile] if profile else [PR.default_profile()])
    rows = []
    for p in profs:
        try:
            out = RN.evaluate_dirs([str(d) for d in dirs], p, classifier=classifier, mode=mode, role=role, policy="fixed" if role == "hidden" else policy, target_f1=target_f1,
                                   use_cache=not no_cache, max_sec=max_sec, roi_path=str(roi) if roi else None, progress=lambda i, n, m: console.print(f"  [{i}/{n}] {m}"))
        except ValueError as e:
            console.print(f"[red]{p.name}: {e}[/]")
            raise typer.Exit(1)
        m, b, ci = out.report.metrics, out.report.budget, out.report.ci.get("f1") or (None, None)
        rows.append([p.name, f"{m['f1']:.3f}", f"{ci[0]:.2f}–{ci[1]:.2f}" if ci[0] is not None else "—", f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['tp']}/{m['fp']}/{m['fn']}",
                     "—" if m["fp_per_hour"] is None else f"{m['fp_per_hour']:.1f}", f"{m['threshold']:.2f}", f"{int(b['errors'])}/{int(b['allowed'])}", "да" if b["reached"] else "нет"])
        for n in out.report.notes:
            console.print(f"[yellow]! {n}[/]")
        console.print(f"журнал: outputs/experiments/{out.run_id}")
    _tbl(f"Event F1 (цель {target_f1:.2f})", ["профиль", "F1", "95%", "P", "R", "TP/FP/FN", "FP/ч", "порог", "ошибок/допустимо", "цель"], rows)


@app.command("gt", help="Эталон событий (labels/events_gt.csv): сколько клипов размечено, предупреждения о разметке, экспорт в формате организаторов (--export файл.csv).")
def gt_cmd(dirs: Annotated[Optional[list[Path]], typer.Option("--dir", "-d", help="показать только клипы этих папок")] = None,
           export: Annotated[Optional[Path], typer.Option(help="записать эталон в формате организаторов")] = None) -> None:
    from . import gt as GT
    from . import library as LIB

    df = GT.load()
    ids = set(LIB.clip_ids(LIB.videos_in(dirs))) if dirs else None
    sub = df if ids is None else df[df.clip_id.isin(ids)]
    console.print(f"{GT.default_path()}: строк {len(df)}, в выборке {len(sub)}; POSITIVE {int((sub.label == 'POSITIVE').sum())}, NEGATIVE {int((sub.label == 'NEGATIVE').sum())}, IGNORE {int((sub.label == 'IGNORE').sum())}")
    if ids is not None:
        st = GT.clip_status(df, sorted(ids))
        console.print(f"клипов {len(st)}, размечено {int(st.reviewed.sum())}")
    for w in GT.validate(sub):
        console.print(f"[yellow]! {w}[/]")
    if export:
        GT.export_organizer(df, ids).to_csv(export, index=False)
        console.print(f"экспорт: {export}")


@app.command(help="Запустить веб-интерфейс (Streamlit): Оценка, Просмотр, Разметка, Модели, Этапы.")
def ui(port: int = 8501, headless: bool = True,
       host: Annotated[str, typer.Option(help="адрес привязки; по умолчанию только эта машина (обработка локальная). В контейнере: 0.0.0.0 + публикация порта на 127.0.0.1 хоста")] = "127.0.0.1") -> None:
    import subprocess

    app_py = Path(__file__).parent / "ui" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_py), "--server.port", str(port), "--server.address", host, "--server.headless", str(headless).lower(),
           "--server.fileWatcherType", "none", "--server.enableStaticServing", "true", "--browser.gatherUsageStats", "false"]
    console.print("запуск:", " ".join(cmd))
    raise typer.Exit(subprocess.call(cmd, cwd=str(ROOT)))


@app.command("monitor", help="Мониторинг (real-time контур): камеры = папки <район>-<индекс>-<время начала> в --streams; тревоги пишутся в outputs/monitor/monitor.db, смотреть — `sd app`. "
                            "--watch следит за папками и подхватывает новые фрагменты; --speed 1 — в реальном времени, 0 — как можно быстрее.")
def monitor_cmd(streams: Annotated[Optional[Path], typer.Option(help="каталог с папками-камерами (по умолчанию SD_STREAMS или ./streams)")] = None,
                profile: Annotated[Optional[str], typer.Option("--profile", "-p", help="профиль моделей (configs/experiments); по умолчанию default")] = None,
                camera: Annotated[Optional[list[str]], typer.Option(help="только эти камеры (район-индекс), можно несколько раз")] = None,
                watch: Annotated[bool, typer.Option("--watch/--once")] = False, speed: Annotated[float, typer.Option(help="скорость воспроизведения: 1 — как в жизни, 0 — без пауз")] = 0.0,
                interval: Annotated[float, typer.Option(help="период опроса папок в режиме --watch, с")] = 5.0,
                solver: Annotated[Optional[Path], typer.Option(help="решатель (*.sdsolver.zip или установленный) вместо профиля")] = None,
                classifier: Annotated[Optional[str], typer.Option(help="классификатор цикла (models/cycle/<имя>); обязателен, подменяет выбранный в профиле")] = None) -> None:
    from . import profiles as PR
    from . import runner as RN
    from . import solver as SV
    from .paths import STREAMS
    from .realtime.worker import run_monitor

    prof = RN.with_classifier(SV.resolve_profile(solver) if solver else (PR.load(profile) if profile else PR.default_profile()), classifier)
    errs = SV.errors(prof, "live", devices=True)
    if errs:
        for e in errs:
            console.print(f"[red]{e.text}[/]")
        raise typer.Exit(1)
    root = streams or STREAMS
    console.print(f"камеры: {root} · профиль {prof.name} · {prof.describe()}")
    if not (prof.opts().get("cycle_bundle")):
        console.print("[yellow]! классификатор цикла не задан (allow_heuristic): оценка — эвристика по длительности паузы, только для отладки[/]")
    try:
        tot = run_monitor(root, prof, cameras=camera, watch=watch, speed=speed, interval=interval, log=console.print)
    except KeyboardInterrupt:
        console.print("остановлено")
        return
    console.print(f"готово: фрагментов {tot['chunks']}, новых тревог {tot['alerts']}, обновлений {tot['updates']}")


@app.command("monitor-demo", help="Демонстрационные тревоги для разработки интерфейса оператора без моделей (помечены «ДЕМО»). --clear удаляет их.")
def monitor_demo_cmd(n: Annotated[int, typer.Option(help="сколько тревог создать")] = 14, clear: Annotated[bool, typer.Option("--clear")] = False) -> None:
    from .realtime.demo import seed_demo
    from .realtime.store import AlertStore

    st = AlertStore()
    if clear:
        console.print(f"удалено демо-тревог: {st.delete_demo()}")
        return
    console.print(f"создано демо-тревог: {seed_demo(st, n)}; смотреть: sd app")


@app.command("feedback-export", help="Решения оператора (подтверждено / ложная) → эталон событий labels/events_gt.csv: подтверждённые — POSITIVE, ложные — NEGATIVE (сложные негативы).")
def feedback_export_cmd() -> None:
    from .realtime.feedback import export_reviewed
    from .realtime.store import AlertStore

    console.print(export_reviewed(AlertStore()))


@app.command("app", help="Веб-приложение оператора БЕЗ Streamlit: карта с камерами, мультипросмотр, тревоги и решения, просмотр видео с выводами модели. По умолчанию http://127.0.0.1:8502")
def app_cmd(port: int = 8502, host: Annotated[str, typer.Option(help="адрес привязки; в контейнере 0.0.0.0")] = "127.0.0.1",
            open_browser: Annotated[bool, typer.Option("--open/--no-open", help="открыть браузер")] = False) -> None:
    from .web.server import serve

    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"
    console.print(f"веб-приложение: [bold]{url}[/]  (Ctrl+C — остановить)")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    serve(host=host, port=port)


@app.command("replay", help="Имитация реального времени по видео: файл идёт через потоковый движок кадр за кадром, тревоги печатаются в момент срабатывания. "
                            "С эталоном (--gt) — ещё TP/FP/FN, F1 и задержка тревоги; всегда — пропускная способность (справится ли устройство с потоком).")
def replay_cmd(video: VideoArg, profile: Annotated[Optional[str], typer.Option("--profile", "-p", help="профиль (configs/experiments)")] = None,
               solver: Annotated[Optional[Path], typer.Option(help="решатель (*.sdsolver.zip или установленный)")] = None,
               classifier: Annotated[Optional[str], typer.Option(help="классификатор цикла (обязателен; подменяет выбранный в профиле)")] = None,
               start: Start = 0.0, end: End = None, speed: Annotated[float, typer.Option(help="0 — как можно быстрее, 1 — реальное время, N — в N раз быстрее")] = 0.0,
               gt: Annotated[bool, typer.Option("--gt/--no-gt", help="сверять с эталоном labels/events_gt.csv, если для клипа он есть")] = True,
               render: Annotated[bool, typer.Option("--render/--no-render", help="видео с рамками, баннером тревоги и разметкой")] = False) -> None:
    from . import gt as GT
    from . import profiles as PR
    from . import runner as RN
    from . import solver as SV
    from .realtime import replay as RP

    prof = RN.with_classifier(SV.resolve_profile(solver) if solver else (PR.load(profile) if profile else PR.default_profile()), classifier)
    errs = SV.errors(prof, "replay", devices=True)
    if errs:
        for e in errs:
            console.print(f"[red]{e.text}[/]")
        raise typer.Exit(1)
    vid = resolve_video(video)
    console.print(f"{vid.name}: профиль {prof.name} · {prof.describe()['cycle_model']} · темп x{speed:g}" if speed else f"{vid.name}: профиль {prof.name} · {prof.describe()['cycle_model']} · без пауз")

    def on_alert(u) -> None:
        if u.kind == "open":
            console.print(f"  [red]ТРЕВОГА[/] поток {u.t_now:6.1f} с · начало события {u.start:6.1f} с · ID {u.tid} · {u.explain} · {u.confidence:.0%} (задержка {u.t_now - u.start:.1f} с)")
        elif u.kind == "update":
            console.print(f"  обновление: событие до {u.end:.1f} с · {u.explain} · {u.confidence:.0%}")

    res = RP.replay_video(vid, prof, start=start, end=end, speed=speed, gt=GT.load() if gt else None, render=render, on_alert=on_alert)
    s = res.stats
    console.print(f"кадров {s['frames']} · {s['stream_sec']} с видео за {s['busy_sec']} с вычислений · {s['fps_proc']} к/с (нужно {s['process_fps_target']:g}) · мс/кадр p50 {s['ms_p50']}, p95 {s['ms_p95']}")
    console.print(f"[bold]{s['verdict']}[/]")
    console.print(f"циклов {s['cycles']}, тревог {s['alerts']}")
    if res.metrics:
        m = res.metrics
        console.print(f"эталон: TP {m['tp']} FP {m['fp']} FN {m['fn']} · P {m['precision']:.2f} R {m['recall']:.2f} F1 {m['f1']:.2f} · задержка тревоги: медиана {m['alert_delay_median']}, максимум {m['alert_delay_max']} с")
    for n in res.notes:
        console.print(f"[yellow]! {n}[/]")
    console.print(f"результаты: {res.out_dir}" + (f" · видео {res.overlay}" if res.overlay else ""))


@app.command(help="Юнит-тесты (pytest).")
def test() -> None:
    import subprocess

    raise typer.Exit(subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=str(ROOT)))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    app()


if __name__ == "__main__":
    main()
