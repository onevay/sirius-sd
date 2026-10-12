"""Команды Dev Studio (вынесены из cli.py, чтобы не раздувать его): отчёт, классификатор по спецификации, выпуск, варианты профиля, версии разметки.

Каждая команда — тонкая обёртка над модулем (reports, classifier, release, sweep, labels_history) и печатает прогресс строками `[i/n] …`, по которым интерфейс
показывает полосу выполнения фоновой задачи (`sd.tasks.progress`).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

console = Console(width=140)
clf_app = typer.Typer(no_args_is_help=True, help="Классификатор цикла по спецификации (configs/classifiers/<имя>.yaml): диагностика переобучения и обучение пакета.")
labels_app = typer.Typer(no_args_is_help=True, help="Версии файлов разметки (labels/history): список, снимок с комментарием, откат.")

CLF_OUT = "classifier"


def _p(i: int, n: int, msg: str) -> None:
    console.print(f"  [{i}/{n}] {msg}", markup=False, highlight=False)


def register(app: typer.Typer) -> None:
    app.add_typer(clf_app, name="clf")
    app.add_typer(labels_app, name="labels")
    app.command("report", help="PDF-отчёт прогона оценки (в папке прогона outputs/experiments): конфигурация, данные, метрики, порог, ошибки, клипы, "
                               "производительность, слабые места; с --baseline — парное сравнение.")(report_cmd)
    app.command("release", help="Выпуск: ворота качества (прогон без сбоев, F1, тот же профиль, нет утечки, лицензии) → решатель + PDF + release.json "
                                "в outputs/releases/<имя>/.")(release_cmd)
    app.command("variants", help="Серия оценок «что если»: базовый профиль и варианты из JSON (строки name/param/value) на одних папках.")(variants_cmd)


def report_cmd(run: Annotated[str, typer.Argument(help="id прогона из журнала (outputs/experiments)")],
               baseline: Annotated[Optional[str], typer.Option(help="базовый прогон для парного сравнения")] = None,
               out: Annotated[Optional[Path], typer.Option(help="куда записать PDF")] = None,
               author: Annotated[str, typer.Option(help="кто собрал отчёт")] = "") -> None:
    from . import reports as R

    _p(1, 2, f"отчёт по прогону {run}")
    p = R.build(run, out=out, baseline=baseline or None, author=author)
    _p(2, 2, "готово")
    console.print(f"PDF: {p}")


def _table(run: str | None):
    from . import classifier as C

    tab = C.load_table(True, run=run or None)
    console.print(f"таблица циклов: {len(tab)} (размечено {int(tab.y.notna().sum())}, затяжек {int((tab.y == 1).sum())}); источник: {tab.attrs.get('source')}")
    return tab


@clf_app.command("diagnose", help="Диагностика спецификации: oof против train, перестановка меток, кривая обучения, важность. Пишет outputs/classifier/<время>_<спецификация>/.")
def clf_diagnose(spec: Annotated[str, typer.Option(help="имя спецификации (configs/classifiers/<имя>.yaml)")],
                 run: Annotated[str, typer.Option(help="таблица циклов из прогона оценки (id); пусто — последние запуски")] = "",
                 perm: Annotated[int, typer.Option(help="перестановок меток")] = 3,
                 importance: Annotated[bool, typer.Option("--importance/--no-importance", help="важность признаков (долго)")] = True) -> None:
    from . import classifier as C
    from .paths import OUTPUTS

    sp = C.load_spec(spec)
    tab = _table(run)
    lab = tab[tab.y.notna()].reset_index(drop=True)
    res = C.diagnose(lab, sp, n_perm=perm, importance=importance, progress=lambda i, n, m: _p(i, n, m))
    d = OUTPUTS / CLF_OUT / f"{time.strftime('%Y%m%d_%H%M%S')}_{spec}"
    d.mkdir(parents=True, exist_ok=True)
    res["members"].to_csv(d / "members.csv", index=False, encoding="utf-8-sig")
    res["curve"].to_csv(d / "curve.csv", index=False, encoding="utf-8-sig")
    res["importance"].to_csv(d / "importance.csv", index=False, encoding="utf-8-sig")
    import pandas as pd

    pd.DataFrame(dict(video=res["table"].video, y=res["y"], oof=res["oof"])).to_csv(d / "oof.csv", index=False, encoding="utf-8-sig")
    warn = C.warnings_for(lab, sp, res)
    (d / "diag.json").write_text(json.dumps(dict(spec=spec, spec_full=sp, run=run or None, source=tab.attrs.get("source"), n=res["n"], n_pos=res["n_pos"], n_groups=res["n_groups"],
                                                 perm_auc=res["perm_auc"], perm_mean=res["perm_mean"], verdict=res["verdict"], warnings=warn,
                                                 created=time.strftime("%Y-%m-%d %H:%M:%S")), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    console.print(res["members"].to_string(index=False))
    for v in res["verdict"]:
        console.print(f"• {v}")
    console.print(f"результат: {d}")


@clf_app.command("train", help="Обучить пакет models/cycle/<имя> по спецификации; в манифест — спецификация, источник таблицы и видео обучения.")
def clf_train(spec: Annotated[str, typer.Option(help="имя спецификации")], bundle: Annotated[str, typer.Option(help="имя пакета: models/cycle/<имя>")],
              run: Annotated[str, typer.Option(help="таблица циклов из прогона оценки (id); пусто — последние запуски")] = "",
              overwrite: Annotated[bool, typer.Option("--overwrite", help="перезаписать существующий пакет")] = False,
              attach: Annotated[str, typer.Option(help="подключить пакет к профилю (имя); пусто — не подключать")] = "") -> None:
    from . import classifier as C
    from . import profiles as PR
    from .paths import CYCLE_MODELS

    name = C._check_name(bundle)
    out = CYCLE_MODELS / name
    if out.exists() and not overwrite:
        console.print(f"[red]пакет {out} уже есть (флаг --overwrite — перезаписать; профили, ссылающиеся на него, получат другие оценки)[/]")
        raise typer.Exit(1)
    sp = C.load_spec(spec)
    _p(1, 3, "таблица циклов")
    tab = _table(run)
    bad = [t for lvl, t in C.warnings_for(tab, sp) if lvl == "error"]
    if bad:
        console.print("[red]обучение остановлено:[/]\n  - " + "\n  - ".join(bad))
        raise typer.Exit(1)
    _p(2, 3, "обучение и oof-оценка")
    man = C.train(tab, sp, out, meta=dict(source="dev-studio", spec_name=spec))
    console.print(f"пакет {out}: AUC oof ансамбля {man['cv']['auc_ensemble']}, циклов {man['n']}, затяжек {man['n_pos']}, видео {man['n_groups']}")
    if attach:
        p = PR.load_named(attach)
        PR.save(PR.Profile(name=p.name, description=p.description, base=p.base, config=p.config, options={**p.options, "cycle_bundle": f"models/cycle/{name}"}))
        console.print(f"профиль «{attach}» теперь использует models/cycle/{name} (отпечаток изменился: оцените профиль заново)")
    _p(3, 3, "готово")


def release_cmd(profile: Annotated[str, typer.Option(help="профиль (configs/experiments/<имя>)")], run: Annotated[str, typer.Option(help="прогон оценки, подтверждающий качество")],
                name: Annotated[str, typer.Option(help="имя выпуска (пусто — профиль_дата)")] = "",
                min_f1: Annotated[float, typer.Option(help="минимальная Event F1")] = 0.7,
                with_weights: Annotated[bool, typer.Option("--with-weights", help="положить веса моделей в архив")] = False,
                force: Annotated[bool, typer.Option("--force", help="выпустить при непройденных воротах (записывается в манифест)")] = False) -> None:
    from . import release as RL

    for g in RL.gates(profile, run, min_f1):
        mark = {True: "[green]✓[/]", False: "[red]✗[/]", None: "[yellow]![/]"}[g["ok"]]
        console.print(f"{mark} {g['title']}: {g['detail']}")
    try:
        rec = RL.make(profile, run, name or None, min_f1, with_weights, force, progress=_p)
    except (ValueError, FileExistsError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    console.print(f"выпуск {rec['name']}: {RL.RELEASES / rec['name']}" + (" [yellow](ворота не пройдены, выпущен принудительно)[/]" if rec["forced"] else ""))


def variants_cmd(dirs: Annotated[list[Path], typer.Option("--dir", "-d", help="папка с видео (можно несколько раз)")],
                 base: Annotated[str, typer.Option(help="базовый профиль")], rows: Annotated[Path, typer.Option(help="JSON: список {name, param, value}")],
                 policy: Annotated[str, typer.Option(help="fixed | plateau")] = "fixed") -> None:
    from . import profiles as PR
    from . import sweep as SW

    vs = SW.variants_from_rows(PR.load_named(base), json.loads(Path(rows).read_text(encoding="utf-8")))
    console.print(f"вариантов {len(vs)}: {', '.join(v.name for v in vs)}")
    for k, v in enumerate(vs, 1):
        _p(k, len(vs), f"вариант {v.name}")
        try:
            o = SW.run_variants([str(d) for d in dirs], [v], policy=policy, progress=lambda i, n, m: console.print(f"    {i}/{n} {m}", markup=False))[0]
            console.print(f"    F1 {o.report.metrics['f1']:.3f} → outputs/experiments/{o.run_id}")
        except Exception as e:      # noqa: BLE001 — один вариант не останавливает серию
            console.print(f"[red]    {v.name}: {e}[/]")


@labels_app.command("list", help="Версии файла разметки (events | gestures | cycles).")
def labels_list(kind: Annotated[str, typer.Argument(help="events | gestures | cycles")] = "events") -> None:
    from . import labels_history as LH

    v = LH.versions(kind)
    console.print(v.to_string(index=False) if len(v) else "версий нет")


@labels_app.command("snapshot", help="Зафиксировать текущую версию файла разметки с комментарием.")
def labels_snapshot(kind: Annotated[str, typer.Argument(help="events | gestures | cycles")], note: Annotated[str, typer.Option(help="зачем")] = "",
                    author: Annotated[str, typer.Option(help="кто")] = "") -> None:
    from . import labels_history as LH

    r = LH.snapshot(kind, author, note, "manual")
    console.print(r or "содержимое не менялось с прошлой версии — снимок не нужен")


@labels_app.command("restore", help="Вернуть файл разметки к версии (текущее содержимое сначала сохраняется версией).")
def labels_restore(kind: Annotated[str, typer.Argument(help="events | gestures | cycles")], snapshot: Annotated[str, typer.Argument(help="путь снимка из `sd labels list`")],
                   author: Annotated[str, typer.Option(help="кто")] = "") -> None:
    from . import labels_history as LH

    console.print(LH.restore(kind, snapshot, author))
