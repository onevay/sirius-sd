"""Команды `sd solver ...`: решатель (полный пайплайн) как переносимый пакет — упаковка, просмотр метаданных, установка, проверка совместимости."""
from __future__ import annotations

from . import _env  # noqa: F401

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

solver_app = typer.Typer(no_args_is_help=True, help="Решатель = профиль + классификатор цикла + (по желанию) веса и набор разметки в одном архиве *.sdsolver.zip с метаданными для автоопределения архитектуры.")
console = Console(width=140)


def _tbl(title: str, cols: list[str], rows: list[list]) -> None:
    t = Table(title=title, box=box.SIMPLE_HEAVY)
    for c in cols:
        t.add_column(c)
    for r in rows:
        t.add_row(*[str(x) for x in r])
    console.print(t)


def _issues(issues) -> None:
    for i in issues:
        console.print(f"[{'red' if i.level == 'error' else 'yellow'}]{'ОШИБКА' if i.level == 'error' else 'внимание'}[/] {i.text}")


@solver_app.command("pack", help="Собрать архив решателя из профиля. Профиль без классификатора цикла не упаковывается.")
def pack_cmd(profile: Annotated[str, typer.Option("--profile", "-p", help="профиль из configs/experiments")],
             out: Annotated[Optional[Path], typer.Option("--out", "-o", help="куда писать (по умолчанию <имя>.sdsolver.zip)")] = None,
             name: Annotated[Optional[str], typer.Option(help="имя решателя (по умолчанию имя профиля)")] = None,
             classifier: Annotated[Optional[str], typer.Option(help="подменить классификатор цикла (models/cycle/<имя>)")] = None,
             with_weights: Annotated[bool, typer.Option("--with-weights", help="включить веса моделей (onnx/safetensors/pt), если они лежат на диске")] = False,
             with_dataset: Annotated[bool, typer.Option("--with-dataset", help="включить набор разметки: events_gt.csv, cycle_labels.csv, индекс клипов")] = False,
             run: Annotated[Optional[str], typer.Option("--run", help="id эксперимента (outputs/experiments/<id>): его отчёт войдёт в метаданные")] = None) -> None:
    from . import profiles as PR
    from . import runner as RN
    from . import solver as SV

    prof = RN.with_classifier(PR.load(profile), classifier)
    try:
        res = SV.pack(prof, out, name=name, with_weights=with_weights, with_dataset=with_dataset, run_id=run)
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    m = res["meta"]
    console.print(f"архив: {res['path']} · {res['size_mb']} МБ · файлов {res['files']} · архитектура {m['architecture']['mode']} · отпечаток {m['fingerprint']}")
    for n in m["notes"]:
        console.print(f"[yellow]! {n}[/]")


@solver_app.command("inspect", help="Метаданные решателя из архива или каталога: архитектура по стадиям, компоненты, требования, оценка качества, целостность (sha256).")
def inspect_cmd(path: Path, as_json: Annotated[bool, typer.Option("--json")] = False) -> None:
    from . import solver as SV

    info = SV.inspect(path)
    m = info["meta"]
    if as_json:
        console.print_json(json.dumps(m, ensure_ascii=False, default=str))
        raise typer.Exit(0 if info["ok"] else 1)
    a = m["architecture"]
    console.print(f"решатель [bold]{m['id']}[/] · {a['mode']} · создан {m['created']} · отпечаток {m['fingerprint']} · целостность: {'ОК' if info['ok'] else 'НАРУШЕНА'}")
    for p in info["problems"]:
        console.print(f"[red]{p}[/]")
    rows = []
    for s in a["stages"]:
        d = {k: v for k, v in s.items() if k not in ("id", "summary", "weights", "detectors")}
        extra = s.get("weights", {}).get("id") or ",".join(x["id"] for x in s.get("detectors", [])) or ""
        rows.append([s["id"], extra, "; ".join(f"{k}={v}" for k, v in d.items())[:100]])
    _tbl("Стадии", ["стадия", "модель", "параметры"], rows)
    cl = next((s.get("summary") for s in a["stages"] if s["id"] == "cycle_classifier"), None)
    if cl:
        console.print(f"классификатор: {cl['name']} · признаков {cl['n_features']} {cl['groups']} · члены {cl['members']} · AUC out-of-fold {cl['auc_oof']} · обучен на {cl['n']} циклах ({cl['n_pos']} затяжек)")
    r = a["resources_estimate"]
    console.print(f"память (оценка): пик {r['peak_gb']} ГБ при всех моделях вместе, ≈ {r['sequential_gb']} ГБ при последовательной загрузке · режимы: {a['streaming']}")
    if m.get("evaluation"):
        e = m["evaluation"]["metrics"]
        console.print(f"оценка: F1 {e['f1']} (P {e['precision']}, R {e['recall']}), порог {e['threshold']}, клипов {e['clips']}, эталон {m['evaluation'].get('gt_fingerprint')}")
    w = m.get("weights") or []
    if w:
        _tbl("Веса", ["id", "в архиве", "файл", "лицензия", "sha256"], [[x["id"], "да" if x["included"] else "нет", x["file"] or x.get("hf_repo") or "rtmlib", x.get("license"), (x.get("sha256") or "")[:12]] for x in w])
    for n in m.get("notes", []):
        console.print(f"[yellow]! {n}[/]")
    raise typer.Exit(0 if info["ok"] else 1)


@solver_app.command("install", help="Установить решатель на этой машине: пакеты → models/, профиль → configs/experiments/, метаданные → models/solvers/. Всё проверяется до записи.")
def install_cmd(path: Path, overwrite: Annotated[bool, typer.Option("--overwrite")] = False, with_dataset: Annotated[bool, typer.Option("--with-dataset", help="разложить набор разметки в labels/ (существующие файлы не затираются)")] = False,
                allow_unknown_weights: Annotated[bool, typer.Option(help="принять .pt, не распознанные сканером безопасности (осознанно)")] = False) -> None:
    from . import solver as SV

    try:
        out = SV.install(path, overwrite=overwrite, install_dataset=with_dataset, allow_unknown_weights=allow_unknown_weights)
    except (ValueError, FileExistsError, RuntimeError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    console.print(f"установлен: {out['name']} ({out['architecture']}) · профиль {out['profile']}")
    for x in out["installed"]:
        console.print(f"  + {x}")
    for x in out["skipped"]:
        console.print(f"[yellow]  ! {x}[/]")
    for i in out["issues"]:
        console.print(f"[{'red' if i['level'] == 'error' else 'yellow'}]  {i['level']}: {i['text']}[/]")
    console.print("проверка окружения (веса, версии пакетов): sd doctor")


@solver_app.command("check", help="Проверить профиль: классификатор обязателен, его признаки обеспечены экстракторами, режим (offline | replay | live) допустим.")
def check_cmd(profile: Annotated[str, typer.Option("--profile", "-p")], mode: Annotated[str, typer.Option(help="offline | replay | live")] = "offline") -> None:
    from . import profiles as PR
    from . import solver as SV

    issues = SV.check(PR.load(profile), mode)
    if not issues:
        console.print("[green]профиль в порядке[/]")
    _issues(issues)
    raise typer.Exit(1 if any(i.level == "error" for i in issues) else 0)


@solver_app.command("list", help="Установленные решатели и доступные классификаторы цикла.")
def list_cmd() -> None:
    from . import model_tools as MT
    from . import solver as SV

    s = SV.installed_solvers()
    _tbl("Решатели", ["имя", "архитектура", "F1", "создан", "отпечаток"], [[x["name"], x["mode"], x["f1"], x["created"], x["fingerprint"]] for x in s] or [["—", "", "", "", ""]])
    rows = []
    for b in MT.list_bundles():
        if b["kind"] == "cycle":
            d = SV.describe_bundle(f"models/cycle/{b['name']}")
            rows.append([b["name"], d["n_features"], d["groups"], d["auc_oof"], d["requires"]["objects"] or "—", "да" if d["requires"]["photo"] else "—", "да" if d["requires"]["vlm"] else "—"])
    _tbl("Классификаторы цикла", ["имя", "признаков", "группы", "AUC oof", "нужны детекторы", "фото", "VLM"], rows or [["—", "", "", "", "", "", ""]])
