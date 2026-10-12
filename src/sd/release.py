"""Выпуск модели: ворота качества → решатель `.sdsolver.zip` + PDF-отчёт + манифест выпуска. Единственный путь модели из Dev Studio в продукт.

Ворота (`gates`) проверяют то, что раньше держалось только на внимательности:
    run_ok        прогон оценки существует, по ручному эталону событий, без клипов с ошибкой обработки
    f1            F1 прогона не ниже порога (и показывается нижняя граница интервала)
    same_profile  профиль на диске — тот же, что оценивался (отпечаток совпадает): нельзя выпустить «не то, что проверяли»
    no_leak       классификатор не обучался на клипах этого прогона (иначе цифра качества оптимистична)
    pipeline      профиль собирает рабочий конвейер (классификатор есть, детекторы и модели на месте)
    licenses      веса с некоммерческой/AGPL-лицензией — предупреждение (выпуск возможен, решение за владельцем)
`force` позволяет выпустить при непройденных воротах, но это записывается в манифест выпуска и видно при установке.

    outputs/releases/<имя>/<имя>.sdsolver.zip   решатель (профиль, пакеты, при желании веса)
    outputs/releases/<имя>/report.pdf           отчёт прогона, подтверждающего качество
    outputs/releases/<имя>/release.json         ворота, метрики, отпечатки, git, sha256 архива, подпись HMAC (если задан ключ SD_RELEASE_KEY)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from pathlib import Path

from . import experiments as XP
from . import profiles as PR
from .paths import OUTPUTS

RELEASES = OUTPUTS / "releases"
RISKY_LICENSES = ("agpl", "nc", "non-commercial", "unknown", "research")


def _license_issues(profile: PR.Profile) -> list[str]:
    from .models import load_registry

    try:
        reg = load_registry()
    except Exception:       # noqa: BLE001
        return []
    o, c = profile.opts(), profile.cfg()
    ids = [str(c.get("pose", {}).get("weights") or "")] + list(o.get("objects") or [])
    out = []
    for i in ids:
        spec = reg.get(i)
        lic = str(getattr(spec, "license", "") or "").lower() if spec else ""
        if spec and any(k in lic for k in RISKY_LICENSES):
            out.append(f"{i}: лицензия {spec.license}")
    return out


def gates(profile_name: str, run_id: str, min_f1: float = 0.7, exp_root: Path | None = None) -> list[dict]:
    """Список ворот: {id, title, ok (True/False/None — справочно), detail}."""
    from . import solver as SV
    from .runner import training_leak

    out: list[dict] = []
    try:
        prof = PR.load_named(profile_name)
    except Exception as e:      # noqa: BLE001
        return [dict(id="profile", title="профиль существует", ok=False, detail=str(e))]
    try:
        rep, meta, _, _ = XP.load_run(run_id, exp_root)
    except Exception as e:      # noqa: BLE001
        return [dict(id="run_ok", title="прогон оценки существует", ok=False, detail=str(e))]
    failed = [c["clip_id"] for c in meta.get("clips", []) if c.get("error")]
    out.append(dict(id="run_ok", title="прогон по ручному эталону, без сбоев", ok=rep.mode == "events" and not failed,
                    detail=f"режим {rep.mode}, клипов {len(meta.get('clips', []))}, со сбоем {len(failed)}" + (f": {', '.join(failed[:3])}" if failed else "")))
    m, ci = rep.metrics, (rep.ci.get("f1") or (None, None))
    out.append(dict(id="f1", title=f"Event F1 ≥ {min_f1:.2f}", ok=m["f1"] >= min_f1,
                    detail=f"F1 {m['f1']:.3f} (95 %: {ci[0]:.2f}–{ci[1]:.2f})" if ci[0] is not None else f"F1 {m['f1']:.3f}"))
    from .runner import with_classifier

    fp_now = prof.fingerprint()
    out.append(dict(id="same_profile", title="профиль не менялся после оценки", ok=fp_now == meta.get("fingerprint"),
                    detail=f"сейчас {fp_now}, в прогоне {meta.get('fingerprint')}" + ("" if fp_now == meta.get("fingerprint") else
                                                                                      " — оценивался другой профиль/классификатор/параметры: оцените текущий")))
    leak = training_leak(with_classifier(prof, None), [c["clip_id"] for c in meta.get("clips", [])])
    out.append(dict(id="no_leak", title="классификатор не обучался на клипах оценки", ok=not leak["clips"],
                    detail=(f"пересечение {len(leak['clips'])} клипов (источник: {leak['source']}) — нужна оценка на отложенных клипах или вложенная oof-оценка"
                            if leak["clips"] else f"пересечений нет (источник: {leak['source']})")))
    errs = [i.text for i in SV.errors(prof)]
    out.append(dict(id="pipeline", title="конвейер собирается", ok=not errs, detail="; ".join(errs) or "классификатор и модели на месте"))
    lic = _license_issues(prof)
    out.append(dict(id="licenses", title="лицензии весов допускают коммерцию", ok=None if lic else True,
                    detail=("требует решения: " + "; ".join(lic)) if lic else "рискованных лицензий в реестре не найдено"))
    return out


def passed(gs: list[dict]) -> bool:
    return all(g["ok"] is not False for g in gs)


def _sign(payload: bytes) -> str | None:
    key = os.environ.get("SD_RELEASE_KEY")
    return hmac.new(key.encode("utf-8"), payload, hashlib.sha256).hexdigest() if key else None


def make(profile_name: str, run_id: str, name: str | None = None, min_f1: float = 0.7, with_weights: bool = False, force: bool = False, root: Path | None = None,
         exp_root: Path | None = None, progress=None) -> dict:
    """Проверить ворота, упаковать решатель, собрать PDF прогона, записать release.json. Ворота не пройдены и не `force` → ValueError со списком."""
    from . import reports as R
    from . import solver as SV
    from .paths import slug

    gs = gates(profile_name, run_id, min_f1, exp_root)
    if not passed(gs) and not force:
        bad = [f"{g['title']}: {g['detail']}" for g in gs if g["ok"] is False]
        raise ValueError("ворота выпуска не пройдены:\n  - " + "\n  - ".join(bad))
    nm = slug(name or f"{profile_name}_{time.strftime('%Y%m%d_%H%M')}")
    d = (root or RELEASES) / nm
    if d.exists():
        raise FileExistsError(f"выпуск {nm} уже есть: {d}")
    d.mkdir(parents=True)
    if progress:
        progress(1, 3, "упаковка решателя")
    prof = PR.load_named(profile_name)
    packed = SV.pack(prof, d / nm, name=nm, with_weights=with_weights, run_id=run_id, exp_root=exp_root)
    if progress:
        progress(2, 3, "PDF-отчёт прогона")
    report = R.build(run_id, out=d / "report.pdf", root=exp_root)
    zpath = Path(packed["path"])
    sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    rep, meta, _, _ = XP.load_run(run_id, exp_root)
    rec = dict(name=nm, created=time.strftime("%Y-%m-%dT%H:%M:%S"), profile=profile_name, fingerprint=prof.fingerprint(), run=run_id,
               metrics={k: rep.metrics.get(k) for k in ("f1", "precision", "recall", "tp", "fp", "fn", "fp_per_hour", "threshold", "clips")}, ci=rep.ci.get("f1"),
               gt_fingerprint=meta.get("gt_fingerprint"), gates=gs, passed=passed(gs), forced=bool(force and not passed(gs)), solver=zpath.name, solver_sha256=sha,
               report=report.name, git=XP.git_state(), host=XP.host_info())
    payload = json.dumps(rec, ensure_ascii=False, sort_keys=True).encode("utf-8")
    rec["signature"] = _sign(payload)
    rec["signature_kind"] = "hmac-sha256 (SD_RELEASE_KEY)" if rec["signature"] else None
    (d / "release.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if progress:
        progress(3, 3, "готово")
    return rec


def verify(release_dir: str | Path) -> dict:
    """Проверка выпуска: sha256 архива и (если задан SD_RELEASE_KEY) подпись манифеста."""
    d = Path(release_dir)
    rec = json.loads((d / "release.json").read_text(encoding="utf-8"))
    z = d / rec["solver"]
    sha_ok = z.exists() and hashlib.sha256(z.read_bytes()).hexdigest() == rec["solver_sha256"]
    sig = rec.get("signature")
    body = {k: v for k, v in rec.items() if k not in ("signature", "signature_kind")}
    expect = _sign(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return dict(sha_ok=sha_ok, signed=bool(sig), signature_ok=(sig == expect) if sig and expect else None)


def list_releases(root: Path | None = None) -> list[dict]:
    base = root or RELEASES
    out = []
    for f in sorted(base.glob("*/release.json"), reverse=True) if base.exists() else []:
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            out.append(dict(name=r["name"], created=r["created"], profile=r["profile"], run=r["run"], f1=(r.get("metrics") or {}).get("f1"), passed=r.get("passed"),
                            forced=r.get("forced"), dir=str(f.parent)))
        except Exception:       # noqa: BLE001
            continue
    return out
