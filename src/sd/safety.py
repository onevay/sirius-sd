"""Статическая проверка PyTorch-весов (.pt/.pth) ПЕРЕД загрузкой.

`torch.load(..., weights_only=False)` исполняет pickle, а значит `.pt` от неизвестного автора может выполнить произвольный код.
Ultralytics грузит чекпойнты именно так (в чекпойнте лежат классы модели), поэтому веса от сообщества сначала сканируем:
читаем опкоды pickle через `pickletools` (код НЕ выполняется) и проверяем, какие глобальные объекты чекпойнт собирается вызвать.

Политика (консервативная, лучше отказать, чем пропустить):
- GLOBAL/STACK_GLOBAL с точкой в имени (`warnings.sys.modules`) -> danger (обход белого списка через qualified name);
- разрешены только «классоподобные» имена (CamelCase) из разрешённых модулей torch/ultralytics.nn/collections/numpy/... и
  явный короткий список функций (`_rebuild_tensor_v2`, `set`, ...); всё остальное -> unknown (нужен ручной разбор);
- os / subprocess / sys / socket / importlib / eval / exec / open / __import__ ... -> danger;
- `getattr` разрешён ТОЛЬКО по шаблону `getattr(<класс ultralytics.nn.modules.*>, 'forward')` (так сохраняет forward голов Pose/Detect
  официальный YOLO11); любой другой getattr (в т.ч. через memo) -> danger.
Форматы safetensors/ONNX сканировать не нужно: они не исполняют код.
"""
from __future__ import annotations

import pickletools
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

DANGER_MODULES = {
    "os", "posix", "nt", "subprocess", "sys", "socket", "shutil", "importlib", "runpy", "pty", "ctypes", "code", "codeop",
    "pickle", "_pickle", "marshal", "webbrowser", "urllib", "urllib2", "http", "ftplib", "smtplib", "requests", "asyncio",
    "multiprocessing", "threading", "pdb", "signal", "tempfile", "glob", "fileinput", "platform", "getpass", "pip",
    "builtins_", "types", "inspect", "operator", "functools", "itertools", "warnings", "linecache", "io", "_io", "zipimport",
    "torch.hub", "torch.jit", "torch.utils", "torch.distributed", "torch.package", "torch._dynamo", "torch.fx",
}
DANGER_BUILTINS = {"eval", "exec", "compile", "open", "__import__", "getattr", "setattr", "delattr", "breakpoint", "input",
                   "globals", "locals", "vars", "exit", "quit", "memoryview", "type", "super", "object.__reduce__"}

CLASS_LIKE = re.compile(r"^[A-Z][A-Za-z0-9_]*$")

# модули, из которых допустимы классоподобные имена
ALLOW_CLASS_MODULES = (
    "torch", "torch.nn", "torch._utils", "torch.storage", "torch._tensor", "torch.optim",
    "ultralytics.nn", "ultralytics.utils", "collections", "pathlib", "datetime", "argparse", "numpy", "numpy.core.multiarray",
    "numpy._core.multiarray", "numpy.dtypes",
)
# явные пары (модуль, имя) для функций/не-CamelCase объектов
ALLOW_EXACT = {
    ("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_tensor"), ("torch._utils", "_rebuild_parameter"),
    ("torch._utils", "_rebuild_parameter_with_state"), ("torch._utils", "_rebuild_qtensor"),
    ("collections", "OrderedDict"), ("copyreg", "_reconstructor"), ("copy_reg", "_reconstructor"), ("_codecs", "encode"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"), ("numpy", "ndarray"), ("numpy", "dtype"),
    # проверено вручную 2026-10-07 по чекпойнту basant18/Smoking-detection-YOLO26s (сырой тренировочный .pt):
    # torch.device — конструктор устройства; v8DetectionLoss — класс функции потерь Ultralytics (без побочных эффектов)
    ("torch", "device"), ("ultralytics.utils.loss", "v8DetectionLoss"),
    ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "dict"), ("builtins", "list"), ("builtins", "tuple"),
    ("builtins", "int"), ("builtins", "float"), ("builtins", "bool"), ("builtins", "bytes"), ("builtins", "bytearray"),
    ("builtins", "slice"), ("builtins", "complex"), ("builtins", "str"), ("builtins", "range"),
    ("__builtin__", "set"), ("__builtin__", "frozenset"), ("__builtin__", "dict"), ("__builtin__", "list"),
    ("__builtin__", "tuple"), ("__builtin__", "int"), ("__builtin__", "float"), ("__builtin__", "bool"),
    ("__builtin__", "bytes"), ("__builtin__", "slice"), ("__builtin__", "str"), ("__builtin__", "range"),
    *[("torch", n) for n in ("float32", "float16", "bfloat16", "float64", "int8", "int16", "int32", "int64", "uint8", "bool",
                             "complex64", "float", "half", "double", "long", "int")],
}
GETATTR_NAMES = {("builtins", "getattr"), ("__builtin__", "getattr")}
GETATTR_OK_ATTRS = {"forward"}

_STR_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "STRING", "BINSTRING", "SHORT_BINSTRING"}
_MEMO_PUT = {"BINPUT", "LONG_BINPUT", "MEMOIZE", "PUT"}
_MEMO_GET = {"BINGET", "LONG_BINGET", "GET"}


@dataclass
class ScanReport:
    path: str
    verdict: str = "ok"                       # ok | unknown | danger | error
    globals_found: list[str] = field(default_factory=list)
    danger: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict == "ok"

    def summary(self) -> str:
        s = f"{self.verdict.upper()}: {Path(self.path).name} — глобалов {len(self.globals_found)}"
        if self.danger:
            s += f", ОПАСНЫХ {len(self.danger)}: {self.danger[:5]}"
        if self.unknown:
            s += f", неизвестных {len(self.unknown)}: {self.unknown[:5]}"
        return s


def classify(module: str, name: str) -> str:
    """ok | unknown | danger для пары (module, name) из pickle."""
    if "." in name or not module or not name:
        return "danger"  # qualified name — обход белого списка (pickle protocol >= 4)
    top = module.split(".")[0]
    if module in ("builtins", "__builtin__") and name in DANGER_BUILTINS:
        return "danger"
    if module in DANGER_MODULES or top in DANGER_MODULES or any(module.startswith(m + ".") for m in DANGER_MODULES):
        return "danger"
    if (module, name) in ALLOW_EXACT:
        return "ok"
    if CLASS_LIKE.match(name) and any(module == m or module.startswith(m + ".") for m in ALLOW_CLASS_MODULES):
        return "ok"
    return "unknown"


def _scan_pickle(data: bytes) -> tuple[list[tuple[str, str]], list[str]]:
    """Возвращает (все глобалы, список нарушений по getattr/memo). Код pickle не исполняется."""
    ops = list(pickletools.genops(data))
    out: list[tuple[str, str]] = []
    recent: list[str] = []
    violations: list[str] = []
    getattr_memo: set = set()          # индексы memo, где лежит getattr
    last_global_getattr = None         # позиция GLOBAL getattr, ожидающая проверки шаблона
    for i, (op, arg, _pos) in enumerate(ops):
        if op.name in _STR_OPS:
            recent = (recent + [arg])[-4:]
        if op.name in ("GLOBAL", "INST"):
            mod, name = str(arg).split(" ", 1)
            out.append((mod, name))
            if (mod, name) in GETATTR_NAMES:
                last_global_getattr = i
                # ожидаем: [memo*] GLOBAL <класс> [memo*] STRING [memo*] TUPLE2 [memo*] REDUCE
                seq = [(o.name, a) for (o, a, _) in ops[i + 1:i + 14] if o.name not in _MEMO_PUT]
                ok = (len(seq) >= 4 and seq[0][0] == "GLOBAL" and seq[1][0] in _STR_OPS and seq[2][0] == "TUPLE2" and seq[3][0] == "REDUCE")
                if ok:
                    cmod, cname = str(seq[0][1]).split(" ", 1)
                    ok = (cmod.startswith("ultralytics.nn.modules") and CLASS_LIKE.match(cname) is not None
                          and seq[1][1] in GETATTR_OK_ATTRS)
                if not ok:
                    violations.append(f"getattr вне разрешённого шаблона (позиция опкода {i})")
                else:  # запомним memo-индекс, если getattr кладут в memo (повторное использование запрещено)
                    nxt = ops[i + 1] if i + 1 < len(ops) else None
                    if nxt and nxt[0].name in _MEMO_PUT and nxt[1] is not None:
                        getattr_memo.add(nxt[1])
        elif op.name == "STACK_GLOBAL":
            if len(recent) >= 2:
                out.append((recent[-2], recent[-1]))
                if (recent[-2], recent[-1]) in GETATTR_NAMES:
                    violations.append("getattr через STACK_GLOBAL")
            else:
                out.append(("<unresolved>", "STACK_GLOBAL"))
        elif op.name in _MEMO_GET and arg in getattr_memo:
            violations.append(f"повторное использование getattr из memo (опкод {i})")
    return out, violations


def scan_torch_archive(path: str | Path) -> ScanReport:
    path = Path(path)
    rep = ScanReport(path=str(path))
    glob: list[tuple[str, str]] = []
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
                bad = [n for n in names if n.startswith(("/", "\\")) or ".." in Path(n).parts]
                if bad:
                    rep.danger.append(f"путь в архиве выходит за корень: {bad[:3]}")
                pkls = [n for n in names if n.endswith(".pkl")]
                if not pkls:
                    rep.notes.append("в архиве нет *.pkl (возможно, не чекпойнт PyTorch)")
                for n in pkls:
                    g, viol = _scan_pickle(z.read(n))
                    glob += g
                    rep.danger += viol
        else:
            rep.notes.append("legacy-формат (не zip): сканирую поток целиком")
            glob, viol = _scan_pickle(path.read_bytes())
            rep.danger += viol
    except Exception as e:  # битый файл
        rep.verdict = "error"
        rep.notes.append(f"{type(e).__name__}: {e}")
        return rep

    seen: dict[str, str] = {}
    for m, n in glob:
        full = f"{m} {n}"
        if full not in seen:
            seen[full] = classify(m, n)
    # getattr, прошедший шаблон, не считаем опасным; остальные уже добавлены в rep.danger
    rep.globals_found = sorted(seen)
    for g, c in seen.items():
        if c == "danger" and g.split(" ")[1] != "getattr":
            rep.danger.append(g)
    rep.unknown = [g for g, c in seen.items() if c == "unknown"]
    rep.verdict = "danger" if rep.danger else ("unknown" if rep.unknown else "ok")
    return rep


def require_safe(path: str | Path, allow_unknown: bool = False) -> ScanReport:
    """Бросает RuntimeError, если файл нельзя безопасно загружать."""
    rep = scan_torch_archive(path)
    if rep.verdict in ("danger", "error") or (rep.verdict == "unknown" and not allow_unknown):
        raise RuntimeError("Веса не прошли проверку безопасности: " + rep.summary())
    return rep
