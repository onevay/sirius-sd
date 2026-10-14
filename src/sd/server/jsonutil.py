"""Приведение результатов прежних модулей (pandas, numpy, Path, NaN) к валидному JSON."""
from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Any


def clean(o: Any) -> Any:
    """Рекурсивно: NaN/Inf → None, numpy-скаляры → числа Python, даты → ISO, Path → строка, DataFrame → список записей."""
    try:
        import numpy as np
        import pandas as pd
    except ImportError:      # pragma: no cover
        np = pd = None
    if o is None or isinstance(o, (str, bool)):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, int):
        return o
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if pd is not None:
        if isinstance(o, pd.DataFrame):
            return clean(o.to_dict("records"))
        if isinstance(o, pd.Series):
            return clean(o.to_dict())
        if o is pd.NaT:
            return None
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
    if np is not None:
        if isinstance(o, np.generic):
            return clean(o.item())
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
    return str(o)
