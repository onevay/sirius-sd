"""Положение камер на карте. Камера = папка `<район>-<индекс>-<время>`; координаты и название улицы берутся из `cameras.yaml` рядом с папками (`SD_STREAMS`), а пока файла нет —
раскладываются детерминированно вокруг центра района, чтобы карта работала на имитации сразу.

cameras.yaml (ключ — `<район>-<индекс>`, как в имени папки):
    gatchina-03: {lat: 59.5762, lon: 30.1291, street: "ул. Соборная", title: "Перекрёсток у вокзала"}
"""
from __future__ import annotations

from pathlib import Path

import yaml

# центры районов (широта, долгота); неизвестный район — центр Санкт-Петербурга
DISTRICTS = {
    "gatchina": (59.5764, 30.1286), "гатчина": (59.5764, 30.1286),
    "pavlovsk": (59.6833, 30.4500), "павловск": (59.6833, 30.4500),
    "tsarskoe-selo": (59.7167, 30.3939), "pushkin": (59.7167, 30.3939), "пушкин": (59.7167, 30.3939),
    "peterhof": (59.8833, 29.9000), "петергоф": (59.8833, 29.9000),
    "kolpino": (59.7500, 30.6000), "колпино": (59.7500, 30.6000),
    "kronstadt": (59.9900, 29.7700), "кронштадт": (59.9900, 29.7700),
}
DEFAULT_CENTER = (59.9386, 30.3141)
STREETS = ["Центральная ул.", "Вокзальная ул.", "Советский пр.", "Школьная ул.", "Парковая ул.", "Рыночная пл.", "Набережная ул.", "Лесная ул."]


def load_overrides(root: str | Path) -> dict[str, dict]:
    f = Path(root) / "cameras.yaml"
    if not f.exists():
        return {}
    try:
        raw = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)} if isinstance(raw, dict) else {}


def locate(district: str, index: str, overrides: dict[str, dict] | None = None) -> dict:
    """{lat, lon, street, title, placed}: `placed=True` — координаты заданы вручную (иначе условная раскладка)."""
    cam = f"{district}-{index}"
    o = (overrides or {}).get(cam, {})
    n = int(index) if str(index).isdigit() else sum(map(ord, str(index)))
    lat0, lon0 = DISTRICTS.get(district.lower(), DEFAULT_CENTER)
    dlat = (((n * 37) % 11) - 5) * 0.0035
    dlon = (((n * 53) % 13) - 6) * 0.0055
    lat, lon = o.get("lat"), o.get("lon")
    placed = isinstance(lat, (int, float)) and isinstance(lon, (int, float))
    return dict(lat=float(lat) if placed else lat0 + dlat, lon=float(lon) if placed else lon0 + dlon, placed=bool(placed),
                street=str(o.get("street") or STREETS[n % len(STREETS)]), title=str(o.get("title") or f"{district.capitalize()}, камера {index}"))
