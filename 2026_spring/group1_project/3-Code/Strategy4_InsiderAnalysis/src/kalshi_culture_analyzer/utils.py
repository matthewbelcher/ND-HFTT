from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

_DURATION_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)?)(?P<unit>[smhdw])?$", re.IGNORECASE)


def utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def parse_duration_to_seconds(value: str) -> int:
    if value is None:
        raise ValueError("Duration is None")
    value = value.strip()
    m = _DURATION_RE.match(value)
    if not m:
        raise ValueError(f"Invalid duration: {value}")
    num = float(m.group("num"))
    unit = (m.group("unit") or "s").lower()
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
    return int(num * mult)


def parse_since_to_ms(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    value = value.strip()
    # Relative duration like 7d
    if _DURATION_RE.match(value):
        seconds = parse_duration_to_seconds(value)
        return utc_now_ms() - seconds * 1000
    # ISO date or datetime
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError:
        raise ValueError(f"Unrecognized --since value: {value}")


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), default=str)


def to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def from_ms(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

