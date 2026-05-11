from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def parse_iso_ts(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError:
        return None


def get_price_dollars(obj: Dict[str, Any], key_base: str) -> Optional[float]:
    val = obj.get(f"{key_base}_dollars")
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    cents = obj.get(f"{key_base}_cents")
    if cents is not None:
        try:
            return float(cents) / 100.0
        except (TypeError, ValueError):
            pass
    legacy = obj.get(key_base)
    if legacy is not None:
        try:
            legacy_val = float(legacy)
            return legacy_val / 100.0 if legacy_val > 1.0 else legacy_val
        except (TypeError, ValueError):
            pass
    return None


def extract_settlement_value(market: Dict[str, Any]) -> Optional[float]:
    for key in ("settlement_value_dollars", "settlement_value"):
        val = market.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    cents = market.get("settlement_value_cents")
    if cents is not None:
        try:
            return float(cents) / 100.0
        except (TypeError, ValueError):
            pass
    return None

