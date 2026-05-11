"""One-way feed latency from broker message timestamps vs local receive time (UTC)."""

from __future__ import annotations

import statistics
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clamp_fractional_z(s: str) -> str:
    """Coinbase sometimes sends nanosecond fractions; trim to 6 digits before Z."""
    s = s.strip()
    if not s.endswith("Z") or "T" not in s or "." not in s.split("T", 1)[1]:
        return s
    head, rest = s[:-1].split(".", 1)  # drop Z from rest side
    digits = "".join(ch for ch in rest if ch.isdigit())
    if len(digits) <= 6:
        return s
    return f"{head}.{digits[:6]}Z"


def parse_message_ts(value: Optional[object]) -> Optional[datetime]:
    """Parse Kalshi / Coinbase ISO strings or numeric Unix seconds/ms."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        sec = float(value)
        if sec > 1e12:  # ms
            sec /= 1000.0
        return datetime.fromtimestamp(sec, tz=timezone.utc)
    if not isinstance(value, str):
        return None
    s = _clamp_fractional_z(value.strip())
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def coinbase_reference_datetime(data: Dict[str, Any]) -> Optional[datetime]:
    """
    Prefer top-level ``timestamp``; if missing, use the newest ``time`` on trades
    inside ``market_trades`` events (some builds omit the envelope field).
    """
    dt = parse_message_ts(data.get("timestamp"))
    if dt is not None:
        return dt
    if data.get("channel") != "market_trades":
        return None
    best: Optional[datetime] = None
    for ev in data.get("events") or []:
        for tr in ev.get("trades") or []:
            t = parse_message_ts(tr.get("time"))
            if t is not None and (best is None or t > best):
                best = t
    return best


def latency_ms_recv_minus_msg(msg_ts: datetime, recv_ts: datetime) -> float:
    """Positive => message timestamp is in the past (normal). Negative => local clock behind server."""
    return (recv_ts - msg_ts).total_seconds() * 1000.0


def summarize_samples(samples: Deque[float], last_ms: Optional[float]) -> str:
    if last_ms is None and not samples:
        return "n/a"
    parts: List[str] = []
    if last_ms is not None:
        parts.append(f"last={last_ms:+.0f}ms")
    if samples:
        arr = list(samples)
        parts.append(f"avg={statistics.mean(arr):+.0f}ms")
        if len(arr) >= 5:
            arr_sorted = sorted(arr)
            k = max(0, min(len(arr_sorted) - 1, round((len(arr_sorted) - 1) * 0.95)))
            p95 = arr_sorted[k]
            parts.append(f"p95={p95:+.0f}ms")
        parts.append(f"n={len(arr)}")
    return " ".join(parts)
