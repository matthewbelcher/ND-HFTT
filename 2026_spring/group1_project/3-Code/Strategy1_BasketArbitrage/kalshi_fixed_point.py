from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from typing import Any, Dict, Iterable, List, Optional, Sequence

ONE_CENT = Decimal("0.01")
ONE_DOLLAR = Decimal("1")
PRICE_EPS = Decimal("0.0000001")


@dataclass(frozen=True)
class PriceRange:
    min_cents: Decimal
    max_cents: Decimal
    tick_cents: Decimal

    def contains(self, cents: Decimal) -> bool:
        return self.min_cents - PRICE_EPS <= cents <= self.max_cents + PRICE_EPS


def _to_decimal(value: Any) -> Optional[Decimal]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        return None


def parse_price_cents(value: Any, *, price_in_dollars: bool) -> Optional[float]:
    dec = _to_decimal(value)
    if dec is None:
        return None
    if price_in_dollars:
        return float(dec * Decimal("100"))
    if dec <= ONE_DOLLAR + PRICE_EPS:
        # Compatibility: many endpoints emit *_dollars strings even in generic fields.
        return float(dec * Decimal("100"))
    return float(dec)


def parse_qty(value: Any) -> Optional[float]:
    dec = _to_decimal(value)
    if dec is None or dec <= Decimal("0"):
        return None
    return float(dec)


def parse_price_ranges(market_snapshot: Optional[Dict[str, Any]]) -> List[PriceRange]:
    if not market_snapshot:
        return []
    raw = market_snapshot.get("price_ranges")
    if not isinstance(raw, list):
        return []
    out: List[PriceRange] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        min_d = _to_decimal(item.get("min_price_dollars"))
        max_d = _to_decimal(item.get("max_price_dollars"))
        tick_d = _to_decimal(item.get("tick_size_dollars"))
        if min_d is None or max_d is None or tick_d is None:
            continue
        if tick_d <= Decimal("0"):
            continue
        out.append(
            PriceRange(
                min_cents=min_d * Decimal("100"),
                max_cents=max_d * Decimal("100"),
                tick_cents=tick_d * Decimal("100"),
            )
        )
    out.sort(key=lambda r: (r.min_cents, r.max_cents, r.tick_cents))
    return out


def _quantize_cents(cents: Decimal) -> Decimal:
    # Keep enough precision for subpenny prices.
    return cents.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def _snap_in_range(cents: Decimal, r: PriceRange, mode: str) -> Optional[Decimal]:
    if r.tick_cents <= Decimal("0"):
        return None
    pos = (cents - r.min_cents) / r.tick_cents
    if mode == "floor":
        n = pos.to_integral_value(rounding=ROUND_FLOOR)
    elif mode == "ceil":
        n = pos.to_integral_value(rounding=ROUND_CEILING)
    else:
        n = pos.to_integral_value(rounding=ROUND_HALF_UP)
    snapped = r.min_cents + (n * r.tick_cents)
    if snapped < r.min_cents - PRICE_EPS or snapped > r.max_cents + PRICE_EPS:
        return None
    return _quantize_cents(snapped)


def snap_price_to_tick(
    price_cents: float,
    ranges: Sequence[PriceRange],
    *,
    mode: str = "nearest",
) -> float:
    cents = _to_decimal(price_cents)
    if cents is None:
        return float(price_cents)
    cents = _quantize_cents(cents)

    if not ranges:
        nearest_cent = (cents * Decimal("10")).to_integral_value(rounding=ROUND_HALF_UP) / Decimal(
            "10"
        )
        return float(_quantize_cents(nearest_cent))

    candidates: List[Decimal] = []
    for r in ranges:
        if mode == "ceil" and cents > r.max_cents + PRICE_EPS:
            continue
        if mode == "floor" and cents < r.min_cents - PRICE_EPS:
            continue
        target = min(max(cents, r.min_cents), r.max_cents)
        snapped = _snap_in_range(target, r, mode)
        if snapped is None:
            continue
        candidates.append(snapped)

    if not candidates:
        # Fallback to nearest bound.
        bounds = [r.min_cents for r in ranges] + [r.max_cents for r in ranges]
        if not bounds:
            return float(cents)
        best = min(bounds, key=lambda b: abs(b - cents))
        return float(_quantize_cents(best))

    if mode == "ceil":
        valid = [c for c in candidates if c >= cents - PRICE_EPS]
        if valid:
            return float(min(valid))
        return float(max(candidates))
    if mode == "floor":
        valid = [c for c in candidates if c <= cents + PRICE_EPS]
        if valid:
            return float(max(valid))
        return float(min(candidates))
    # nearest
    best = min(candidates, key=lambda c: (abs(c - cents), c))
    return float(best)


def next_tick_price(price_cents: float, ranges: Sequence[PriceRange]) -> Optional[float]:
    base = _to_decimal(price_cents)
    if base is None:
        return None
    target = base + Decimal("0.0001")
    nxt = snap_price_to_tick(float(target), ranges, mode="ceil")
    if nxt <= float(base) + 1e-9:
        return None
    return nxt


def prev_tick_price(price_cents: float, ranges: Sequence[PriceRange]) -> Optional[float]:
    base = _to_decimal(price_cents)
    if base is None:
        return None
    target = base - Decimal("0.0001")
    prev = snap_price_to_tick(float(target), ranges, mode="floor")
    if prev >= float(base) - 1e-9:
        return None
    return prev


def format_count_fp(count: float) -> str:
    dec = _to_decimal(count) or Decimal("0")
    return f"{dec.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):.2f}"


def is_whole_contract_count(count: float) -> bool:
    dec = _to_decimal(count)
    if dec is None:
        return False
    return dec == dec.to_integral_value(rounding=ROUND_HALF_UP)


def format_price_dollars_from_cents(price_cents: float) -> str:
    dec = _to_decimal(price_cents) or Decimal("0")
    dollars = dec / Decimal("100")
    return f"{dollars.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP):.4f}"


def is_whole_cent_price(price_cents: float) -> bool:
    dec = _to_decimal(price_cents)
    if dec is None:
        return False
    return dec == dec.to_integral_value(rounding=ROUND_HALF_UP)


def price_key(price_cents: float) -> int:
    # 1/10th-cent key to avoid float dictionary key precision issues.
    dec = _to_decimal(price_cents) or Decimal("0")
    return int((dec * Decimal("10")).to_integral_value(rounding=ROUND_HALF_UP))


def apply_order_count_fields(order: Dict[str, Any], count: float) -> None:
    order["count_fp"] = format_count_fp(count)
    if is_whole_contract_count(count):
        order["count"] = int(round(float(count)))
    else:
        order.pop("count", None)


def apply_order_price_fields(
    order: Dict[str, Any],
    side: str,
    price_cents: float,
    *,
    ranges: Optional[Iterable[PriceRange]] = None,
) -> None:
    side_l = side.lower()
    if side_l not in {"yes", "no"}:
        raise ValueError(f"Invalid side: {side}")
    price = float(price_cents)
    if ranges is not None:
        price = snap_price_to_tick(price, list(ranges), mode="nearest")
    dollars_key = f"{side_l}_price_dollars"
    legacy_key = f"{side_l}_price"
    order[dollars_key] = format_price_dollars_from_cents(price)
    if is_whole_cent_price(price):
        order[legacy_key] = int(round(price))
    else:
        order.pop(legacy_key, None)
