from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from kalshi_fixed_point import PriceRange, parse_price_cents, parse_price_ranges, parse_qty

PriceLevel = Tuple[float, float]


@dataclass(frozen=True)
class NormalizedBook:
    market_ticker: str
    yes_bid: Optional[PriceLevel]
    yes_ask: Optional[PriceLevel]
    no_bid: Optional[PriceLevel]
    no_ask: Optional[PriceLevel]
    yes_bid_levels: List[PriceLevel]
    no_bid_levels: List[PriceLevel]
    yes_ask_levels: List[PriceLevel]
    no_ask_levels: List[PriceLevel]
    price_ranges: List[PriceRange]


def _price_eq(a: float, b: float, eps: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= eps


def _parse_levels(raw: Any, *, price_in_dollars: bool) -> List[PriceLevel]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("levels") or raw.get("orderbook") or []
    if not isinstance(raw, Iterable):
        return []

    out: List[PriceLevel] = []
    for item in raw:
        price = None
        qty = None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            price, qty = item[0], item[1]
        elif isinstance(item, dict):
            price = item.get("price") or item.get("px") or item.get("price_cents")
            qty = (
                item.get("qty")
                or item.get("quantity")
                or item.get("count")
                or item.get("count_fp")
            )
        if price is None or qty is None:
            continue
        p = parse_price_cents(price, price_in_dollars=price_in_dollars)
        q = parse_qty(qty)
        if p is None or q is None:
            continue
        if p < 0 or p > 100:
            continue
        out.append((p, q))
    return out


def _sort_bids(levels: List[PriceLevel]) -> List[PriceLevel]:
    out = [(p, q) for p, q in levels if q > 0]
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def _snap_price(snapshot: Optional[Dict[str, Any]], keys: List[str]) -> Optional[float]:
    if not snapshot:
        return None
    for k in keys:
        if k not in snapshot:
            continue
        v = snapshot.get(k)
        if v is None:
            continue
        price = parse_price_cents(v, price_in_dollars=False)
        if price is not None:
            return price
    return None


def _level_qty_at_price(levels: List[PriceLevel], price: float) -> Optional[float]:
    for p, q in levels:
        if _price_eq(p, price):
            return q
    return None


def _apply_snapshot_level(
    levels: List[PriceLevel],
    snap_price: Optional[float],
) -> Optional[PriceLevel]:
    if snap_price is None:
        return None
    qty = _level_qty_at_price(levels, snap_price)
    if qty is not None:
        return (snap_price, qty)
    if not levels:
        if _price_eq(snap_price, 0.0) or _price_eq(snap_price, 100.0):
            return None
        return (snap_price, 0.0)
    return None


def normalize_orderbook(
    market_ticker: str,
    orderbook: Dict[str, Any],
    *,
    depth: Optional[int] = None,
    market_snapshot: Optional[Dict[str, Any]] = None,
) -> NormalizedBook:
    ob = orderbook.get("orderbook") or {}
    ob_fp = orderbook.get("orderbook_fp") or {}
    ranges = parse_price_ranges(market_snapshot)

    yes_levels = _parse_levels(ob.get("yes"), price_in_dollars=False)
    no_levels = _parse_levels(ob.get("no"), price_in_dollars=False)
    if not yes_levels and ob_fp.get("yes_dollars"):
        yes_levels = _parse_levels(ob_fp.get("yes_dollars"), price_in_dollars=True)
    if not no_levels and ob_fp.get("no_dollars"):
        no_levels = _parse_levels(ob_fp.get("no_dollars"), price_in_dollars=True)

    yes_levels = _sort_bids(yes_levels)
    no_levels = _sort_bids(no_levels)

    if depth is not None and depth > 0:
        yes_levels = yes_levels[:depth]
        no_levels = no_levels[:depth]

    yes_bid = yes_levels[0] if yes_levels else None
    no_bid = no_levels[0] if no_levels else None

    yes_ask_levels = [(100.0 - p, q) for p, q in no_levels if 0 < 100.0 - p < 100.0]
    no_ask_levels = [(100.0 - p, q) for p, q in yes_levels if 0 < 100.0 - p < 100.0]

    yes_ask = yes_ask_levels[0] if yes_ask_levels else None
    no_ask = no_ask_levels[0] if no_ask_levels else None

    yes_bid_snap = _snap_price(market_snapshot, ["yes_bid", "yes_bid_dollars"])
    no_bid_snap = _snap_price(market_snapshot, ["no_bid", "no_bid_dollars"])
    yes_ask_snap = _snap_price(market_snapshot, ["yes_ask", "yes_ask_dollars"])
    no_ask_snap = _snap_price(market_snapshot, ["no_ask", "no_ask_dollars"])

    snap_yes_bid = _apply_snapshot_level(yes_levels, yes_bid_snap)
    if snap_yes_bid is not None:
        yes_bid = snap_yes_bid
    snap_no_bid = _apply_snapshot_level(no_levels, no_bid_snap)
    if snap_no_bid is not None:
        no_bid = snap_no_bid
    snap_yes_ask = _apply_snapshot_level(yes_ask_levels, yes_ask_snap)
    if snap_yes_ask is not None:
        yes_ask = snap_yes_ask
    snap_no_ask = _apply_snapshot_level(no_ask_levels, no_ask_snap)
    if snap_no_ask is not None:
        no_ask = snap_no_ask

    return NormalizedBook(
        market_ticker=market_ticker,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        yes_bid_levels=yes_levels,
        no_bid_levels=no_levels,
        yes_ask_levels=yes_ask_levels,
        no_ask_levels=no_ask_levels,
        price_ranges=ranges,
    )
