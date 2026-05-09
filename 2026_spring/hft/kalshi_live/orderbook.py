"""Kalshi binary market order book (YES / NO bids)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


class KalshiOrderBook:
    """Maintains YES and NO aggregated price levels from snapshot + deltas."""

    def __init__(self) -> None:
        self.yes: Dict[float, float] = {}
        self.no: Dict[float, float] = {}

    def clear(self) -> None:
        self.yes.clear()
        self.no.clear()

    def apply_snapshot(self, msg: Dict[str, Any]) -> None:
        self.clear()
        for side_key, book in (("yes_dollars_fp", self.yes), ("no_dollars_fp", self.no)):
            levels = msg.get(side_key) or []
            for pair in levels:
                if not pair or len(pair) < 2:
                    continue
                price = float(pair[0])
                qty = float(pair[1])
                if qty > 0:
                    book[price] = qty

    def apply_delta(self, msg: Dict[str, Any]) -> None:
        price = float(msg.get("price_dollars", 0))
        delta = float(msg.get("delta_fp", 0))
        side = msg.get("side", "")
        book = self.yes if side == "yes" else self.no if side == "no" else None
        if book is None:
            return
        new_qty = book.get(price, 0.0) + delta
        if new_qty <= 0:
            book.pop(price, None)
        else:
            book[price] = new_qty

    def snapshot_tuple(self) -> Tuple[Dict[float, float], Dict[float, float]]:
        return (dict(self.yes), dict(self.no))


def compute_ob_from_book(
    yes_book: Dict[float, float], no_book: Dict[float, float], n_levels: int = 10
) -> Dict[str, float]:
    """Match full_analysis.compute_ob_from_book for feature parity."""
    yes_prices = sorted(yes_book.keys(), reverse=True)
    no_prices = sorted(no_book.keys(), reverse=True)

    best_yes_bid = yes_prices[0] if yes_prices else 0.0
    best_yes_bid_qty = float(yes_book.get(best_yes_bid, 0.0))
    best_no_bid = no_prices[0] if no_prices else 0.0
    best_no_bid_qty = float(no_book.get(best_no_bid, 0.0))

    yes_ask_implied = 1.0 - best_no_bid
    if best_yes_bid > 0 or best_no_bid > 0:
        mid_price = (best_yes_bid + yes_ask_implied) / 2.0
    else:
        mid_price = 0.5

    spread = yes_ask_implied - best_yes_bid if yes_ask_implied > best_yes_bid else 0.0

    obi_levels: Dict[str, float] = {}
    yes_depth = 0.0
    no_depth = 0.0
    for level in range(1, n_levels + 1):
        if level <= len(yes_prices):
            yes_depth += float(yes_book.get(yes_prices[level - 1], 0.0))
        if level <= len(no_prices):
            no_depth += float(no_book.get(no_prices[level - 1], 0.0))
        total = yes_depth + no_depth
        obi_levels[f"obi_{level}"] = (yes_depth - no_depth) / total if total > 0 else 0.0

    features = {
        "mid_price": mid_price,
        "best_yes_bid": best_yes_bid,
        "best_yes_bid_qty": best_yes_bid_qty,
        "best_no_bid": best_no_bid,
        "best_no_bid_qty": best_no_bid_qty,
        "spread": spread,
        "yes_depth_total": float(sum(yes_book.values())),
        "no_depth_total": float(sum(no_book.values())),
        "n_yes_levels": float(len(yes_book)),
        "n_no_levels": float(len(no_book)),
    }
    features.update(obi_levels)
    return features


def _fmt_ob_qty(q: float) -> str:
    if abs(q - round(q)) < 1e-6:
        return f"{q:,.0f}"
    return f"{q:,.2f}"


def format_order_book_display(
    yes_book: Dict[float, float],
    no_book: Dict[float, float],
    *,
    max_levels: int = 14,
) -> str:
    """
    Human-readable depth: YES bids (high → low) and NO bids (high → low), with qty per level.
    Kalshi lists aggregated resting size at each one-cent increment on each side.
    """
    yes_levels: List[Tuple[float, float]] = sorted(
        ((p, q) for p, q in yes_book.items() if q and q > 0),
        key=lambda t: t[0],
        reverse=True,
    )[:max_levels]
    no_levels: List[Tuple[float, float]] = sorted(
        ((p, q) for p, q in no_book.items() if q and q > 0),
        key=lambda t: t[0],
        reverse=True,
    )[:max_levels]

    lines = [
        "  ORDER BOOK  (YES bid = buy YES at $; NO bid = buy NO at $; implied YES ask ≈ 1 − best NO bid)",
        f"  {'YES price':>10}  {'YES qty':>12}  │  {'NO price':>10}  {'NO qty':>12}",
        f"  {'-' * 10}  {'-' * 12}  │  {'-' * 10}  {'-' * 12}",
    ]
    if not yes_levels and not no_levels:
        lines.append("  (empty — waiting for snapshot or no resting liquidity)")
        return "\n".join(lines)

    n = max(len(yes_levels), len(no_levels))
    wq = 12
    for i in range(n):
        if i < len(yes_levels):
            yp, yq = yes_levels[i]
            yqs = _fmt_ob_qty(yq)
            ys = f"{yp:10.4f}  {yqs:>{wq}}"
        else:
            ys = f"{'':>10}  {'':>{wq}}"
        if i < len(no_levels):
            np_, nq = no_levels[i]
            nqs = _fmt_ob_qty(nq)
            ns = f"{np_:10.4f}  {nqs:>{wq}}"
        else:
            ns = f"{'':>10}  {'':>{wq}}"
        lines.append(f"  {ys}  │  {ns}")
    return "\n".join(lines)
