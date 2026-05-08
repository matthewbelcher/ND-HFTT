from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from basket import FeeModel, BasketEvaluation, evaluate_basket, max_qty_depth, max_qty_top, simulate_buy
from client import load_client_from_env, load_trade_client_from_env
from normalize import NormalizedBook, normalize_orderbook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kalshi event-level arb analyzer (minimal)"
    )
    parser.add_argument(
        "event_ticker",
        nargs="?",
        help="Event ticker (required unless --markets is provided)",
    )
    parser.add_argument("--markets", help="Comma-separated market tickers")
    parser.add_argument("--depth", type=int, default=1, help="Orderbook depth to request")
    parser.add_argument("--qty", type=int, default=1, help="Contracts per leg")
    parser.add_argument(
        "--yes-winners",
        type=int,
        default=1,
        help="Number of outcomes that resolve YES (default 1)",
    )
    parser.add_argument(
        "--allow-crossed",
        action="store_true",
        help="Keep markets with crossed spreads (use asks only for crossed side)",
    )
    parser.add_argument(
        "--allow-ask-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow markets with asks but missing bids (default true)",
    )
    parser.add_argument(
        "--fee-mode",
        choices=["taker", "maker", "none"],
        default="taker",
        help="Fee mode for cost calculation",
    )
    parser.add_argument("--cache-seconds", type=int, default=0, help="Cache GETs for N seconds")
    parser.add_argument("--json", dest="json_path", help="Write JSON output to file")
    parser.add_argument("--verbose", action="store_true", help="Print per-market quotes")
    parser.add_argument(
        "--print-bbo",
        action="store_true",
        help="Print full BBO (YES/NO bid+ask) per market",
    )
    parser.add_argument("--breakdown", action="store_true", help="Print per-leg pricing breakdown")
    parser.add_argument(
        "--trade-sync",
        action="store_true",
        help="Include positions and resting orders from trade API (event_ticker required)",
    )
    parser.add_argument("--subaccount", type=int, default=None, help="Trade subaccount (optional)")
    parser.add_argument(
        "--fills-lookback-hours",
        type=float,
        default=None,
        help="If set, include fills from last N hours to estimate cost basis",
    )
    parser.add_argument(
        "--move-resting",
        action="store_true",
        help="Assume resting orders will be moved to current maker price for PnL calc",
    )
    parser.add_argument(
        "--print-resting",
        action="store_true",
        help="Print resting orders with current maker price and move eligibility",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit orders to trade API (default: dry run)",
    )
    parser.add_argument(
        "--confirm",
        help="Required for --execute. Use EVENT_TICKER or EXECUTE to confirm.",
    )
    parser.add_argument(
        "--exec-side",
        choices=["yes", "no", "auto"],
        default="auto",
        help="Which basket side to execute (default auto = best edge)",
    )
    parser.add_argument(
        "--liquidate-side",
        choices=["yes", "no", "auto"],
        default="auto",
        help="Which basket side to liquidate (default auto = better liquidation delta)",
    )
    parser.add_argument(
        "--exec-only-maker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only place maker orders (default true)",
    )
    parser.add_argument(
        "--exec-allow-partial",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow partial legs if some prices missing (default false)",
    )
    parser.add_argument(
        "--exec-post-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set post_only on maker orders (default true)",
    )
    parser.add_argument(
        "--exec-max-orders",
        type=int,
        default=20,
        help="Max orders to submit (default 20)",
    )
    parser.add_argument(
        "--exec-min-edge",
        type=float,
        default=0.0,
        help="Minimum edge (dollars) required to execute (default 0.0)",
    )
    parser.add_argument(
        "--exec-max-cost",
        type=float,
        default=None,
        help="Max total estimated cost (dollars) to execute",
    )
    parser.add_argument(
        "--exec-use-hybrid-edge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use hybrid edge (maker+taker plan) for exec min-edge check (default false).",
    )
    parser.add_argument(
        "--exec-move-resting",
        action="store_true",
        help="Cancel+replace resting orders to current maker price before executing",
    )
    parser.add_argument(
        "--exec-ignore-pending-taker",
        action="store_true",
        help="Ignore resting orders when building taker legs (default false)",
    )
    parser.add_argument(
        "--exec-pending-same-price",
        action="store_true",
        help="Only subtract pending qty if there is an outstanding order at the same price",
    )
    parser.add_argument(
        "--print-trade",
        action="store_true",
        help="Print the trade packet for the chosen execution side (no submit)",
    )
    parser.add_argument(
        "--exec-maker-mode",
        choices=["improve", "midpoint"],
        default="improve",
        help="Maker price mode for hybrid execution (default improve).",
    )
    parser.add_argument(
        "--exec-liquidate",
        action="store_true",
        help="Execute liquidation sells instead of basket buys",
    )
    return parser.parse_args()


def parse_markets(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [m.strip() for m in raw.split(",") if m.strip()]


def fmt_money(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    val = abs(cents) / 100.0
    return f"{sign}${val:.2f}"


def fmt_roi(roi: Optional[float]) -> str:
    if roi is None:
        return "n/a"
    return f"{roi * 100.0:.2f}%"


def summarize_book(book: NormalizedBook) -> dict:
    def lvl(val):
        if not val:
            return None
        return {"price": val[0], "qty": val[1]}

    return {
        "ticker": book.market_ticker,
        "yes_bid": lvl(book.yes_bid),
        "yes_ask": lvl(book.yes_ask),
        "no_bid": lvl(book.no_bid),
        "no_ask": lvl(book.no_ask),
    }


def _has_price(quote: Optional[Tuple[int, int]]) -> bool:
    return quote is not None and 0 < quote[0] < 100 and quote[1] > 0


def _has_depth(levels: List[Tuple[int, int]]) -> bool:
    return any(q > 0 for _, q in levels)


def _book_is_complete(
    book: NormalizedBook,
    allow_crossed: bool = False,
    allow_ask_only: bool = False,
) -> Tuple[NormalizedBook, bool, str, Optional[str]]:
    yes_ask_ok = _has_price(book.yes_ask)
    no_ask_ok = _has_price(book.no_ask)
    yes_bid_ok = _has_price(book.yes_bid)
    no_bid_ok = _has_price(book.no_bid)

    if not yes_ask_ok and not no_ask_ok:
        return book, False, "missing YES/NO ask", None

    warnings: List[str] = []
    if not yes_ask_ok:
        warnings.append("missing YES ask")
    if not no_ask_ok:
        warnings.append("missing NO ask")
    if not yes_bid_ok:
        if allow_ask_only:
            warnings.append("missing YES bid (maker disabled)")
        else:
            return book, False, "missing YES bid/ask", None
    if not no_bid_ok:
        if allow_ask_only:
            warnings.append("missing NO bid (maker disabled)")
        else:
            return book, False, "missing NO bid/ask", None
    if not _has_depth(book.yes_bid_levels):
        if allow_ask_only:
            warnings.append("missing YES bid depth")
        else:
            return book, False, "missing YES bid depth", None
    if not _has_depth(book.no_bid_levels):
        if allow_ask_only:
            warnings.append("missing NO bid depth")
        else:
            return book, False, "missing NO bid depth", None

    warning = "; ".join(warnings) if warnings else None
    if yes_ask_ok and yes_bid_ok and book.yes_ask[0] < book.yes_bid[0]:
        no_bid = book.no_bid[0] if book.no_bid else None
        reason = (
            f"crossed YES spread (yes_bid {book.yes_bid[0]}c > yes_ask {book.yes_ask[0]}c"
            f"{'' if no_bid is None else f'; no_bid {no_bid}c'}"
            ")"
        )
        if not allow_crossed and not allow_ask_only:
            return book, False, reason, None
        warning = f"{reason} [kept: YES bid ignored]"
        book = NormalizedBook(
            market_ticker=book.market_ticker,
            yes_bid=None,
            yes_ask=book.yes_ask,
            no_bid=book.no_bid,
            no_ask=book.no_ask,
            yes_bid_levels=[],
            no_bid_levels=book.no_bid_levels,
            yes_ask_levels=book.yes_ask_levels,
            no_ask_levels=book.no_ask_levels,
        )

    if no_ask_ok and no_bid_ok and book.no_ask[0] < book.no_bid[0]:
        yes_bid = book.yes_bid[0] if book.yes_bid else None
        reason = (
            f"crossed NO spread (no_bid {book.no_bid[0]}c > no_ask {book.no_ask[0]}c"
            f"{'' if yes_bid is None else f'; yes_bid {yes_bid}c'}"
            ")"
        )
        if not allow_crossed and not allow_ask_only:
            return book, False, reason, None
        warning = f"{reason} [kept: NO bid ignored]"
        book = NormalizedBook(
            market_ticker=book.market_ticker,
            yes_bid=book.yes_bid,
            yes_ask=book.yes_ask,
            no_bid=None,
            no_ask=book.no_ask,
            yes_bid_levels=book.yes_bid_levels,
            no_bid_levels=[],
            yes_ask_levels=book.yes_ask_levels,
            no_ask_levels=book.no_ask_levels,
        )

    if warning and warnings:
        warning = f"{warning}; {', '.join(warnings)}"
    return book, True, "", warning


def _side_bid(book: NormalizedBook, side: str) -> Optional[Tuple[int, int]]:
    quote = book.yes_bid if side == "yes" else book.no_bid
    return quote if _has_price(quote) else None


def _side_ask(book: NormalizedBook, side: str) -> Optional[Tuple[int, int]]:
    quote = book.yes_ask if side == "yes" else book.no_ask
    return quote if _has_price(quote) else None


def _side_bid_levels(book: NormalizedBook, side: str) -> List[Tuple[int, int]]:
    return book.yes_bid_levels if side == "yes" else book.no_bid_levels


def _side_ask_levels(book: NormalizedBook, side: str) -> List[Tuple[int, int]]:
    return book.yes_ask_levels if side == "yes" else book.no_ask_levels


def _filter_books_with_ask(
    books: List[NormalizedBook],
    side: str,
) -> Tuple[List[NormalizedBook], List[str]]:
    kept: List[NormalizedBook] = []
    dropped: List[str] = []
    for book in books:
        if _side_ask(book, side) is None:
            dropped.append(str(book.market_ticker))
            continue
        kept.append(book)
    return kept, dropped


def _limit_price_maker_bid(book: NormalizedBook, side: str) -> Optional[int]:
    bid = _side_bid(book, side)
    return bid[0] if bid else None


def _limit_price_improve_1(book: NormalizedBook, side: str) -> Optional[int]:
    bid = _side_bid(book, side)
    ask = _side_ask(book, side)
    if not bid or not ask:
        return None
    price = bid[0] + 1
    if price >= ask[0]:
        return None
    return price


def _limit_price_mid(book: NormalizedBook, side: str) -> Optional[int]:
    bid = _side_bid(book, side)
    ask = _side_ask(book, side)
    if not bid or not ask:
        return None
    if ask[0] - bid[0] < 2:
        return None
    price = (ask[0] + bid[0]) // 2
    if price <= bid[0] or price >= ask[0]:
        return None
    return price


def _limit_cost(prices: List[Optional[int]], qty: int, fee_model: FeeModel) -> Optional[int]:
    if qty <= 0:
        return None
    if any(p is None for p in prices):
        return None
    cost = 0
    for price in prices:
        cost += fee_model.buy_cost_cents(int(price), qty)
    return cost


def _queue_ahead(levels: List[Tuple[int, int]], price: int) -> int:
    for p, q in levels:
        if p == price:
            return q
    return 0


def _top_bid_maps(
    books: List[NormalizedBook],
    side: str,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    bid_px: Dict[str, int] = {}
    bid_qty: Dict[str, int] = {}
    for book in books:
        bid = _side_bid(book, side)
        if not bid:
            continue
        bid_px[str(book.market_ticker)] = int(bid[0])
        bid_qty[str(book.market_ticker)] = int(bid[1])
    return bid_px, bid_qty


def _should_move_resting(
    *,
    cur_px: Optional[int],
    maker_px: Optional[int],
    top_bid_px: Optional[int],
    top_bid_qty: Optional[int],
    own_qty_at_price: int,
) -> bool:
    if cur_px is None or maker_px is None:
        return False
    if cur_px >= maker_px:
        return False
    if top_bid_px is not None and cur_px == top_bid_px:
        if top_bid_qty is not None and own_qty_at_price >= top_bid_qty:
            return False
    return True


def _unique_sizes(items: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    seen = set()
    out: List[Tuple[str, int]] = []
    for label, size in items:
        if size <= 0:
            continue
        if size in seen:
            continue
        seen.add(size)
        out.append((label, size))
    return out


def _spread_stats(books: List[NormalizedBook], side: str) -> Tuple[Optional[int], Optional[int]]:
    spreads: List[int] = []
    for book in books:
        bid = _side_bid(book, side)
        ask = _side_ask(book, side)
        if not bid or not ask:
            continue
        spreads.append(ask[0] - bid[0])
    if not spreads:
        return None, None
    return min(spreads), max(spreads)


def _print_strategy_table(
    side: str,
    rows: List[Dict[str, object]],
    *,
    min_spread: Optional[int],
    max_spread: Optional[int],
) -> None:
    spread_desc = "n/a"
    if min_spread is not None and max_spread is not None:
        spread_desc = f"{min_spread}c..{max_spread}c"
    print(f"Limit strategies for {side.upper()} basket (spreads {spread_desc})")
    header = (
        f"{'Strategy':<10} | {'Size':>5} | {'Target':>6} | "
        f"{'L-Cost':>8} | {'L-Pay':>8} | {'L-Edge':>8} | {'L-ROI':>7} | "
        f"{'W-Cost':>8} | {'W-Pay':>8} | {'W-Edge':>8} | {'W-ROI':>7} | "
        f"{'Fill':>4} | {'Queue':>5}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        limit_cost = row.get("limit_cost")
        limit_payout = row.get("limit_payout")
        limit_edge = row.get("limit_edge")
        limit_roi = row.get("limit_roi")
        worst_cost = row.get("worst_cost")
        worst_payout = row.get("worst_payout")
        worst_edge = row.get("worst_edge")
        worst_roi = row.get("worst_roi")
        taker_fill = row.get("taker_fill")
        max_queue = row.get("max_queue")
        print(
            f"{row['strategy']:<10} | {row['size_label']:>5} | {row['target_qty']:>6} | "
            f"{fmt_money(limit_cost) if limit_cost is not None else 'n/a':>8} | "
            f"{fmt_money(limit_payout) if limit_payout is not None else 'n/a':>8} | "
            f"{fmt_money(limit_edge) if limit_edge is not None else 'n/a':>8} | "
            f"{fmt_roi(limit_roi) if limit_roi is not None else 'n/a':>7} | "
            f"{fmt_money(worst_cost) if worst_cost is not None else 'n/a':>8} | "
            f"{fmt_money(worst_payout) if worst_payout is not None else 'n/a':>8} | "
            f"{fmt_money(worst_edge) if worst_edge is not None else 'n/a':>8} | "
            f"{fmt_roi(worst_roi) if worst_roi is not None else 'n/a':>7} | "
            f"{taker_fill if taker_fill is not None else 'n/a':>4} | "
            f"{max_queue if max_queue is not None else 'n/a':>5}"
        )
    print("")


def _maker_price_for_mode(
    bid_price: Optional[int],
    ask_price: Optional[int],
    mode: str,
) -> Optional[int]:
    if bid_price is None:
        return None
    if mode == "midpoint":
        if ask_price is None:
            return None
        if ask_price - bid_price < 2:
            return None
        price = (ask_price + bid_price) // 2
        if price <= bid_price or price >= ask_price:
            return None
        return price
    # default: improve
    price = bid_price
    if ask_price is not None and ask_price - bid_price >= 2:
        price = min(bid_price + 1, ask_price - 1)
    return price


def _hybrid_plan_rows(
    books: List[NormalizedBook],
    side: str,
    qty: int,
    maker_model: FeeModel,
    taker_model: FeeModel,
    maker_mode: str = "improve",
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for book in books:
        bid = _side_bid(book, side)
        ask = _side_ask(book, side)
        bid_price = bid[0] if bid else None
        ask_price = ask[0] if ask else None
        spread = (ask_price - bid_price) if ask_price is not None and bid_price is not None else None
        maker_price = _maker_price_for_mode(bid_price, ask_price, maker_mode)
        queue = (
            _queue_ahead(_side_bid_levels(book, side), maker_price)
            if maker_price is not None
            else None
        )
        maker_fee = maker_model.fee_cents(maker_price, qty) if maker_price is not None else None
        taker_fee = taker_model.fee_cents(ask_price, qty) if ask_price is not None else None
        maker_cost = (
            maker_model.buy_cost_cents(maker_price, qty) if maker_price is not None else None
        )
        taker_cost = (
            taker_model.buy_cost_cents(ask_price, qty) if ask_price is not None else None
        )
        savings = (
            taker_cost - maker_cost
            if taker_cost is not None and maker_cost is not None
            else None
        )

        if ask_price is None:
            rec = "n/a"
            reason = "no taker ask"
        elif bid_price is None:
            rec = "taker"
            reason = "no maker bid"
        elif maker_price is None:
            rec = "taker"
            reason = "no midpoint" if maker_mode == "midpoint" else "no maker price"
        elif spread is not None and spread <= 1:
            rec = "taker"
            reason = "tight spread"
        elif queue is not None and queue > qty * 50:
            rec = "taker"
            reason = "large queue"
        else:
            rec = "maker"
            if maker_mode == "midpoint":
                reason = "midpoint"
            else:
                reason = "improve" if maker_price is not None and maker_price > bid_price else "rest"

        rows.append(
            {
                "ticker": book.market_ticker,
                "bid": bid_price,
                "ask": ask_price,
                "maker_px": maker_price,
                "maker_mode": maker_mode,
                "spread": spread,
                "queue": queue,
                "maker_fee": maker_fee,
                "taker_fee": taker_fee,
                "savings": savings,
                "rec": rec,
                "reason": reason,
            }
        )
    return rows


def _print_hybrid_plan(side: str, rows: List[Dict[str, object]], qty: int) -> None:
    print(f"Hybrid execution plan for {side.upper()} basket (qty={qty})")
    header = (
        f"{'Ticker':<24} | {'Bid':>3} | {'Ask':>3} | {'MkPx':>4} | {'Spr':>3} | {'Queue':>6} | "
        f"{'MakerFee':>8} | {'TakerFee':>8} | {'Save':>8} | {'Rec':>5} | Reason"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        bid = row.get("bid")
        ask = row.get("ask")
        maker_px = row.get("maker_px")
        spr = row.get("spread")
        queue = row.get("queue")
        maker_fee = row.get("maker_fee")
        taker_fee = row.get("taker_fee")
        savings = row.get("savings")
        print(
            f"{row['ticker']:<24} | "
            f"{bid if bid is not None else 'n/a':>3} | "
            f"{ask if ask is not None else 'n/a':>3} | "
            f"{maker_px if maker_px is not None else 'n/a':>4} | "
            f"{spr if spr is not None else 'n/a':>3} | "
            f"{queue if queue is not None else 'n/a':>6} | "
            f"{fmt_money(maker_fee) if maker_fee is not None else 'n/a':>8} | "
            f"{fmt_money(taker_fee) if taker_fee is not None else 'n/a':>8} | "
            f"{fmt_money(savings) if savings is not None else 'n/a':>8} | "
            f"{row['rec']:<5} | {row['reason']}"
        )
    print("")


def _hybrid_summary(
    side: str,
    rows: List[Dict[str, object]],
    qty: int,
    payout_per_contract: int,
    maker_model: FeeModel,
    taker_model: FeeModel,
) -> Dict[str, object]:
    maker_cost = 0
    taker_cost = 0
    maker_fees = 0
    taker_fees = 0
    maker_legs = 0
    taker_legs = 0
    missing: List[str] = []
    for row in rows:
        ticker = row.get("ticker")
        bid = row.get("bid")
        ask = row.get("ask")
        rec = row.get("rec")
        if rec == "maker":
            maker_px = row.get("maker_px")
            use_px = maker_px if maker_px is not None else bid
            if use_px is None:
                missing.append(str(ticker))
                continue
            fee = maker_model.fee_cents(int(use_px), qty)
            cost = maker_model.buy_cost_cents(int(use_px), qty)
            maker_fees += fee
            maker_cost += cost
            maker_legs += 1
        elif rec == "taker":
            if ask is None:
                missing.append(str(ticker))
                continue
            fee = taker_model.fee_cents(int(ask), qty)
            cost = taker_model.buy_cost_cents(int(ask), qty)
            taker_fees += fee
            taker_cost += cost
            taker_legs += 1
        else:
            missing.append(str(ticker))

    legs = len(rows)
    filled_legs = maker_legs + taker_legs
    total_cost = maker_cost + taker_cost if filled_legs > 0 else None
    payout = payout_per_contract * qty if filled_legs == legs else None
    edge = (payout - total_cost) if payout is not None and total_cost is not None else None
    roi = (edge / total_cost) if edge is not None and total_cost else None
    return {
        "side": side,
        "qty": qty,
        "legs": legs,
        "filled_legs": filled_legs,
        "maker_legs": maker_legs,
        "taker_legs": taker_legs,
        "maker_cost": maker_cost if maker_legs > 0 else None,
        "taker_cost": taker_cost if taker_legs > 0 else None,
        "maker_fees": maker_fees if maker_legs > 0 else None,
        "taker_fees": taker_fees if taker_legs > 0 else None,
        "total_cost": total_cost,
        "payout": payout,
        "edge": edge,
        "roi": roi,
        "missing": missing,
    }


def _print_hybrid_summary(summary: Dict[str, object]) -> None:
    side = summary.get("side")
    qty = summary.get("qty")
    legs = summary.get("legs")
    filled_legs = summary.get("filled_legs")
    print(f"Hybrid summary for {str(side).upper()} basket (qty={qty})")
    missing = summary.get("missing") or []
    if filled_legs != legs:
        print(f"Incomplete pricing for {filled_legs}/{legs} legs.")
        if missing:
            sample = ", ".join(str(m) for m in list(missing)[:5])
            tail = "..." if len(missing) > 5 else ""
            print(f"Missing legs: {sample}{tail}")

    total_cost = summary.get("total_cost")
    payout = summary.get("payout")
    edge = summary.get("edge")
    roi = summary.get("roi")
    maker_cost = summary.get("maker_cost")
    taker_cost = summary.get("taker_cost")
    maker_fees = summary.get("maker_fees")
    taker_fees = summary.get("taker_fees")
    maker_legs = summary.get("maker_legs")
    taker_legs = summary.get("taker_legs")

    print(
        f"Total: cost {fmt_money(int(total_cost)) if total_cost is not None else 'n/a'}, "
        f"payout {fmt_money(int(payout)) if payout is not None else 'n/a'}, "
        f"edge {fmt_money(int(edge)) if edge is not None else 'n/a'}, "
        f"roi {fmt_roi(roi) if roi is not None else 'n/a'}"
    )
    print(
        f"Maker: legs {maker_legs}, cost {fmt_money(int(maker_cost)) if maker_cost is not None else 'n/a'}, "
        f"fees {fmt_money(int(maker_fees)) if maker_fees is not None else 'n/a'}"
    )
    print(
        f"Taker: legs {taker_legs}, cost {fmt_money(int(taker_cost)) if taker_cost is not None else 'n/a'}, "
        f"fees {fmt_money(int(taker_fees)) if taker_fees is not None else 'n/a'}"
    )
    print("")


def _resting_move_rows(
    orders: List[dict],
    yes_maker_px: Dict[str, int],
    no_maker_px: Dict[str, int],
    yes_bid_px: Dict[str, int],
    yes_bid_qty: Dict[str, int],
    no_bid_px: Dict[str, int],
    no_bid_qty: Dict[str, int],
    own_qty_by_price: Dict[Tuple[str, str, int], int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for o in orders:
        if str(o.get("action") or "").lower() != "buy":
            continue
        side = str(o.get("side") or "").lower()
        if side not in ("yes", "no"):
            continue
        ticker = o.get("ticker")
        if not ticker:
            continue
        remaining = o.get("remaining_count")
        if remaining is None:
            remaining = o.get("remaining_count_fp") or o.get("initial_count") or 0
        try:
            remaining_int = int(float(remaining))
        except (TypeError, ValueError):
            remaining_int = 0
        if remaining_int <= 0:
            continue
        cur_px = _order_price_cents(o, side)
        maker_px = yes_maker_px.get(str(ticker)) if side == "yes" else no_maker_px.get(str(ticker))
        top_bid_px = yes_bid_px.get(str(ticker)) if side == "yes" else no_bid_px.get(str(ticker))
        top_bid_qty = yes_bid_qty.get(str(ticker)) if side == "yes" else no_bid_qty.get(str(ticker))
        own_qty = own_qty_by_price.get((str(ticker), side, int(cur_px))) if cur_px is not None else 0
        own_qty = own_qty or 0
        move = _should_move_resting(
            cur_px=cur_px,
            maker_px=maker_px,
            top_bid_px=top_bid_px,
            top_bid_qty=top_bid_qty,
            own_qty_at_price=own_qty,
        )
        rows.append(
            {
                "ticker": str(ticker),
                "side": side,
                "qty": remaining_int,
                "price": cur_px,
                "maker_px": maker_px,
                "top_qty": top_bid_qty,
                "own_qty": own_qty,
                "move": move,
                "delta": (maker_px - cur_px) if cur_px is not None and maker_px is not None else None,
            }
        )
    return rows


def _print_resting_moves(rows: List[Dict[str, object]]) -> None:
    print("Resting orders vs maker price")
    header = (
        f"{'Ticker':<24} | {'Side':>4} | {'Qty':>3} | {'Px':>3} | "
        f"{'MkPx':>4} | {'Top':>4} | {'Own':>4} | {'Delta':>5} | Move"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        delta = row.get("delta")
        print(
            f"{row['ticker']:<24} | "
            f"{row['side']:>4} | "
            f"{row['qty']:>3} | "
            f"{row['price'] if row['price'] is not None else 'n/a':>3} | "
            f"{row['maker_px'] if row['maker_px'] is not None else 'n/a':>4} | "
            f"{row.get('top_qty') if row.get('top_qty') is not None else 'n/a':>4} | "
            f"{row.get('own_qty') if row.get('own_qty') is not None else 'n/a':>4} | "
            f"{delta if delta is not None else 'n/a':>5} | "
            f"{'yes' if row['move'] else 'no'}"
        )
    print("")


def _simulate_buy_with_fee(
    levels: List[Tuple[int, int]],
    qty: int,
    fee_model: FeeModel,
) -> Tuple[int, int, int]:
    remaining = qty
    gross = 0
    fee = 0
    filled = 0
    for price, level_qty in levels:
        if remaining <= 0:
            break
        take = min(remaining, level_qty)
        if take <= 0:
            continue
        gross += price * take
        fee += fee_model.fee_cents(price, take)
        filled += take
        remaining -= take
    return gross + fee, filled, fee


def _print_leg_breakdown(
    side: str,
    books: List[NormalizedBook],
    qty: int,
    fee_model: FeeModel,
) -> None:
    print(f"{side.upper()} basket leg breakdown (taker, qty={qty})")
    header = f"{'Ticker':<24} | {'TopPx':>5} | {'TopQty':>6} | {'Fill':>4} | {'Cost':>8} | {'Fee':>6}"
    print(header)
    print("-" * len(header))
    for book in books:
        levels = book.yes_ask_levels if side == "yes" else book.no_ask_levels
        top_px = levels[0][0] if levels else None
        top_qty = levels[0][1] if levels else None
        cost, filled, fee = _simulate_buy_with_fee(levels, qty, fee_model)
        print(
            f"{book.market_ticker:<24} | "
            f"{top_px if top_px is not None else 'n/a':>5} | "
            f"{top_qty if top_qty is not None else 'n/a':>6} | "
            f"{filled:>4} | {fmt_money(cost):>8} | {fmt_money(fee):>6}"
        )
    print("")


def _print_taker_arb_summary(
    side: str,
    books: List[NormalizedBook],
    qty: int,
    payout_per_contract: int,
    fee_model: FeeModel,
) -> None:
    levels = [b.yes_ask_levels if side == "yes" else b.no_ask_levels for b in books]
    fill_qty = min(qty, max_qty_depth(levels))
    if fill_qty <= 0:
        print(f"Taker arb summary for {side.upper()} basket: no depth.")
        print("")
        return
    print(f"Taker arb summary for {side.upper()} basket (qty={fill_qty})")
    header = (
        f"{'Ticker':<24} | {'Ask':>3} | {'Fill':>4} | {'Cost':>8} | {'Fee':>6}"
    )
    print(header)
    print("-" * len(header))
    total_cost = 0
    total_fee = 0
    for book, leg_levels in zip(books, levels):
        top_px = leg_levels[0][0] if leg_levels else None
        cost, filled, fee = _simulate_buy_with_fee(leg_levels, fill_qty, fee_model)
        total_cost += cost
        total_fee += fee
        print(
            f"{book.market_ticker:<24} | "
            f"{top_px if top_px is not None else 'n/a':>3} | "
            f"{filled:>4} | {fmt_money(cost):>8} | {fmt_money(fee):>6}"
        )
    payout = payout_per_contract * fill_qty
    edge = payout - total_cost
    roi = (edge / total_cost) if total_cost > 0 else None
    print(
        f"Total: cost {fmt_money(total_cost)}, payout {fmt_money(payout)}, "
        f"edge {fmt_money(edge)}, roi {fmt_roi(roi) if roi is not None else 'n/a'}"
    )
    print("")


def _price_to_cents(value: Optional[object]) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if abs(f - round(f)) < 1e-9:
        return int(round(f))
    if f <= 1.0:
        return int(round(f * 100.0))
    return int(round(f))


def _order_price_cents(order: dict, side: str) -> Optional[int]:
    side = side.lower()
    if side == "yes":
        for k in ("yes_price", "yes_price_fixed", "yes_price_dollars"):
            if k in order:
                return _price_to_cents(order.get(k))
    if side == "no":
        for k in ("no_price", "no_price_fixed", "no_price_dollars"):
            if k in order:
                return _price_to_cents(order.get(k))
    if "price" in order:
        px = _price_to_cents(order.get("price"))
        if px is not None and side == "no":
            return 100 - px
        return px
    return None


def _to_int_safe(value: object) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _position_qty(p: dict) -> Optional[int]:
    if not isinstance(p, dict):
        return None
    for key in (
        "position",
        "position_fp",
        "net_position",
        "net_position_fp",
        "position_count",
        "position_qty",
    ):
        if key in p:
            val = _to_int_safe(p.get(key))
            if val is not None:
                return val
    pos_obj = p.get("position")
    if isinstance(pos_obj, dict):
        yes_val = _to_int_safe(pos_obj.get("yes"))
        no_val = _to_int_safe(pos_obj.get("no"))
        if yes_val is not None or no_val is not None:
            return (yes_val or 0) - (no_val or 0)
    yes_val = _to_int_safe(p.get("yes_position"))
    no_val = _to_int_safe(p.get("no_position"))
    if yes_val is not None or no_val is not None:
        return (yes_val or 0) - (no_val or 0)
    side = str(p.get("side") or "").lower()
    if side in ("yes", "no"):
        qty = _to_int_safe(p.get("quantity") or p.get("count"))
        if qty is not None:
            return qty if side == "yes" else -qty
    return None


def _build_trade_state(positions: List[dict], orders: List[dict]) -> Dict[str, object]:
    pos_yes: Dict[str, int] = {}
    for p in positions:
        ticker = p.get("ticker") or p.get("market_ticker")
        if not ticker:
            continue
        qty = _position_qty(p)
        if qty is None:
            continue
        pos_yes[str(ticker)] = qty

    pending_yes_qty: Dict[str, int] = {}
    pending_no_qty: Dict[str, int] = {}
    pending_yes_cost: Dict[str, int] = {}
    pending_no_cost: Dict[str, int] = {}
    for o in orders:
        if str(o.get("action") or "").lower() != "buy":
            continue
        side = str(o.get("side") or "").lower()
        if side not in ("yes", "no"):
            continue
        ticker = o.get("ticker")
        if not ticker:
            continue
        remaining = o.get("remaining_count")
        if remaining is None:
            remaining = o.get("remaining_count_fp") or o.get("initial_count") or 0
        try:
            remaining_int = int(float(remaining))
        except (TypeError, ValueError):
            remaining_int = 0
        if remaining_int <= 0:
            continue
        px = _order_price_cents(o, side)
        if px is None:
            continue
        if side == "yes":
            pending_yes_qty[ticker] = pending_yes_qty.get(ticker, 0) + remaining_int
            pending_yes_cost[ticker] = pending_yes_cost.get(ticker, 0) + (px * remaining_int)
        else:
            pending_no_qty[ticker] = pending_no_qty.get(ticker, 0) + remaining_int
            pending_no_cost[ticker] = pending_no_cost.get(ticker, 0) + (px * remaining_int)

    pos_yes_total = sum(v for v in pos_yes.values() if v > 0)
    pos_no_total = sum(-v for v in pos_yes.values() if v < 0)
    resting_yes_qty_total = sum(pending_yes_qty.values())
    resting_no_qty_total = sum(pending_no_qty.values())

    return {
        "pending_orders": orders,
        "pos_yes": pos_yes,
        "pos_yes_total": pos_yes_total,
        "pos_no_total": pos_no_total,
        "pending_yes_qty": pending_yes_qty,
        "pending_no_qty": pending_no_qty,
        "pending_yes_cost": pending_yes_cost,
        "pending_no_cost": pending_no_cost,
        "resting_yes_buy_qty_total": resting_yes_qty_total,
        "resting_no_buy_qty_total": resting_no_qty_total,
    }


def _build_fills_state(fills: List[dict]) -> Dict[str, Dict[str, int]]:
    fills_yes_qty: Dict[str, int] = {}
    fills_yes_cost: Dict[str, int] = {}
    fills_no_qty: Dict[str, int] = {}
    fills_no_cost: Dict[str, int] = {}
    for f in fills:
        if f.get("action") != "buy":
            continue
        side = f.get("side")
        if side not in ("yes", "no"):
            continue
        ticker = f.get("ticker")
        if not ticker:
            continue
        count = f.get("count") or f.get("count_fp") or 0
        try:
            count_int = int(float(count))
        except (TypeError, ValueError):
            count_int = 0
        if count_int <= 0:
            continue
        px = _order_price_cents(f, side)
        if px is None:
            continue
        fee = f.get("fee_cost") or 0
        try:
            fee_cents = int(round(float(fee) * 100.0))
        except (TypeError, ValueError):
            fee_cents = 0
        cost = px * count_int + fee_cents
        if side == "yes":
            fills_yes_qty[ticker] = fills_yes_qty.get(ticker, 0) + count_int
            fills_yes_cost[ticker] = fills_yes_cost.get(ticker, 0) + cost
        else:
            fills_no_qty[ticker] = fills_no_qty.get(ticker, 0) + count_int
            fills_no_cost[ticker] = fills_no_cost.get(ticker, 0) + cost
    return {
        "fills_yes_qty": fills_yes_qty,
        "fills_yes_cost": fills_yes_cost,
        "fills_no_qty": fills_no_qty,
        "fills_no_cost": fills_no_cost,
    }


def _pending_qty_by_ticker_side(orders: List[dict]) -> Dict[Tuple[str, str], int]:
    out: Dict[Tuple[str, str], int] = {}
    for o in orders:
        if str(o.get("action") or "").lower() != "buy":
            continue
        side = str(o.get("side") or "").lower()
        if side not in ("yes", "no"):
            continue
        ticker = o.get("ticker")
        if not ticker:
            continue
        remaining = o.get("remaining_count")
        if remaining is None:
            remaining = o.get("remaining_count_fp") or o.get("initial_count") or 0
        try:
            remaining_int = int(float(remaining))
        except (TypeError, ValueError):
            remaining_int = 0
        if remaining_int <= 0:
            continue
        key = (str(ticker), str(side))
        out[key] = out.get(key, 0) + remaining_int
    return out


def _pending_qty_by_ticker_side_price(orders: List[dict]) -> Dict[Tuple[str, str, int], int]:
    out: Dict[Tuple[str, str, int], int] = {}
    for o in orders:
        if str(o.get("action") or "").lower() != "buy":
            continue
        side = str(o.get("side") or "").lower()
        if side not in ("yes", "no"):
            continue
        ticker = o.get("ticker")
        if not ticker:
            continue
        remaining = o.get("remaining_count")
        if remaining is None:
            remaining = o.get("remaining_count_fp") or o.get("initial_count") or 0
        try:
            remaining_int = int(float(remaining))
        except (TypeError, ValueError):
            remaining_int = 0
        if remaining_int <= 0:
            continue
        px = _order_price_cents(o, side)
        if px is None:
            continue
        key = (str(ticker), str(side), int(px))
        out[key] = out.get(key, 0) + remaining_int
    return out


def _pending_cost_by_ticker(
    orders: List[dict],
    side: str,
    maker_px_by_ticker: Optional[Dict[str, int]] = None,
    move_resting: bool = False,
    maker_model: Optional[FeeModel] = None,
    top_bid_px_by_ticker: Optional[Dict[str, int]] = None,
    top_bid_qty_by_ticker: Optional[Dict[str, int]] = None,
    own_qty_by_price: Optional[Dict[Tuple[str, str, int], int]] = None,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], int, int]:
    cost_map: Dict[str, int] = {}
    qty_map: Dict[str, int] = {}
    fee_map: Dict[str, int] = {}
    moved_qty = 0
    moved_orders = 0
    for o in orders:
        if str(o.get("action") or "").lower() != "buy":
            continue
        o_side = str(o.get("side") or "").lower()
        if o_side != side:
            continue
        ticker = o.get("ticker")
        if not ticker:
            continue
        remaining = o.get("remaining_count")
        if remaining is None:
            remaining = o.get("remaining_count_fp") or o.get("initial_count") or 0
        try:
            remaining_int = int(float(remaining))
        except (TypeError, ValueError):
            remaining_int = 0
        if remaining_int <= 0:
            continue
        cur_px = _order_price_cents(o, side)
        if cur_px is None:
            continue
        use_px = cur_px
        maker_px = None
        if move_resting and maker_px_by_ticker is not None:
            maker_px = maker_px_by_ticker.get(str(ticker))
        if move_resting and maker_px is not None:
            top_bid_px = top_bid_px_by_ticker.get(str(ticker)) if top_bid_px_by_ticker else None
            top_bid_qty = top_bid_qty_by_ticker.get(str(ticker)) if top_bid_qty_by_ticker else None
            own_qty = 0
            if own_qty_by_price is not None:
                own_qty = own_qty_by_price.get((str(ticker), str(side), int(cur_px)), 0)
            if _should_move_resting(
                cur_px=cur_px,
                maker_px=maker_px,
                top_bid_px=top_bid_px,
                top_bid_qty=top_bid_qty,
                own_qty_at_price=own_qty,
            ):
                use_px = maker_px
                moved_qty += remaining_int
                moved_orders += 1
        key = str(ticker)
        cost_map[key] = cost_map.get(key, 0) + use_px * remaining_int
        qty_map[key] = qty_map.get(key, 0) + remaining_int
        if maker_model is not None:
            fee_map[key] = fee_map.get(key, 0) + maker_model.fee_cents(use_px, remaining_int)
    return cost_map, qty_map, fee_map, moved_qty, moved_orders


def _pending_move_cost(
    orders: List[dict],
    side: str,
    maker_px_by_ticker: Dict[str, int],
    maker_model: FeeModel,
) -> Tuple[int, int, int, int]:
    cost_map, _, fee_map, moved_qty, moved_orders = _pending_cost_by_ticker(
        orders,
        side,
        maker_px_by_ticker=maker_px_by_ticker,
        move_resting=True,
        maker_model=maker_model,
    )
    return sum(cost_map.values()), sum(fee_map.values()), moved_qty, moved_orders


def _print_trade_state_summary(state: Dict[str, object]) -> None:
    print("Trade sync: positions + resting orders")
    print(
        "Totals: "
        f"YES pos {state.get('pos_yes_total', 0)}, "
        f"NO pos {state.get('pos_no_total', 0)}, "
        f"resting YES qty {state.get('resting_yes_buy_qty_total', 0)}, "
        f"resting NO qty {state.get('resting_no_buy_qty_total', 0)}"
    )
    if state.get("fills_lookback_hours") is not None:
        print(f"Fills lookback (hours): {state.get('fills_lookback_hours')}")
    print("")


def _adjusted_basket(
    side: str,
    books: List[NormalizedBook],
    target_qty: int,
    trade_state: Dict[str, object],
    taker_model: FeeModel,
    maker_model: FeeModel,
    payout_per_contract: int,
    move_resting: bool,
    maker_px_by_ticker: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    pos_yes = trade_state.get("pos_yes", {})
    pending_yes_qty = trade_state.get("pending_yes_qty", {})
    pending_no_qty = trade_state.get("pending_no_qty", {})
    pending_yes_cost = trade_state.get("pending_yes_cost", {})
    pending_no_cost = trade_state.get("pending_no_cost", {})
    fills_yes_qty = trade_state.get("fills_yes_qty", {})
    fills_yes_cost = trade_state.get("fills_yes_cost", {})
    fills_no_qty = trade_state.get("fills_no_qty", {})
    fills_no_cost = trade_state.get("fills_no_cost", {})
    pending_orders = trade_state.get("pending_orders", [])

    top_bid_px_by_ticker: Dict[str, int] = {}
    top_bid_qty_by_ticker: Dict[str, int] = {}
    own_qty_by_price: Dict[Tuple[str, str, int], int] = {}
    if move_resting and maker_px_by_ticker is not None:
        top_bid_px_by_ticker, top_bid_qty_by_ticker = _top_bid_maps(books, side)
        if isinstance(pending_orders, list):
            own_qty_by_price = _pending_qty_by_ticker_side_price(pending_orders)

    remaining_cost = 0
    remaining_cost_no_pending = 0
    pending_cost = 0
    pending_fee_total = 0
    pending_cost_used = None
    pending_fee_used = None
    pending_cost_map: Optional[Dict[str, int]] = None
    pending_qty_map: Optional[Dict[str, int]] = None
    pending_fee_map: Optional[Dict[str, int]] = None
    pending_moved_qty = 0
    pending_moved_orders = 0
    remaining_total = 0
    short_legs = 0
    short_legs_no_pending = 0
    spent_cost = 0
    spent_missing = 0
    resting_fill_cost = 0
    resting_fee_total = 0
    resting_missing = 0
    remaining_min = None
    remaining_max = None

    if move_resting and maker_px_by_ticker is not None:
        pending_cost_map, pending_qty_map, pending_fee_map, pending_moved_qty, pending_moved_orders = (
            _pending_cost_by_ticker(
                pending_orders if isinstance(pending_orders, list) else [],
                side,
                maker_px_by_ticker=maker_px_by_ticker,
                move_resting=True,
                maker_model=maker_model,
                top_bid_px_by_ticker=top_bid_px_by_ticker,
                top_bid_qty_by_ticker=top_bid_qty_by_ticker,
                own_qty_by_price=own_qty_by_price,
            )
        )

    for book in books:
        ticker = book.market_ticker
        pos = int(pos_yes.get(ticker, 0)) if isinstance(pos_yes, dict) else 0
        pos_qty = max(0, pos) if side == "yes" else max(0, -pos)
        if side == "yes":
            pending_qty = int(pending_yes_qty.get(ticker, 0)) if isinstance(pending_yes_qty, dict) else 0
            pending_cost_leg = (
                int(pending_yes_cost.get(ticker, 0)) if isinstance(pending_yes_cost, dict) else 0
            )
        else:
            pending_qty = int(pending_no_qty.get(ticker, 0)) if isinstance(pending_no_qty, dict) else 0
            pending_cost_leg = (
                int(pending_no_cost.get(ticker, 0)) if isinstance(pending_no_cost, dict) else 0
            )
        pending_cost += pending_cost_leg
        if pending_qty > 0 and pending_cost_leg > 0:
            avg_pending_px = pending_cost_leg / float(pending_qty)
            pending_fee_total += maker_model.fee_cents(int(round(avg_pending_px)), pending_qty)
        remaining = max(0, target_qty - pos_qty - pending_qty)
        remaining_no_pending = max(0, target_qty - pos_qty)
        remaining_total += remaining
        remaining_min = remaining if remaining_min is None else min(remaining_min, remaining)
        remaining_max = remaining if remaining_max is None else max(remaining_max, remaining)

        if pos_qty > 0:
            if side == "yes":
                fq = int(fills_yes_qty.get(ticker, 0)) if isinstance(fills_yes_qty, dict) else 0
                fc = int(fills_yes_cost.get(ticker, 0)) if isinstance(fills_yes_cost, dict) else 0
            else:
                fq = int(fills_no_qty.get(ticker, 0)) if isinstance(fills_no_qty, dict) else 0
                fc = int(fills_no_cost.get(ticker, 0)) if isinstance(fills_no_cost, dict) else 0
            if fq > 0:
                avg = fc / float(fq)
                spent_cost += int(round(avg * pos_qty))
            else:
                spent_missing += 1

        covered = min(target_qty, pos_qty + pending_qty)
        pos_used = min(pos_qty, covered)
        pending_used = max(0, covered - pos_used)
        if covered == target_qty:
            if side == "yes":
                fq = int(fills_yes_qty.get(ticker, 0)) if isinstance(fills_yes_qty, dict) else 0
                fc = int(fills_yes_cost.get(ticker, 0)) if isinstance(fills_yes_cost, dict) else 0
            else:
                fq = int(fills_no_qty.get(ticker, 0)) if isinstance(fills_no_qty, dict) else 0
                fc = int(fills_no_cost.get(ticker, 0)) if isinstance(fills_no_cost, dict) else 0
            if pos_used > 0 and fq > 0:
                avg_fill = fc / float(fq)
                resting_fill_cost += int(round(avg_fill * pos_used))
            elif pos_used > 0:
                resting_missing += 1
            if pending_used > 0:
                if move_resting and pending_cost_map is not None and pending_qty_map is not None:
                    pend_qty = pending_qty_map.get(ticker, pending_qty)
                    pend_cost_total = pending_cost_map.get(ticker, pending_cost_leg)
                else:
                    pend_qty = pending_qty
                    pend_cost_total = pending_cost_leg
                pend_qty = pend_qty if pend_qty > 0 else 0
                pend_cost_total = pend_cost_total if pend_cost_total > 0 else 0
                if pend_qty > 0 and pend_cost_total > 0:
                    avg_pending_px = pend_cost_total / float(pend_qty)
                    resting_fill_cost += int(round(avg_pending_px * pending_used))
                    resting_fee_total += maker_model.fee_cents(int(round(avg_pending_px)), pending_used)
                else:
                    resting_missing += 1

        levels = book.yes_ask_levels if side == "yes" else book.no_ask_levels
        if remaining > 0:
            cost, filled = simulate_buy(levels, remaining, taker_model)
            remaining_cost += cost
            if filled < remaining:
                short_legs += 1
        if remaining_no_pending > 0:
            cost_np, filled_np = simulate_buy(levels, remaining_no_pending, taker_model)
            remaining_cost_no_pending += cost_np
            if filled_np < remaining_no_pending:
                short_legs_no_pending += 1

    payout = payout_per_contract * target_qty
    if move_resting and maker_px_by_ticker is not None:
        cost_map = pending_cost_map or {}
        fee_map = pending_fee_map or {}
        pending_cost_used = sum(cost_map.values())
        pending_fee_used = sum(fee_map.values()) if fee_map else 0
    else:
        pending_cost_used = pending_cost
        pending_fee_used = pending_fee_total

    total_cost_ex_pos = (pending_cost_used or 0) + (pending_fee_used or 0) + remaining_cost
    edge_ex_pos = payout - total_cost_ex_pos if total_cost_ex_pos > 0 else None
    roi_ex_pos = (edge_ex_pos / total_cost_ex_pos) if edge_ex_pos is not None else None
    total_cost_incl_pos = None
    edge_incl_pos = None
    roi_incl_pos = None
    if spent_missing == 0:
        total_cost_incl_pos = spent_cost + total_cost_ex_pos
        edge_incl_pos = payout - total_cost_incl_pos if total_cost_incl_pos > 0 else None
        roi_incl_pos = (edge_incl_pos / total_cost_incl_pos) if edge_incl_pos is not None else None
    pnl_resting_fill = None
    roi_resting_fill = None
    total_resting_cost = None
    if remaining_min == 0 and resting_missing == 0:
        total_resting_cost = resting_fill_cost + resting_fee_total
        pnl_resting_fill = payout - total_resting_cost
        if total_resting_cost > 0:
            roi_resting_fill = pnl_resting_fill / total_resting_cost
    pnl_resting_plus_taker = None
    roi_resting_plus_taker = None
    total_resting_plus_taker = None
    if spent_missing == 0:
        total_resting_plus_taker = (
            spent_cost + (pending_cost_used or 0) + (pending_fee_used or 0) + remaining_cost
        )
        pnl_resting_plus_taker = payout - total_resting_plus_taker
        if total_resting_plus_taker > 0:
            roi_resting_plus_taker = pnl_resting_plus_taker / total_resting_plus_taker
    pnl_cancel_resting_taker = None
    roi_cancel_resting_taker = None
    total_cancel_taker = None
    if spent_missing == 0 and short_legs_no_pending == 0:
        total_cancel_taker = spent_cost + remaining_cost_no_pending
        pnl_cancel_resting_taker = payout - total_cancel_taker
        if total_cancel_taker > 0:
            roi_cancel_resting_taker = pnl_cancel_resting_taker / total_cancel_taker

    return {
        "side": side,
        "target_qty": target_qty,
        "remaining_total": remaining_total,
        "remaining_min": remaining_min,
        "remaining_max": remaining_max,
        "pending_cost": pending_cost,
        "pending_fee_total": pending_fee_total if pending_fee_total > 0 else None,
        "pending_cost_used": pending_cost_used,
        "pending_fee_used": pending_fee_used,
        "pending_moved_qty": pending_moved_qty,
        "pending_moved_orders": pending_moved_orders,
        "remaining_cost": remaining_cost,
        "remaining_cost_no_pending": remaining_cost_no_pending,
        "spent_cost": spent_cost if spent_cost > 0 else None,
        "spent_missing_legs": spent_missing,
        "total_cost_ex_positions": total_cost_ex_pos,
        "payout": payout,
        "edge_ex_positions": edge_ex_pos,
        "roi_ex_positions": roi_ex_pos,
        "total_cost_incl_positions": total_cost_incl_pos,
        "edge_incl_positions": edge_incl_pos,
        "roi_incl_positions": roi_incl_pos,
        "pnl_resting_fill": pnl_resting_fill,
        "roi_resting_fill": roi_resting_fill,
        "total_resting_cost": total_resting_cost,
        "resting_fee_total": resting_fee_total if resting_fee_total > 0 else None,
        "resting_missing_legs": resting_missing,
        "pnl_resting_plus_taker": pnl_resting_plus_taker,
        "roi_resting_plus_taker": roi_resting_plus_taker,
        "total_resting_plus_taker": total_resting_plus_taker,
        "pnl_cancel_resting_taker": pnl_cancel_resting_taker,
        "roi_cancel_resting_taker": roi_cancel_resting_taker,
        "total_cancel_taker": total_cancel_taker,
        "short_legs_no_pending": short_legs_no_pending,
        "short_legs": short_legs,
    }


def _print_adjusted_basket(summary: Dict[str, object]) -> None:
    side = summary.get("side")
    print(f"Position-adjusted {str(side).upper()} basket")
    print(
        "Remaining qty per leg: "
        f"min {summary.get('remaining_min')}, max {summary.get('remaining_max')} "
        f"(total {summary.get('remaining_total')})"
    )
    spent_cost = summary.get("spent_cost")
    spent_missing = summary.get("spent_missing_legs") or 0
    if spent_cost is None and spent_missing:
        spent_text = "n/a (no fills data)"
    elif spent_missing:
        spent_text = f"{fmt_money(int(spent_cost))} + unknown"
    else:
        spent_text = fmt_money(int(spent_cost)) if spent_cost is not None else "n/a"
    print(f"Spent cost (positions, est): {spent_text}")
    pending_fee = summary.get("pending_fee_total")
    pending_line = f"Pending cost (resting, excl fees): {fmt_money(int(summary.get('pending_cost') or 0))}"
    if pending_fee:
        pending_line += f" (maker fees {fmt_money(int(pending_fee))})"
    print(pending_line)
    pending_used = summary.get("pending_cost_used")
    if pending_used is not None and pending_used != summary.get("pending_cost"):
        moved_qty = summary.get("pending_moved_qty") or 0
        moved_orders = summary.get("pending_moved_orders") or 0
        fee_used = summary.get("pending_fee_used")
        note = ""
        if fee_used:
            note = f" (maker fees {fmt_money(int(fee_used))})"
        print(
            f"Pending cost (moved to maker px): {fmt_money(int(pending_used))}{note} "
            f"[moved {moved_qty} qty across {moved_orders} orders]"
        )
    print(
        f"Suggested remaining cost (taker): {fmt_money(int(summary.get('remaining_cost') or 0))}"
    )
    edge = summary.get("edge_ex_positions")
    roi = summary.get("roi_ex_positions")
    print(
        f"Edge excl positions: {fmt_money(int(edge)) if edge is not None else 'n/a'}, "
        f"ROI {fmt_roi(roi) if roi is not None else 'n/a'}"
    )
    edge_incl = summary.get("edge_incl_positions")
    roi_incl = summary.get("roi_incl_positions")
    if edge_incl is not None:
        print(
            f"Edge incl positions: {fmt_money(int(edge_incl))}, "
            f"ROI {fmt_roi(roi_incl) if roi_incl is not None else 'n/a'}"
        )
    pnl_resting = summary.get("pnl_resting_fill")
    if pnl_resting is not None:
        fee_note = ""
        rest_fee = summary.get("resting_fee_total")
        if rest_fee:
            fee_note = f" (maker fees {fmt_money(int(rest_fee))})"
        roi_resting = summary.get("roi_resting_fill")
        roi_note = f", ROI {fmt_roi(roi_resting)}" if roi_resting is not None else ""
        print(
            f"PNL if all resting fill at maker: {fmt_money(int(pnl_resting))}{fee_note}{roi_note}"
        )
    pnl_rest_plus = summary.get("pnl_resting_plus_taker")
    if pnl_rest_plus is not None:
        roi_rest_plus = summary.get("roi_resting_plus_taker")
        roi_note = f", ROI {fmt_roi(roi_rest_plus)}" if roi_rest_plus is not None else ""
        print(
            f"PNL if resting fill + remaining taker: {fmt_money(int(pnl_rest_plus))}{roi_note}"
        )
    pnl_cancel = summary.get("pnl_cancel_resting_taker")
    if pnl_cancel is not None:
        roi_cancel = summary.get("roi_cancel_resting_taker")
        roi_note = f", ROI {fmt_roi(roi_cancel)}" if roi_cancel is not None else ""
        print(
            f"PNL if cancel resting + take remaining at taker: {fmt_money(int(pnl_cancel))}{roi_note}"
        )
    if summary.get("short_legs"):
        print(f"Warning: insufficient depth on {summary.get('short_legs')} legs.")
    print("")


def _liquidation_summary(
    side: str,
    books: List[NormalizedBook],
    trade_state: Dict[str, object],
    fee_model: FeeModel,
    hold_payout: Optional[int] = None,
) -> Dict[str, object]:
    pos_yes = trade_state.get("pos_yes", {})
    rows: List[Dict[str, object]] = []
    total_net = 0
    total_gross = 0
    total_fee = 0
    missing: List[str] = []
    for book in books:
        ticker = book.market_ticker
        pos = int(pos_yes.get(ticker, 0)) if isinstance(pos_yes, dict) else 0
        qty = max(0, pos) if side == "yes" else max(0, -pos)
        if qty <= 0:
            continue
        bid = _side_bid(book, side)
        if bid is None:
            missing.append(ticker)
            continue
        bid_px = int(bid[0])
        gross = bid_px * qty
        fee = fee_model.fee_cents(bid_px, qty)
        net = fee_model.sell_proceeds_cents(bid_px, qty)
        total_gross += gross
        total_fee += fee
        total_net += net
        rows.append(
            {
                "ticker": ticker,
                "qty": qty,
                "bid": bid_px,
                "gross": gross,
                "fee": fee,
                "net": net,
            }
        )

    delta = None
    roi = None
    if hold_payout is not None:
        delta = total_net - hold_payout
        roi = (delta / hold_payout) if hold_payout > 0 else None

    return {
        "side": side,
        "rows": rows,
        "total_net": total_net if rows else None,
        "total_gross": total_gross if rows else None,
        "total_fee": total_fee if rows else None,
        "hold_payout": hold_payout,
        "delta": delta,
        "roi": roi,
        "missing": missing,
    }


def _print_liquidation_summary(summary: Dict[str, object]) -> None:
    side = str(summary.get("side") or "").upper()
    rows = summary.get("rows") or []
    if not rows:
        return
    print(f"Liquidation comparison for {side} basket")
    header = f"{'Ticker':<24} | {'Qty':>3} | {'Bid':>3} | {'Gross':>8} | {'Fee':>6} | {'Net':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['ticker']:<24} | "
            f"{int(row['qty']):>3} | "
            f"{int(row['bid']):>3} | "
            f"{fmt_money(int(row['gross'])):>8} | "
            f"{fmt_money(int(row['fee'])):>6} | "
            f"{fmt_money(int(row['net'])):>8}"
        )
    total_net = summary.get("total_net")
    hold_payout = summary.get("hold_payout")
    delta = summary.get("delta")
    roi = summary.get("roi")
    print(
        f"Total liquidation net: {fmt_money(int(total_net)) if total_net is not None else 'n/a'}"
    )
    print(
        f"Guaranteed hold payout: {fmt_money(int(hold_payout)) if hold_payout is not None else 'n/a'}"
    )
    if delta is not None:
        print(
            f"Liquidate minus hold: {fmt_money(int(delta))}, ROI {fmt_roi(roi) if roi is not None else 'n/a'}"
        )
    missing = summary.get("missing") or []
    if missing:
        sample = ", ".join(str(t) for t in missing[:5])
        tail = "..." if len(missing) > 5 else ""
        print(f"Missing bids for liquidation on: {sample}{tail}")
    print("")


def _recommend_now(
    yes_eval: BasketEvaluation,
    no_eval: BasketEvaluation,
    yes_adj: Optional[Dict[str, object]],
    no_adj: Optional[Dict[str, object]],
) -> str:
    if yes_adj is not None and no_adj is not None:
        yes_edge = yes_adj.get("edge_ex_positions")
        no_edge = no_adj.get("edge_ex_positions")
        yes_remain = yes_adj.get("remaining_total")
        no_remain = no_adj.get("remaining_total")
        if (yes_edge is None or yes_edge <= 0) and (no_edge is None or no_edge <= 0):
            return "Recommended action now: do nothing (no positive incremental edge)."
        if yes_edge is None:
            return "Recommended action now: consider NO basket (positive incremental edge)."
        if no_edge is None:
            return "Recommended action now: consider YES basket (positive incremental edge)."
        if yes_edge >= no_edge:
            if yes_remain == 0:
                return "Recommended action now: do nothing (YES basket already covered)."
            return f"Recommended action now: complete YES basket at taker for remaining qty (edge {fmt_money(int(yes_edge))})."
        if no_remain == 0:
            return "Recommended action now: do nothing (NO basket already covered)."
        return f"Recommended action now: complete NO basket at taker for remaining qty (edge {fmt_money(int(no_edge))})."

    if yes_eval.edge_cents <= 0 and no_eval.edge_cents <= 0:
        return "Recommended action now: do nothing (no positive taker edge)."
    if yes_eval.edge_cents >= no_eval.edge_cents:
        return f"Recommended action now: YES basket at taker (edge {fmt_money(yes_eval.edge_cents)})."
    return f"Recommended action now: NO basket at taker (edge {fmt_money(no_eval.edge_cents)})."


def _recommend_liquidation(
    yes_liq: Optional[Dict[str, object]],
    no_liq: Optional[Dict[str, object]],
) -> str:
    best_side = None
    best_delta = None
    if yes_liq is not None and yes_liq.get("delta") is not None:
        best_side = "yes"
        best_delta = int(yes_liq.get("delta"))
    if no_liq is not None and no_liq.get("delta") is not None:
        no_delta = int(no_liq.get("delta"))
        if best_delta is None or no_delta > best_delta:
            best_side = "no"
            best_delta = no_delta
    if best_side is None or best_delta is None:
        return "Recommended liquidation action: no complete side available."
    if best_delta > 0:
        return (
            f"Recommended liquidation action: sell {best_side.upper()} basket now "
            f"(liquidation advantage {fmt_money(best_delta)} vs hold)."
        )
    return (
        f"Recommended liquidation action: hold positions "
        f"(liquidation advantage {fmt_money(best_delta)} vs hold)."
    )


def _select_exec_side(
    choice: str,
    yes_eval: BasketEvaluation,
    no_eval: BasketEvaluation,
) -> str:
    if choice in ("yes", "no"):
        return choice
    if yes_eval.edge_cents >= no_eval.edge_cents:
        return "yes"
    return "no"


def _select_liquidate_side(choice: str, yes_liq: Dict[str, object], no_liq: Dict[str, object]) -> str:
    if choice in ("yes", "no"):
        return choice
    yes_delta = yes_liq.get("delta")
    no_delta = no_liq.get("delta")
    if yes_delta is None and no_delta is None:
        return "yes"
    if yes_delta is None:
        return "no"
    if no_delta is None:
        return "yes"
    if yes_delta >= no_delta:
        return "yes"
    return "no"


def _build_exec_orders(
    side: str,
    plan_rows: List[Dict[str, object]],
    qty: int,
    only_maker: bool,
    post_only: bool,
    client_prefix: str,
    pos_yes: Optional[Dict[str, int]] = None,
    pending_qty: Optional[Dict[Tuple[str, str], int]] = None,
    pending_qty_by_price: Optional[Dict[Tuple[str, str, int], int]] = None,
    ignore_pending_for_taker: bool = False,
    match_pending_price: bool = False,
) -> Tuple[List[dict], List[str]]:
    orders: List[dict] = []
    skipped: List[str] = []
    pos_yes = pos_yes or {}
    pending_qty = pending_qty or {}
    pending_qty_by_price = pending_qty_by_price or {}
    for idx, row in enumerate(plan_rows):
        ticker = row.get("ticker")
        rec = row.get("rec")
        bid = row.get("bid")
        ask = row.get("ask")
        maker_px = row.get("maker_px")
        price = None
        use_post_only = post_only
        if rec == "maker":
            price = maker_px if maker_px is not None else bid
        elif rec == "taker" and not only_maker:
            price = ask
            use_post_only = False
        elif rec == "taker" and only_maker:
            skipped.append(f"{ticker}: taker leg skipped (maker-only)")
            continue
        else:
            skipped.append(str(ticker))
            continue
        if price is None:
            skipped.append(f"{ticker}: missing price")
            continue
        pos = int(pos_yes.get(str(ticker), 0))
        pos_qty = max(0, pos) if side == "yes" else max(0, -pos)
        pending = int(pending_qty.get((str(ticker), side), 0))
        if match_pending_price and price is not None:
            pending = int(pending_qty_by_price.get((str(ticker), side, int(price)), 0))
        if rec == "taker" and ignore_pending_for_taker:
            pending = 0
        remaining = max(0, qty - pos_qty - pending)
        if remaining <= 0:
            skipped.append(f"{ticker}: already covered (pos+pending)")
            continue
        order = {
            "ticker": ticker,
            "side": side,
            "action": "buy",
            "count": remaining,
            "type": "limit",
            "client_order_id": f"{client_prefix}-{idx}",
        }
        if side == "yes":
            order["yes_price"] = int(price)
        else:
            order["no_price"] = int(price)
        if use_post_only:
            order["post_only"] = True
        orders.append(order)
    return orders, skipped


def _build_liquidate_orders(
    side: str,
    books: List[NormalizedBook],
    pos_yes: Dict[str, int],
    client_prefix: str,
) -> Tuple[List[dict], List[str]]:
    orders: List[dict] = []
    skipped: List[str] = []
    for idx, book in enumerate(books):
        pos = int(pos_yes.get(book.market_ticker, 0)) if isinstance(pos_yes, dict) else 0
        qty = max(0, pos) if side == "yes" else max(0, -pos)
        if qty <= 0:
            continue
        bid = _side_bid(book, side)
        if bid is None:
            skipped.append(f"{book.market_ticker}: missing {side.upper()} bid")
            continue
        price = int(bid[0])
        order = {
            "ticker": book.market_ticker,
            "side": side,
            "action": "sell",
            "count": qty,
            "type": "limit",
            "client_order_id": f"{client_prefix}-{idx}",
        }
        if side == "yes":
            order["yes_price"] = price
        else:
            order["no_price"] = price
        orders.append(order)
    return orders, skipped


def _estimate_orders_cost(
    orders: List[dict],
    maker_model: FeeModel,
    taker_model: FeeModel,
) -> int:
    total = 0
    for o in orders:
        qty = int(o.get("count") or 0)
        if qty <= 0:
            continue
        post_only = bool(o.get("post_only"))
        if "yes_price" in o:
            px = int(o["yes_price"])
        else:
            px = int(o.get("no_price") or 0)
        fee_model = maker_model if post_only else taker_model
        total += fee_model.buy_cost_cents(px, qty)
    return total


def _estimate_orders_proceeds(
    orders: List[dict],
    maker_model: FeeModel,
    taker_model: FeeModel,
) -> int:
    total = 0
    for o in orders:
        qty = int(o.get("count") or 0)
        if qty <= 0:
            continue
        post_only = bool(o.get("post_only"))
        if "yes_price" in o:
            px = int(o["yes_price"])
        else:
            px = int(o.get("no_price") or 0)
        fee_model = maker_model if post_only else taker_model
        total += fee_model.sell_proceeds_cents(px, qty)
    return total


def _order_id(order: dict) -> Optional[str]:
    for k in ("order_id", "id", "order"):
        if k in order and order.get(k):
            return str(order.get(k))
    return None


def _build_move_resting_plan(
    orders: List[dict],
    side: str,
    maker_px_by_ticker: Dict[str, int],
    post_only: bool,
    top_bid_px_by_ticker: Optional[Dict[str, int]] = None,
    top_bid_qty_by_ticker: Optional[Dict[str, int]] = None,
    own_qty_by_price: Optional[Dict[Tuple[str, str, int], int]] = None,
) -> Tuple[List[str], List[dict]]:
    cancel_ids: List[str] = []
    replacements: List[dict] = []
    for o in orders:
        if str(o.get("action") or "").lower() != "buy":
            continue
        if str(o.get("side") or "").lower() != side:
            continue
        ticker = o.get("ticker")
        if not ticker:
            continue
        maker_px = maker_px_by_ticker.get(str(ticker))
        if maker_px is None:
            continue
        remaining = o.get("remaining_count")
        if remaining is None:
            remaining = o.get("remaining_count_fp") or o.get("initial_count") or 0
        try:
            remaining_int = int(float(remaining))
        except (TypeError, ValueError):
            remaining_int = 0
        if remaining_int <= 0:
            continue
        cur_px = _order_price_cents(o, side)
        if cur_px is None:
            continue
        top_bid_px = top_bid_px_by_ticker.get(str(ticker)) if top_bid_px_by_ticker else None
        top_bid_qty = top_bid_qty_by_ticker.get(str(ticker)) if top_bid_qty_by_ticker else None
        own_qty = 0
        if own_qty_by_price is not None:
            own_qty = own_qty_by_price.get((str(ticker), str(side), int(cur_px)), 0)
        if not _should_move_resting(
            cur_px=cur_px,
            maker_px=maker_px,
            top_bid_px=top_bid_px,
            top_bid_qty=top_bid_qty,
            own_qty_at_price=own_qty,
        ):
            continue
        oid = _order_id(o)
        if oid:
            cancel_ids.append(oid)
        repl = {
            "ticker": ticker,
            "side": side,
            "action": "buy",
            "count": remaining_int,
            "type": "limit",
        }
        if side == "yes":
            repl["yes_price"] = int(maker_px)
        else:
            repl["no_price"] = int(maker_px)
        if post_only:
            repl["post_only"] = True
        replacements.append(repl)
    return cancel_ids, replacements


def _print_order_response(resp: dict) -> None:
    if not isinstance(resp, dict):
        print("Trade API response: <non-dict>")
        return
    if "error" in resp or "errors" in resp or "message" in resp:
        print("Trade API error:", resp.get("error") or resp.get("errors") or resp.get("message"))
    raw_results = resp.get("orders") or resp.get("results") or resp.get("order_results") or []
    if not isinstance(raw_results, list):
        print("Trade API response missing per-order results.")
        return
    success = 0
    fail = 0
    for idx, item in enumerate(raw_results):
        err = item.get("error") or item.get("errors") or item.get("message")
        ok = err is None and (item.get("order") is not None or item.get("status") in ("accepted", "resting"))
        if ok:
            success += 1
        else:
            fail += 1
        if err:
            print(f"Order {idx} error: {err}")
    print(f"Batch create orders: {success} succeeded, {fail} failed (total {len(raw_results)})")

def print_table(rows: List[BasketEvaluation]) -> None:
    header = (
        f"{'Basket':<6} | {'Legs':>4} | {'Target':>6} | {'Fill':>4} | "
        f"{'MaxTop':>6} | {'MaxDepth':>8} | {'Cost':>10} | {'Payout':>10} | "
        f"{'Edge':>10} | {'ROI':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.side.upper():<6} | {row.legs:>4} | {row.target_qty:>6} | "
            f"{row.filled_qty:>4} | {row.max_qty_top:>6} | {row.max_qty_depth:>8} | "
            f"{fmt_money(row.cost_cents):>10} | {fmt_money(row.payout_cents):>10} | "
            f"{fmt_money(row.edge_cents):>10} | {fmt_roi(row.roi):>7}"
        )


def verdict(rows: List[BasketEvaluation]) -> str:
    viable = [r for r in rows if r.filled_qty > 0]
    if not viable:
        return "No arb (no executable liquidity)."
    best = max(viable, key=lambda r: r.edge_cents)
    if best.edge_cents <= 0:
        return "No arb (edges <= 0 at available depth)."
    extra = ""
    if best.filled_qty < best.target_qty:
        extra = f" Only {best.filled_qty}/{best.target_qty} qty available."
    return f"ARB: buy {best.side.upper()} basket (edge {fmt_money(best.edge_cents)}).{extra}"


def main() -> None:
    args = parse_args()
    market_list = parse_markets(args.markets)
    if not args.event_ticker and not market_list:
        print("Error: provide EVENT_TICKER or --markets.", file=sys.stderr)
        sys.exit(2)

    try:
        client = load_client_from_env(cache_seconds=args.cache_seconds)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    event = {}
    tickers = market_list
    market_by_ticker: Dict[str, dict] = {}
    if not tickers:
        event, markets = client.get_event_markets(args.event_ticker)
        tickers = [m.get("ticker") for m in markets if isinstance(m, dict) and m.get("ticker")]
        market_by_ticker = {
            m.get("ticker"): m for m in markets if isinstance(m, dict) and m.get("ticker")
        }
    if not tickers:
        print("No markets found.", file=sys.stderr)
        sys.exit(1)

    books: List[NormalizedBook] = []
    for idx, ticker in enumerate(tickers, start=1):
        ob = client.get_orderbook(ticker, depth=args.depth)
        snapshot = market_by_ticker.get(ticker)
        book = normalize_orderbook(ticker, ob, depth=args.depth, market_snapshot=snapshot)
        books.append(book)
        if args.verbose:
            print(
                f"{idx:>3}/{len(tickers)} {ticker} "
                f"YES ask={book.yes_ask} NO ask={book.no_ask}"
            )
        if args.print_bbo:
            def _fmt(q: Optional[Tuple[int, int]]) -> str:
                if not q:
                    return "n/a"
                return f"{q[0]}c x{q[1]}"

            print(
                f"{idx:>3}/{len(tickers)} {ticker} "
                f"YES bid={_fmt(book.yes_bid)} ask={_fmt(book.yes_ask)} | "
                f"NO bid={_fmt(book.no_bid)} ask={_fmt(book.no_ask)}"
            )

    kept: List[NormalizedBook] = []
    dropped: List[Tuple[str, str]] = []
    warnings: List[Tuple[str, str]] = []
    for book in books:
        clean_book, ok, reason, warning = _book_is_complete(
            book,
            allow_crossed=args.allow_crossed,
            allow_ask_only=args.allow_ask_only,
        )
        if ok:
            kept.append(clean_book)
            if warning:
                warnings.append((book.market_ticker, warning))
        else:
            dropped.append((book.market_ticker, reason))

    if dropped:
        print(f"Filtered {len(dropped)}/{len(books)} markets due to incomplete/crossed books.")
        for ticker, reason in dropped:
            print(f"  drop {ticker}: {reason}")
        print(
            "Warning: basket excludes dropped outcomes; edge is not guaranteed for the full event."
        )
    if warnings:
        print("Kept markets with warnings:")
        for ticker, reason in warnings:
            print(f"  keep {ticker}: {reason}")

    books = kept
    if not books:
        print("No markets with usable asks.", file=sys.stderr)
        sys.exit(1)

    trade_state = None
    trade_client = None
    if args.trade_sync or args.execute:
        try:
            trade_client = load_trade_client_from_env(cache_seconds=args.cache_seconds)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)

    if args.trade_sync:
        if not args.event_ticker:
            print("Trade sync requires EVENT_TICKER.", file=sys.stderr)
            sys.exit(2)
        assert trade_client is not None
        positions = trade_client.get_positions(args.event_ticker, subaccount=args.subaccount)
        orders = trade_client.get_orders(
            args.event_ticker, status="resting", subaccount=args.subaccount
        )
        trade_state = _build_trade_state(positions, orders)
        if args.fills_lookback_hours is not None and args.fills_lookback_hours > 0:
            min_ts = int(time.time() * 1000 - args.fills_lookback_hours * 3600 * 1000)
            fills: List[dict] = []
            for t in [b.market_ticker for b in books]:
                fills.extend(
                    trade_client.get_fills(t, min_ts=min_ts, subaccount=args.subaccount)
                )
            trade_state.update(_build_fills_state(fills))
            trade_state["fills_lookback_hours"] = args.fills_lookback_hours
        _print_trade_state_summary(trade_state)

    fee_model = FeeModel(args.fee_mode)

    legs = len(books)
    winners = int(args.yes_winners)
    if winners < 0 or winners > legs:
        print(
            f"Invalid --yes-winners {winners}: must be between 0 and {legs}.",
            file=sys.stderr,
        )
        sys.exit(2)
    books_yes_strat, dropped_yes_ask = _filter_books_with_ask(books, "yes")
    books_no_strat, dropped_no_ask = _filter_books_with_ask(books, "no")
    yes_legs = len(books_yes_strat)
    no_legs = len(books_no_strat)
    yes_payout_per_contract = min(winners, yes_legs) * 100
    no_payout_per_contract = max(0, (no_legs - winners) * 100)
    yes_levels = [b.yes_ask_levels for b in books_yes_strat]
    no_levels = [b.no_ask_levels for b in books_no_strat]

    yes_eval = evaluate_basket(
        "yes",
        yes_levels,
        args.qty,
        payout_per_contract=yes_payout_per_contract,
        fee_model=fee_model,
    )
    no_eval = evaluate_basket(
        "no",
        no_levels,
        args.qty,
        payout_per_contract=no_payout_per_contract,
        fee_model=fee_model,
    )

    print_table([yes_eval, no_eval])
    assumption = f"Assumption: exactly {winners} outcome(s) resolve YES."
    if winners == 1:
        assumption = "Assumption: exactly one outcome resolves YES (mutually exclusive event)."
    print(assumption)
    if dropped_yes_ask:
        print(
            f"Warning: YES basket uses {len(books_yes_strat)}/{legs} markets with YES asks; "
            "edge not guaranteed."
        )
    if dropped_no_ask:
        print(
            f"Warning: NO basket uses {len(books_no_strat)}/{legs} markets with NO asks; "
            "edge not guaranteed."
        )
    print(verdict([yes_eval, no_eval]))
    print("")

    maker_model = FeeModel("maker")
    taker_model = FeeModel("taker")

    taker_yes_eval = evaluate_basket(
        "yes",
        yes_levels,
        args.qty,
        payout_per_contract=yes_payout_per_contract,
        fee_model=taker_model,
    )
    taker_no_eval = evaluate_basket(
        "no",
        no_levels,
        args.qty,
        payout_per_contract=no_payout_per_contract,
        fee_model=taker_model,
    )
    if taker_yes_eval.edge_cents > 0:
        _print_taker_arb_summary(
            "yes",
            books_yes_strat,
            args.qty,
            yes_payout_per_contract,
            taker_model,
        )
    if taker_no_eval.edge_cents > 0:
        _print_taker_arb_summary(
            "no",
            books_no_strat,
            args.qty,
            no_payout_per_contract,
            taker_model,
        )
    strategies = [
        ("maker_bid", _limit_price_maker_bid),
        ("improve_1", _limit_price_improve_1),
        ("mid", _limit_price_mid),
    ]

    def build_strategy_rows(
        side: str,
        books_for_side: List[NormalizedBook],
        payout_per_contract: int,
    ) -> Tuple[List[Dict[str, object]], Optional[int], Optional[int]]:
        if not books_for_side:
            return [], None, None
        bid_levels = [_side_bid_levels(b, side) for b in books_for_side]
        ask_levels = [_side_ask_levels(b, side) for b in books_for_side]
        sizes = _unique_sizes(
            [
                ("fixed", args.qty),
                ("top", max_qty_top(bid_levels)),
                ("depth", max_qty_depth(bid_levels)),
            ]
        )
        min_spread, max_spread = _spread_stats(books_for_side, side)
        rows: List[Dict[str, object]] = []
        for size_label, qty in sizes:
            taker_eval = evaluate_basket(
                side,
                ask_levels,
                qty,
                payout_per_contract=payout_per_contract,
                fee_model=taker_model,
            )
            worst_cost = taker_eval.cost_cents if taker_eval.filled_qty > 0 else None
            worst_payout = taker_eval.payout_cents if taker_eval.filled_qty > 0 else None
            worst_edge = taker_eval.edge_cents if taker_eval.filled_qty > 0 else None
            worst_roi = taker_eval.roi if taker_eval.filled_qty > 0 else None
            taker_fill = taker_eval.filled_qty if taker_eval.filled_qty > 0 else None
            for strat_name, strat_fn in strategies:
                prices = [strat_fn(book, side) for book in books_for_side]
                limit_cost = _limit_cost(prices, qty, maker_model)
                limit_payout = payout_per_contract * qty if limit_cost is not None else None
                limit_edge = (
                    (limit_payout - limit_cost) if limit_cost is not None else None
                )
                limit_roi = (limit_edge / limit_cost) if limit_cost else None
                max_queue = None
                if all(p is not None for p in prices):
                    max_queue = max(
                        _queue_ahead(levels, int(p))
                        for levels, p in zip(bid_levels, prices)
                    )
                rows.append(
                    {
                        "strategy": strat_name,
                        "size_label": size_label,
                        "target_qty": qty,
                        "limit_cost": limit_cost,
                        "limit_payout": limit_payout,
                        "limit_edge": limit_edge,
                        "limit_roi": limit_roi,
                        "worst_cost": worst_cost,
                        "worst_payout": worst_payout,
                        "worst_edge": worst_edge,
                        "worst_roi": worst_roi,
                        "taker_fill": taker_fill,
                        "max_queue": max_queue,
                    }
                )
        return rows, min_spread, max_spread

    if dropped_no_ask:
        sample = ", ".join(dropped_no_ask[:5])
        tail = "..." if len(dropped_no_ask) > 5 else ""
        print(
            f"Warning: NO strategies exclude {len(dropped_no_ask)} market(s) missing NO ask: "
            f"{sample}{tail}"
        )
    if dropped_yes_ask:
        sample = ", ".join(dropped_yes_ask[:5])
        tail = "..." if len(dropped_yes_ask) > 5 else ""
        print(
            f"Warning: YES strategies exclude {len(dropped_yes_ask)} market(s) missing YES ask: "
            f"{sample}{tail}"
        )

    yes_rows, yes_min_spread, yes_max_spread = build_strategy_rows(
        "yes", books_yes_strat, yes_payout_per_contract
    )
    no_rows, no_min_spread, no_max_spread = build_strategy_rows(
        "no", books_no_strat, no_payout_per_contract
    )
    _print_strategy_table("yes", yes_rows, min_spread=yes_min_spread, max_spread=yes_max_spread)
    _print_strategy_table("no", no_rows, min_spread=no_min_spread, max_spread=no_max_spread)

    yes_plan = _hybrid_plan_rows(
        books_yes_strat,
        "yes",
        args.qty,
        maker_model,
        taker_model,
        maker_mode=args.exec_maker_mode,
    )
    no_plan = _hybrid_plan_rows(
        books_no_strat,
        "no",
        args.qty,
        maker_model,
        taker_model,
        maker_mode=args.exec_maker_mode,
    )
    yes_mid_plan = _hybrid_plan_rows(
        books_yes_strat,
        "yes",
        args.qty,
        maker_model,
        taker_model,
        maker_mode="midpoint",
    )
    no_mid_plan = _hybrid_plan_rows(
        books_no_strat,
        "no",
        args.qty,
        maker_model,
        taker_model,
        maker_mode="midpoint",
    )
    _print_hybrid_plan("yes", yes_plan, args.qty)
    _print_hybrid_plan("no", no_plan, args.qty)
    print("Midpoint execution plan")
    _print_hybrid_plan("yes", yes_mid_plan, args.qty)
    _print_hybrid_plan("no", no_mid_plan, args.qty)
    yes_hybrid = _hybrid_summary(
        "yes",
        yes_plan,
        args.qty,
        payout_per_contract=yes_payout_per_contract,
        maker_model=maker_model,
        taker_model=taker_model,
    )
    no_hybrid = _hybrid_summary(
        "no",
        no_plan,
        args.qty,
        payout_per_contract=no_payout_per_contract,
        maker_model=maker_model,
        taker_model=taker_model,
    )
    yes_mid_hybrid = _hybrid_summary(
        "yes",
        yes_mid_plan,
        args.qty,
        payout_per_contract=yes_payout_per_contract,
        maker_model=maker_model,
        taker_model=taker_model,
    )
    no_mid_hybrid = _hybrid_summary(
        "no",
        no_mid_plan,
        args.qty,
        payout_per_contract=no_payout_per_contract,
        maker_model=maker_model,
        taker_model=taker_model,
    )
    _print_hybrid_summary(yes_hybrid)
    _print_hybrid_summary(no_hybrid)
    print("Midpoint summary")
    _print_hybrid_summary(yes_mid_hybrid)
    _print_hybrid_summary(no_mid_hybrid)

    yes_maker_px = {
        str(r["ticker"]): int(r["maker_px"])
        for r in yes_plan
        if r.get("maker_px") is not None
    }
    no_maker_px = {
        str(r["ticker"]): int(r["maker_px"])
        for r in no_plan
        if r.get("maker_px") is not None
    }

    if trade_state is not None and (args.print_resting or args.move_resting):
        pending_orders = trade_state.get("pending_orders", [])
        yes_bid_px, yes_bid_qty = _top_bid_maps(books, "yes")
        no_bid_px, no_bid_qty = _top_bid_maps(books, "no")
        own_qty_by_price = (
            _pending_qty_by_ticker_side_price(pending_orders)
            if isinstance(pending_orders, list)
            else {}
        )
        rows = _resting_move_rows(
            pending_orders if isinstance(pending_orders, list) else [],
            yes_maker_px,
            no_maker_px,
            yes_bid_px,
            yes_bid_qty,
            no_bid_px,
            no_bid_qty,
            own_qty_by_price,
        )
        if rows:
            _print_resting_moves(rows)
        elif args.print_resting:
            print("Resting orders vs maker price")
            print("<none>")
            print("")

    if args.breakdown:
        _print_leg_breakdown("yes", books, args.qty, taker_model)
        _print_leg_breakdown("no", books, args.qty, taker_model)

    yes_adj: Optional[Dict[str, object]] = None
    no_adj: Optional[Dict[str, object]] = None
    yes_liq: Optional[Dict[str, object]] = None
    no_liq: Optional[Dict[str, object]] = None
    if trade_state is not None:
        yes_adj = _adjusted_basket(
            "yes",
            books,
            args.qty,
            trade_state,
            taker_model=taker_model,
            maker_model=maker_model,
            payout_per_contract=yes_payout_per_contract,
            move_resting=args.move_resting,
            maker_px_by_ticker=yes_maker_px,
        )
        no_adj = _adjusted_basket(
            "no",
            books,
            args.qty,
            trade_state,
            taker_model=taker_model,
            maker_model=maker_model,
            payout_per_contract=no_payout_per_contract,
            move_resting=args.move_resting,
            maker_px_by_ticker=no_maker_px,
        )
        _print_adjusted_basket(yes_adj)
        _print_adjusted_basket(no_adj)
        print(_recommend_now(yes_eval, no_eval, yes_adj, no_adj))
        yes_liq = _liquidation_summary(
            "yes",
            books,
            trade_state,
            taker_model,
            hold_payout=yes_adj.get("payout") if yes_adj.get("remaining_total") == 0 else None,
        )
        no_liq = _liquidation_summary(
            "no",
            books,
            trade_state,
            taker_model,
            hold_payout=no_adj.get("payout") if no_adj.get("remaining_total") == 0 else None,
        )
        _print_liquidation_summary(yes_liq)
        _print_liquidation_summary(no_liq)
        print(_recommend_liquidation(yes_liq, no_liq))
        print("")
    else:
        print(_recommend_now(yes_eval, no_eval, None, None))
        print("")

    exec_side = None
    exec_orders: Optional[List[dict]] = None
    exec_skipped: Optional[List[str]] = None

    if args.execute or args.print_trade:
        if not args.event_ticker:
            print("Execution requires EVENT_TICKER.", file=sys.stderr)
            sys.exit(2)
        move_resting_flag = args.move_resting or args.exec_move_resting
        exec_positions: List[dict] = []
        exec_orders_list: List[dict] = []
        if trade_client is not None:
            exec_positions = trade_client.get_positions(args.event_ticker, subaccount=args.subaccount)
            exec_orders_list = trade_client.get_orders(
                args.event_ticker, status="resting", subaccount=args.subaccount
            )
        exec_pending = _pending_qty_by_ticker_side(exec_orders_list)
        exec_pending_by_price = _pending_qty_by_ticker_side_price(exec_orders_list)

        if args.exec_liquidate:
            exec_side = _select_liquidate_side(
                args.liquidate_side,
                _liquidation_summary(
                    "yes",
                    books,
                    exec_trade_state,
                    taker_model,
                    hold_payout=yes_adj.get("payout") if yes_adj and yes_adj.get("remaining_total") == 0 else None,
                ),
                _liquidation_summary(
                    "no",
                    books,
                    exec_trade_state,
                    taker_model,
                    hold_payout=no_adj.get("payout") if no_adj and no_adj.get("remaining_total") == 0 else None,
                ),
            )
            client_prefix = f"liq-{exec_side}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
            orders, skipped = _build_liquidate_orders(
                exec_side,
                books,
                exec_trade_state.get("pos_yes", {}),
                client_prefix=client_prefix,
            )
            move_cancel_ids = []
            move_replacements = []
            exec_orders = orders
            exec_skipped = skipped
            exec_liq_summary = _liquidation_summary(
                exec_side,
                books,
                exec_trade_state,
                taker_model,
                hold_payout=(
                    yes_adj.get("payout") if exec_side == "yes" and yes_adj and yes_adj.get("remaining_total") == 0
                    else no_adj.get("payout") if exec_side == "no" and no_adj and no_adj.get("remaining_total") == 0
                    else None
                ),
            )
            if args.print_trade:
                if orders:
                    print("Trade packet (liquidation sell orders):")
                    print(json.dumps({"orders": orders}, indent=2))
                    print("")
                if skipped:
                    print("Skipped legs:")
                    for item in skipped:
                        print(f"  {item}")
                    print("")
            if args.execute:
                if trade_client is None:
                    print("Trade client not available for execution.", file=sys.stderr)
                    sys.exit(2)
                if not args.confirm or args.confirm not in (args.event_ticker, "EXECUTE"):
                    print(
                        "Refusing to execute. Provide --confirm EVENT_TICKER (or EXECUTE).",
                        file=sys.stderr,
                    )
                    sys.exit(2)
                if not orders:
                    print("Refusing to execute: no liquidation orders built.", file=sys.stderr)
                    if skipped:
                        print("Skipped legs:")
                        for item in skipped:
                            print(f"  {item}")
                    sys.exit(2)
                min_edge_cents = int(round(args.exec_min_edge * 100.0))
                edge_for_check = exec_liq_summary.get("delta")
                if edge_for_check is None:
                    edge_for_check = 0
                if edge_for_check < min_edge_cents:
                    print(
                        f"Refusing to execute: liquidation edge {fmt_money(int(edge_for_check))} "
                        f"below min {fmt_money(min_edge_cents)}.",
                        file=sys.stderr,
                    )
                    sys.exit(2)
                est_proceeds = _estimate_orders_proceeds(orders, maker_model, taker_model)
                print(f"Submitting {len(orders)} liquidation orders for {exec_side.upper()} basket.")
                print(f"Estimated total proceeds (fees incl): {fmt_money(est_proceeds)}")
                resp = trade_client.place_orders_batch(orders)
                print("Trade API response (summary keys):", ", ".join(sorted(resp.keys())))
                _print_order_response(resp)
            return

        exec_side = _select_exec_side(args.exec_side, yes_eval, no_eval)
        plan_rows = yes_plan if exec_side == "yes" else no_plan
        maker_px_map = yes_maker_px if exec_side == "yes" else no_maker_px
        exec_bid_px, exec_bid_qty = _top_bid_maps(books, exec_side)
        exec_own_qty_by_price = _pending_qty_by_ticker_side_price(exec_orders_list)
        client_prefix = f"arb-{exec_side}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        orders, skipped = _build_exec_orders(
            exec_side,
            plan_rows,
            args.qty,
            only_maker=args.exec_only_maker,
            post_only=args.exec_post_only,
            client_prefix=client_prefix,
            pos_yes=exec_trade_state.get("pos_yes", {}),
            pending_qty=exec_pending,
            pending_qty_by_price=exec_pending_by_price,
            ignore_pending_for_taker=args.exec_ignore_pending_taker,
            match_pending_price=args.exec_pending_same_price,
        )
        move_cancel_ids = []
        move_replacements = []
        if args.exec_move_resting:
            move_cancel_ids, move_replacements = _build_move_resting_plan(
                exec_orders_list,
                exec_side,
                maker_px_map,
                post_only=args.exec_post_only,
                top_bid_px_by_ticker=exec_bid_px,
                top_bid_qty_by_ticker=exec_bid_qty,
                own_qty_by_price=exec_own_qty_by_price,
            )
        exec_orders = orders
        exec_skipped = skipped
        if args.print_trade:
            full_orders, _ = _build_exec_orders(
                exec_side,
                plan_rows,
                args.qty,
                only_maker=False,
                post_only=args.exec_post_only,
                client_prefix=client_prefix + "-full",
                pos_yes=exec_trade_state.get("pos_yes", {}),
                pending_qty=exec_pending,
                pending_qty_by_price=exec_pending_by_price,
                ignore_pending_for_taker=args.exec_ignore_pending_taker,
                match_pending_price=args.exec_pending_same_price,
            )
            if full_orders:
                print("Trade packet (full plan, includes taker legs):")
                print(json.dumps({"orders": full_orders}, indent=2))
                print("")
            if orders:
                print("Trade packet (executable subset):")
                print(json.dumps({"orders": orders}, indent=2))
                print("")
            if args.exec_move_resting:
                if move_cancel_ids:
                    print("Move resting: cancel order ids:")
                    for oid in move_cancel_ids:
                        print(f"  {oid}")
                if move_replacements:
                    print("Move resting: replacement orders:")
                    print(json.dumps({"orders": move_replacements}, indent=2))
                if move_cancel_ids or move_replacements:
                    print("")
            if skipped:
                print("Skipped legs:")
                for item in skipped:
                    print(f"  {item}")
                print("")

        if args.execute:
            if trade_client is None:
                print("Trade client not available for execution.", file=sys.stderr)
                sys.exit(2)
            if not args.confirm or args.confirm not in (args.event_ticker, "EXECUTE"):
                print(
                    "Refusing to execute. Provide --confirm EVENT_TICKER (or EXECUTE).",
                    file=sys.stderr,
                )
                sys.exit(2)
            if skipped and not args.exec_allow_partial:
                uncovered = [s for s in skipped if "already covered" not in s]
                if uncovered:
                    print(
                        f"Refusing to execute: {len(skipped)} legs skipped. "
                        "Use --exec-allow-partial to override.",
                        file=sys.stderr,
                    )
                    sys.exit(2)
                print("Note: skipped legs are already covered by positions/resting orders.")
            all_orders = move_replacements + orders
            if not all_orders:
                print("Refusing to execute: no orders built.", file=sys.stderr)
                if skipped:
                    print("Skipped legs:")
                    for item in skipped:
                        print(f"  {item}")
                sys.exit(2)
            if args.exec_max_orders is not None and len(all_orders) > args.exec_max_orders:
                print(
                    f"Refusing to execute: {len(all_orders)} orders exceeds --exec-max-orders.",
                    file=sys.stderr,
                )
                sys.exit(2)
            min_edge_cents = int(round(args.exec_min_edge * 100.0))
            chosen_eval = yes_eval if exec_side == "yes" else no_eval
            exec_adj = _adjusted_basket(
                exec_side,
                books,
                args.qty,
                exec_trade_state,
                taker_model=taker_model,
                maker_model=maker_model,
                payout_per_contract=yes_payout_per_contract
                if exec_side == "yes"
                else no_payout_per_contract,
                move_resting=move_resting_flag,
                maker_px_by_ticker=yes_maker_px if exec_side == "yes" else no_maker_px,
            )
            adj_edge = exec_adj.get("edge_ex_positions")
            hybrid_edge = None
            if args.exec_use_hybrid_edge:
                hybrid_edge = (
                    yes_hybrid.get("edge") if exec_side == "yes" else no_hybrid.get("edge")
                )
            edge_for_check = adj_edge if adj_edge is not None else chosen_eval.edge_cents
            if hybrid_edge is not None:
                edge_for_check = hybrid_edge
            if edge_for_check is None:
                edge_for_check = chosen_eval.edge_cents
            if edge_for_check < min_edge_cents:
                print(
                    f"Refusing to execute: edge {fmt_money(int(edge_for_check))} "
                    f"below min {fmt_money(min_edge_cents)}.",
                    file=sys.stderr,
                )
                sys.exit(2)
            est_cost = _estimate_orders_cost(all_orders, maker_model, taker_model)
            if args.exec_max_cost is not None:
                max_cost_cents = int(round(args.exec_max_cost * 100.0))
                if est_cost > max_cost_cents:
                    print(
                        f"Refusing to execute: est cost {fmt_money(est_cost)} "
                        f"exceeds max {fmt_money(max_cost_cents)}.",
                        file=sys.stderr,
                    )
                    sys.exit(2)
            print(f"Submitting {len(all_orders)} orders for {exec_side.upper()} basket.")
            print(f"Estimated total cost (fees incl): {fmt_money(est_cost)}")
            if args.exec_move_resting and move_cancel_ids:
                for oid in move_cancel_ids:
                    trade_client.cancel_order(oid)
            resp = trade_client.place_orders_batch(all_orders)
            print("Trade API response (summary keys):", ", ".join(sorted(resp.keys())))
            _print_order_response(resp)

    if args.json_path:
        out = {
            "event_ticker": args.event_ticker,
            "markets": [b.market_ticker for b in books],
            "depth": args.depth,
            "qty": args.qty,
            "fee_mode": args.fee_mode,
            "yes_winners": winners,
            "assumption": f"exactly {winners} outcome(s) resolve YES",
            "event": {
                "title": event.get("title"),
                "series_ticker": event.get("series_ticker"),
            }
            if event
            else None,
            "results": {
                "yes": yes_eval.__dict__,
                "no": no_eval.__dict__,
            },
            "limit_strategies": {
                "yes": yes_rows,
                "no": no_rows,
            },
            "hybrid_plan": {
                "yes": yes_plan,
                "no": no_plan,
            },
            "midpoint_plan": {
                "yes": yes_mid_plan,
                "no": no_mid_plan,
            },
            "hybrid_summary": {
                "yes": yes_hybrid,
                "no": no_hybrid,
            },
            "midpoint_summary": {
                "yes": yes_mid_hybrid,
                "no": no_mid_hybrid,
            },
            "trade_state": trade_state,
            "adjusted_baskets": {
                "yes": yes_adj if trade_state is not None else None,
                "no": no_adj if trade_state is not None else None,
            },
            "recommendation": _recommend_now(yes_eval, no_eval, yes_adj, no_adj)
            if trade_state is not None
            else _recommend_now(yes_eval, no_eval, None, None),
            "exec_preview": {
                "side": exec_side,
                "orders": exec_orders,
                "skipped": exec_skipped,
            },
            "dropped_markets": [{"ticker": t, "reason": r} for t, r in dropped],
            "market_top": [summarize_book(b) for b in books],
        }
        path = Path(args.json_path)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
