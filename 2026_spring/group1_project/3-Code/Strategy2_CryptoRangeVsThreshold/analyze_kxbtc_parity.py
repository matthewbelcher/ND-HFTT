from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from basket import FeeModel
from client import load_client_from_env
from kxbtc_parity_math import (
    Bucket,
    Threshold,
    TradeRecipe,
    bucket_sum_check,
    compute_bucket_from_thresholds,
    compute_constant_payout_trades,
    compute_box_trades,
    compute_threshold_from_buckets,
    negative_implied_bucket_check,
    threshold_monotone_check,
)
from kxbtc_parity_parse import classify_market
from normalize import NormalizedBook, normalize_orderbook

PriceLevel = Tuple[int, int]


@dataclass(frozen=True)
class MarketQuote:
    ticker: str
    yes_bid_px: Optional[int]
    yes_bid_qty: Optional[int]
    yes_ask_px: Optional[int]
    yes_ask_qty: Optional[int]
    no_bid_px: Optional[int]
    no_bid_qty: Optional[int]
    no_ask_px: Optional[int]
    no_ask_qty: Optional[int]
    yes_bid_levels: List[PriceLevel]
    yes_ask_levels: List[PriceLevel]
    no_bid_levels: List[PriceLevel]
    no_ask_levels: List[PriceLevel]
    yes_mid: Optional[float]
    no_mid: Optional[float]


@dataclass(frozen=True)
class Leg:
    ticker: str
    side: str  # YES|NO
    action: str  # BUY|SELL
    limit_px_used: int
    qty: int
    top_size_used: int
    px_source: str  # bid|ask|mid


@dataclass(frozen=True)
class CandidateTrade:
    trade_type: str
    K: Optional[int]
    delta: Optional[int]
    direction: str
    legs: List[Leg]
    edge_cross_cents: int
    edge_mid_cents: Optional[float]
    slip_cents: Optional[float]
    max_size_1lvl: int
    max_size_at_edge: int
    notes: List[str] = field(default_factory=list)
    fee_model: str = "taker"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KXBTC vs KXBTCD parity/arb analyzer")
    p.add_argument("--event_time", help="Event time, e.g., 26FEB2317")
    p.add_argument("--event_kxbtc", help="Full event ticker for KXBTC")
    p.add_argument("--event_kxbtcd", help="Full event ticker for KXBTCD")
    p.add_argument("--mode", choices=["mid", "cross"], default="cross")
    p.add_argument(
        "--fee_model",
        choices=["maker", "taker", "both", "none"],
        default="taker",
    )
    p.add_argument("--min_edge_cents", type=float, default=1.0)
    p.add_argument("--max_legs", type=int, default=10)
    p.add_argument(
        "--require_exact_alignment",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--slippage_qty", type=int, default=10)
    p.add_argument("--top_legs", type=int, default=5)
    p.add_argument(
        "--missing_bucket_cents",
        type=float,
        default=0.1,
        help="Assume missing bucket probability in cents for implied sums (default 0.1).",
    )
    p.add_argument(
        "--stop_at_stale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When building bucket baskets, stop at first stale bucket (treat as barrier).",
    )
    p.add_argument(
        "--near_edge_cents",
        type=float,
        default=None,
        help="If set, include near-arb trades with edge >= -near_edge_cents.",
    )
    p.add_argument(
        "--allow_short_yes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow SELL YES even without inventory (default false; Kalshi requires ownership).",
    )
    p.add_argument(
        "--require_two_sided",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require YES bid+ask on markets (skip one-sided books).",
    )
    p.add_argument(
        "--allow_crossed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow crossed YES markets (bid > ask). Default false skips crossed.",
    )
    p.add_argument(
        "--const_payout_min",
        type=int,
        default=2,
        help="Minimum constant payout basket dollars to search (default 2).",
    )
    p.add_argument(
        "--const_payout_max",
        type=int,
        default=5,
        help="Maximum constant payout basket dollars to search (default 5).",
    )
    p.add_argument("--json", dest="json_path")
    return p.parse_args()


def _mid_price(bid: Optional[int], ask: Optional[int]) -> Optional[float]:
    if bid is None or ask is None:
        return None
    return (float(bid) + float(ask)) / 2.0


def _quote_from_book(ticker: str, book: NormalizedBook) -> MarketQuote:
    yes_bid = book.yes_bid[0] if book.yes_bid else None
    yes_bid_qty = book.yes_bid[1] if book.yes_bid else None
    yes_ask = book.yes_ask[0] if book.yes_ask else None
    yes_ask_qty = book.yes_ask[1] if book.yes_ask else None
    no_bid = book.no_bid[0] if book.no_bid else None
    no_bid_qty = book.no_bid[1] if book.no_bid else None
    no_ask = book.no_ask[0] if book.no_ask else None
    no_ask_qty = book.no_ask[1] if book.no_ask else None
    return MarketQuote(
        ticker=ticker,
        yes_bid_px=yes_bid,
        yes_bid_qty=yes_bid_qty,
        yes_ask_px=yes_ask,
        yes_ask_qty=yes_ask_qty,
        no_bid_px=no_bid,
        no_bid_qty=no_bid_qty,
        no_ask_px=no_ask,
        no_ask_qty=no_ask_qty,
        yes_bid_levels=book.yes_bid_levels,
        yes_ask_levels=book.yes_ask_levels,
        no_bid_levels=book.no_bid_levels,
        no_ask_levels=book.no_ask_levels,
        yes_mid=_mid_price(yes_bid, yes_ask),
        no_mid=_mid_price(no_bid, no_ask),
    )


def _make_fee_models(fee_model: str) -> List[Tuple[str, FeeModel]]:
    if fee_model == "both":
        return [("maker", FeeModel("maker")), ("taker", FeeModel("taker"))]
    return [(fee_model, FeeModel(fee_model))]


def _fmt_px(px: Optional[int]) -> str:
    return "--" if px is None else f"{px:02d}"


def _fmt_qty(qty: Optional[int]) -> str:
    if qty is None:
        return "--"
    return f"{int(qty)}"


def _fmt_spread(bid: Optional[int], ask: Optional[int]) -> str:
    if bid is None or ask is None:
        return "--"
    return f"{ask - bid:02d}"


def _fmt_float(v: Optional[float], *, prec: int = 2) -> str:
    if v is None:
        return "--"
    return f"{v:.{prec}f}"


def _bucket_key(b: Bucket) -> float:
    if b.raw_low is not None:
        return b.raw_low
    if b.raw_high is not None:
        return b.raw_high
    return 0.0


def _threshold_key(t: Threshold) -> float:
    if t.strike_raw is not None:
        return t.strike_raw
    if t.strike_norm is not None:
        return float(t.strike_norm)
    return 0.0


def _print_bbo_table(
    title: str,
    tickers: List[str],
    quotes: Dict[str, MarketQuote],
) -> None:
    if not tickers:
        return
    print(title)
    header = (
        f"{'ticker':<24} {'y_bid':>5} {'y_bq':>5} {'y_ask':>5} {'y_aq':>5} "
        f"{'y_sp':>5} {'y_mid':>7} | "
        f"{'n_bid':>5} {'n_bq':>5} {'n_ask':>5} {'n_aq':>5} {'n_sp':>5} {'n_mid':>7}"
    )
    print(header)
    print("-" * len(header))
    for ticker in tickers:
        q = quotes.get(ticker)
        if q is None:
            continue
        print(
            f"{ticker:<24} "
            f"{_fmt_px(q.yes_bid_px):>5} {_fmt_qty(q.yes_bid_qty):>5} "
            f"{_fmt_px(q.yes_ask_px):>5} {_fmt_qty(q.yes_ask_qty):>5} "
            f"{_fmt_spread(q.yes_bid_px, q.yes_ask_px):>5} {_fmt_float(q.yes_mid, prec=2):>7} | "
            f"{_fmt_px(q.no_bid_px):>5} {_fmt_qty(q.no_bid_qty):>5} "
            f"{_fmt_px(q.no_ask_px):>5} {_fmt_qty(q.no_ask_qty):>5} "
            f"{_fmt_spread(q.no_bid_px, q.no_ask_px):>5} {_fmt_float(q.no_mid, prec=2):>7}"
        )
    print()


def _fetch_event(client, ticker: str) -> Tuple[Optional[dict], List[dict]]:
    event, markets = client.get_event_markets(ticker)
    if not markets:
        return None, []
    return event, markets


def _derive_other_ticker(ticker: str) -> Optional[str]:
    if "-" not in ticker:
        return None
    series, rest = ticker.split("-", 1)
    if not series:
        return None
    if series.endswith("D"):
        other_series = series[:-1]
    else:
        other_series = series + "D"
    if not other_series:
        return None
    return f"{other_series}-{rest}"


def _get_levels_for_leg(leg: Leg, quote: MarketQuote) -> List[PriceLevel]:
    if leg.side == "YES":
        return quote.yes_ask_levels if leg.action == "BUY" else quote.yes_bid_levels
    return quote.no_ask_levels if leg.action == "BUY" else quote.no_bid_levels


def _buy_cost_from_levels(
    levels: List[PriceLevel],
    qty: int,
    fee_model: FeeModel,
) -> Optional[int]:
    remaining = qty
    cost = 0
    for price, level_qty in levels:
        if remaining <= 0:
            break
        take = min(remaining, level_qty)
        if take <= 0:
            continue
        cost += price * take + fee_model.fee_cents(price, take)
        remaining -= take
    if remaining > 0:
        return None
    return cost


def _sell_proceeds_from_levels(
    levels: List[PriceLevel],
    qty: int,
    fee_model: FeeModel,
) -> Optional[int]:
    remaining = qty
    proceeds = 0
    for price, level_qty in levels:
        if remaining <= 0:
            break
        take = min(remaining, level_qty)
        if take <= 0:
            continue
        proceeds += price * take - fee_model.fee_cents(price, take)
        remaining -= take
    if remaining > 0:
        return None
    return proceeds


def _edge_for_legs(
    legs: List[Leg],
    quotes: Dict[str, MarketQuote],
    fee_model: FeeModel,
    mode: str,
    *,
    size: int = 1,
    use_levels: bool = False,
    payout_per_contract: int = 0,
) -> Optional[float]:
    cost = 0.0
    proceeds = 0.0
    for leg in legs:
        quote = quotes.get(leg.ticker)
        if quote is None:
            return None
        if use_levels:
            levels = _get_levels_for_leg(leg, quote)
            if not levels:
                return None
            if leg.action == "BUY":
                c = _buy_cost_from_levels(levels, size, fee_model)
                if c is None:
                    return None
                cost += c
            else:
                p = _sell_proceeds_from_levels(levels, size, fee_model)
                if p is None:
                    return None
                proceeds += p
            continue

        if mode == "mid":
            price = quote.yes_mid if leg.side == "YES" else quote.no_mid
        else:
            price = float(leg.limit_px_used)
        if price is None:
            return None
        fee = fee_model.fee_cents(int(round(price)), size)
        if leg.action == "BUY":
            cost += price * size + fee
        else:
            proceeds += price * size - fee
    return proceeds - cost + (payout_per_contract * size)


def _max_size_at_edge(
    legs: List[Leg],
    quotes: Dict[str, MarketQuote],
    fee_model: FeeModel,
    min_edge_cents: float,
    payout_per_contract: int = 0,
) -> int:
    max_sizes: List[int] = []
    for leg in legs:
        quote = quotes.get(leg.ticker)
        if quote is None:
            return 0
        levels = _get_levels_for_leg(leg, quote)
        if not levels:
            return 0
        max_sizes.append(sum(q for _, q in levels))
    if not max_sizes:
        return 0
    max_size = min(max_sizes)
    if max_size <= 0:
        return 0
    for size in range(1, max_size + 1):
        edge = _edge_for_legs(
            legs,
            quotes,
            fee_model,
            "cross",
            size=size,
            use_levels=True,
            payout_per_contract=payout_per_contract,
        )
        if edge is None:
            return size - 1
        if edge < min_edge_cents:
            return size - 1
    return max_size


def _build_leg_from_spec(
    spec: "LegSpec",
    quote: MarketQuote,
    allow_short_yes: bool,
    *,
    notes: List[str],
) -> Optional[Leg]:
    if spec.action == "BUY":
        if spec.side == "YES":
            if quote.yes_ask_px is None or (quote.yes_ask_qty or 0) <= 0:
                return None
            return Leg(
                ticker=quote.ticker,
                side="YES",
                action="BUY",
                limit_px_used=quote.yes_ask_px,
                qty=1,
                top_size_used=int(quote.yes_ask_qty or 0),
                px_source="ask",
            )
        if quote.no_ask_px is None or (quote.no_ask_qty or 0) <= 0:
            return None
        return Leg(
            ticker=quote.ticker,
            side="NO",
            action="BUY",
            limit_px_used=quote.no_ask_px,
            qty=1,
            top_size_used=int(quote.no_ask_qty or 0),
            px_source="ask",
        )

    # SELL YES default; fallback to BUY NO if shorting not allowed or no bid
    if spec.side == "YES":
        if allow_short_yes and quote.yes_bid_px is not None and (quote.yes_bid_qty or 0) > 0:
            return Leg(
                ticker=quote.ticker,
                side="YES",
                action="SELL",
                limit_px_used=quote.yes_bid_px,
                qty=1,
                top_size_used=int(quote.yes_bid_qty or 0),
                px_source="bid",
            )
        if quote.no_ask_px is None or (quote.no_ask_qty or 0) <= 0:
            return None
        notes.append(f"fallback_buy_no:{quote.ticker}")
        return Leg(
            ticker=quote.ticker,
            side="NO",
            action="BUY",
            limit_px_used=quote.no_ask_px,
            qty=1,
            top_size_used=int(quote.no_ask_qty or 0),
            px_source="ask",
        )

    if quote.no_bid_px is None or (quote.no_bid_qty or 0) <= 0:
        return None
    return Leg(
        ticker=quote.ticker,
        side="NO",
        action="SELL",
        limit_px_used=quote.no_bid_px,
        qty=1,
        top_size_used=int(quote.no_bid_qty or 0),
        px_source="bid",
    )


def main() -> int:
    args = parse_args()

    event_kxbtc = args.event_kxbtc
    event_kxbtcd = args.event_kxbtcd
    if args.event_time and not event_kxbtc and not event_kxbtcd:
        event_kxbtc = f"KXBTC-{args.event_time}"
        event_kxbtcd = f"KXBTCD-{args.event_time}"
    elif args.event_time and (event_kxbtc or event_kxbtcd):
        print("Warning: --event_time ignored because explicit event tickers were provided.", file=sys.stderr)

    if not event_kxbtc and not event_kxbtcd:
        print("Error: Provide --event_time or at least one of --event_kxbtc/--event_kxbtcd.", file=sys.stderr)
        return 2

    if event_kxbtc and not event_kxbtcd:
        event_kxbtcd = _derive_other_ticker(event_kxbtc)
    if event_kxbtcd and not event_kxbtc:
        event_kxbtc = _derive_other_ticker(event_kxbtcd)

    if not event_kxbtc or not event_kxbtcd:
        print("Error: Unable to derive missing event ticker.", file=sys.stderr)
        return 2

    client = load_client_from_env()
    _, kxbtc_markets = _fetch_event(client, event_kxbtc)
    if not kxbtc_markets:
        print(f"Error: No markets found for {event_kxbtc}.", file=sys.stderr)
        return 1
    _, kxbtcd_markets = _fetch_event(client, event_kxbtcd)
    if not kxbtcd_markets:
        print("No matching event found in other series.", file=sys.stderr)
        return 1

    quotes: Dict[str, MarketQuote] = {}
    buckets: List[Bucket] = []
    thresholds: List[Threshold] = []
    parse_warnings: List[str] = []
    skipped_one_sided: List[str] = []
    skipped_crossed: List[str] = []

    for m in kxbtc_markets:
        ticker = m.get("ticker")
        if not ticker:
            continue
        classification = classify_market(m, prefer="bucket")
        if classification.kind != "bucket" or not classification.range_parse:
            continue
        try:
            ob = client.get_orderbook(ticker, depth=args.depth)
        except Exception:
            continue
        book = normalize_orderbook(ticker, ob, depth=args.depth, market_snapshot=m)
        quote = _quote_from_book(ticker, book)
        tradable = True
        if args.require_two_sided:
            if (
                quote.yes_bid_px is None
                or quote.yes_ask_px is None
                or (quote.yes_bid_qty or 0) <= 0
                or (quote.yes_ask_qty or 0) <= 0
            ):
                skipped_one_sided.append(ticker)
                tradable = False
        if not args.allow_crossed and quote.yes_bid_px is not None and quote.yes_ask_px is not None:
            if quote.yes_bid_px > quote.yes_ask_px:
                skipped_crossed.append(ticker)
                tradable = False
        if tradable:
            quotes[ticker] = quote
        rp = classification.range_parse
        buckets.append(
            Bucket(
                ticker=ticker,
                raw_low=rp.raw_low,
                raw_high=rp.raw_high,
                norm_low=rp.norm_low,
                norm_high=rp.norm_high,
                yes_mid=quote.yes_mid if tradable else None,
            )
        )
        parse_warnings.extend(classification.warnings)

    for m in kxbtcd_markets:
        ticker = m.get("ticker")
        if not ticker:
            continue
        classification = classify_market(m, prefer="threshold")
        if classification.kind != "threshold" or not classification.threshold_parse:
            continue
        try:
            ob = client.get_orderbook(ticker, depth=args.depth)
        except Exception:
            continue
        book = normalize_orderbook(ticker, ob, depth=args.depth, market_snapshot=m)
        quote = _quote_from_book(ticker, book)
        tradable = True
        if args.require_two_sided:
            if (
                quote.yes_bid_px is None
                or quote.yes_ask_px is None
                or (quote.yes_bid_qty or 0) <= 0
                or (quote.yes_ask_qty or 0) <= 0
            ):
                skipped_one_sided.append(ticker)
                tradable = False
        if not args.allow_crossed and quote.yes_bid_px is not None and quote.yes_ask_px is not None:
            if quote.yes_bid_px > quote.yes_ask_px:
                skipped_crossed.append(ticker)
                tradable = False
        if tradable:
            quotes[ticker] = quote
        tp = classification.threshold_parse
        if tradable:
            thresholds.append(
                Threshold(
                    ticker=ticker,
                    strike_raw=tp.raw_strike,
                    strike_norm=tp.norm_strike,
                    yes_mid=quote.yes_mid,
                )
            )
        parse_warnings.extend(classification.warnings)

    implied_thresholds, trades_threshold = compute_threshold_from_buckets(
        buckets,
        thresholds,
        args.require_exact_alignment,
        args.min_edge_cents,
        missing_bucket_cents=args.missing_bucket_cents,
        stop_at_stale=args.stop_at_stale,
    )
    implied_buckets, trades_buckets = compute_bucket_from_thresholds(
        buckets,
        thresholds,
        args.min_edge_cents,
        missing_bucket_cents=args.missing_bucket_cents,
    )
    trades_boxes = compute_box_trades(buckets, thresholds)
    trades_constant = compute_constant_payout_trades(
        buckets,
        thresholds,
        min_payout_dollars=args.const_payout_min,
        max_payout_dollars=args.const_payout_max,
    )

    trade_recipes = trades_threshold + trades_buckets + trades_boxes + trades_constant

    fee_models = _make_fee_models(args.fee_model)
    candidates: List[CandidateTrade] = []

    for recipe in trade_recipes:
        for fee_label, fee_model in fee_models:
            notes = recipe.notes[:]
            notes.append(f"fee={fee_label}")
            legs: List[Leg] = []
            ok = True
            for spec in recipe.legs:
                quote = quotes.get(spec.ticker)
                if quote is None:
                    ok = False
                    break
                leg = _build_leg_from_spec(spec, quote, args.allow_short_yes, notes=notes)
                if leg is None:
                    ok = False
                    break
                legs.append(leg)
            if not ok or not legs:
                continue
            if len(legs) > args.max_legs:
                continue

            edge_cross = _edge_for_legs(
                legs,
                quotes,
                fee_model,
                "cross",
                payout_per_contract=recipe.payout_per_contract,
            )
            if edge_cross is None:
                continue
            is_near = False
            if edge_cross < args.min_edge_cents:
                if args.near_edge_cents is None or edge_cross < -args.near_edge_cents:
                    continue
                is_near = True
            edge_mid = _edge_for_legs(
                legs,
                quotes,
                fee_model,
                "mid",
                payout_per_contract=recipe.payout_per_contract,
            )
            edge_cross_slip = _edge_for_legs(
                legs,
                quotes,
                fee_model,
                "cross",
                size=max(1, args.slippage_qty),
                use_levels=True,
                payout_per_contract=recipe.payout_per_contract,
            )
            slip = None
            if edge_cross_slip is not None:
                slip = edge_cross - edge_cross_slip

            max_size_1lvl = min((leg.top_size_used for leg in legs), default=0)
            max_size_at_edge = _max_size_at_edge(
                legs,
                quotes,
                fee_model,
                args.min_edge_cents,
                payout_per_contract=recipe.payout_per_contract,
            )

            if is_near:
                notes.append("near")
            candidates.append(
                CandidateTrade(
                    trade_type=recipe.trade_type,
                    K=recipe.K,
                    delta=recipe.delta,
                    direction=recipe.direction,
                    legs=legs,
                    edge_cross_cents=int(round(edge_cross)),
                    edge_mid_cents=edge_mid,
                    slip_cents=slip,
                    max_size_1lvl=max_size_1lvl,
                    max_size_at_edge=max_size_at_edge,
                    notes=notes,
                    fee_model=fee_label,
                )
            )

    candidates.sort(key=lambda x: x.edge_cross_cents, reverse=True)

    print(f"Event: {event_kxbtc} vs {event_kxbtcd}")
    print(f"Buckets: {len(buckets)} | Thresholds: {len(thresholds)}")
    print(f"Mode: {args.mode} | Fee model: {args.fee_model}")
    print(f"Min edge (cross): {args.min_edge_cents}c")
    print()

    if skipped_one_sided:
        print(f"Skipped one-sided markets: {len(skipped_one_sided)}")
        print()
    if skipped_crossed:
        print(f"Skipped crossed markets: {len(skipped_crossed)}")
        print()

    if parse_warnings:
        print("Warnings:")
        for w in sorted(set(parse_warnings)):
            print(f"  - {w}")
        print()

    bucket_tickers = [b.ticker for b in sorted(buckets, key=_bucket_key)]
    threshold_tickers = [t.ticker for t in sorted(thresholds, key=_threshold_key)]
    _print_bbo_table("BBOs (Buckets)", bucket_tickers, quotes)
    _print_bbo_table("BBOs (Thresholds)", threshold_tickers, quotes)

    print("Implied thresholds (from buckets):")
    th_header = (
        f"{'strike':>8} {'implied':>8} {'actual':>8} {'diff':>8} "
        f"{'align':>6} {'partial':>7} {'lb':>8} {'ub':>8}  notes"
    )
    print(th_header)
    print("-" * len(th_header))
    for row in implied_thresholds:
        strike = row.strike_norm if row.strike_norm is not None else row.strike_raw
        strike_s = "--" if strike is None else str(strike)
        notes = ",".join(row.notes) if row.notes else ""
        print(
            f"{strike_s:>8} {_fmt_float(row.implied_yes_geK, prec=2):>8} "
            f"{_fmt_float(row.actual_yes, prec=2):>8} {_fmt_float(row.diff, prec=2):>8} "
            f"{str(row.alignment):>6} {str(row.partial_bucket_risk):>7} "
            f"{_fmt_float(row.lower_bound, prec=2):>8} {_fmt_float(row.upper_bound, prec=2):>8}  {notes}"
        )
    print()

    print("Implied buckets (from thresholds):")
    b_header = f"{'low':>8} {'high':>8} {'implied':>8} {'actual':>8} {'diff':>8}  notes"
    print(b_header)
    print("-" * len(b_header))
    for row in implied_buckets:
        low_s = "--" if row.low is None else str(row.low)
        high_s = "--" if row.high is None else str(row.high)
        notes = ",".join(row.notes) if row.notes else ""
        print(
            f"{low_s:>8} {high_s:>8} "
            f"{_fmt_float(row.implied_bucket, prec=2):>8} {_fmt_float(row.actual_yes, prec=2):>8} "
            f"{_fmt_float(row.diff, prec=2):>8}  {notes}"
        )
    print()

    if candidates:
        header = (
            f"{'trade_type':<18} {'K':>7} {'Δ':>5} {'direction':<26} "
            f"{'legs':>4} {'edge_x':>7} {'edge_mid':>9} {'slip@N':>8} "
            f"{'max1':>5} {'max_edge':>8}  notes"
        )
        print(header)
        print("-" * len(header))
        for c in candidates:
            edge_mid_str = "--" if c.edge_mid_cents is None else f"{c.edge_mid_cents:7.2f}"
            slip_str = "--" if c.slip_cents is None else f"{c.slip_cents:6.1f}"
            notes = ",".join(c.notes) if c.notes else ""
            print(
                f"{c.trade_type:<18} {str(c.K or '--'):>7} {str(c.delta or '--'):>5} "
                f"{c.direction:<26} {len(c.legs):>4} {c.edge_cross_cents:7d} "
                f"{edge_mid_str:>9} {slip_str:>8} {c.max_size_1lvl:5d} "
                f"{c.max_size_at_edge:8d}  {notes}"
            )
        print()

        top_n = max(0, args.top_legs)
        if top_n > 0:
            print(f"Top {min(top_n, len(candidates))} trades (legs):")
            for idx, c in enumerate(candidates[:top_n], start=1):
                print(
                    f"{idx}. {c.trade_type} K={c.K} Δ={c.delta} "
                    f"edge_x={c.edge_cross_cents}c fee={c.fee_model}"
                )
                for leg in c.legs:
                    quote = quotes.get(leg.ticker)
                    fee = 0
                    if quote:
                        fee = FeeModel(c.fee_model).fee_cents(leg.limit_px_used, 1)
                    print(
                        f"   {leg.action:>4} {leg.side:<3} {leg.ticker:<22} "
                        f"px={leg.limit_px_used:02d}c ({leg.px_source}) "
                        f"top={leg.top_size_used} fee={fee}c"
                    )
            print()
    else:
        print("No trades meet the executable edge threshold.")
        print()

    bucket_sum, bucket_missing = bucket_sum_check(
        buckets, missing_bucket_cents=args.missing_bucket_cents
    )
    if bucket_sum is not None:
        miss_note = f" (missing assumed={args.missing_bucket_cents}c x{bucket_missing})"
        if bucket_missing <= 0:
            miss_note = ""
        print(f"Bucket sum deviation (mid): {bucket_sum:+.2f}c{miss_note}")
    monotone_viol = threshold_monotone_check(thresholds, args.min_edge_cents)
    if monotone_viol:
        print("Threshold monotonicity violations:")
        for k1, k2, p1, p2 in monotone_viol:
            print(f"  YES(>={k1})={p1:.2f} < YES(>={k2})={p2:.2f}")
    neg_implied = negative_implied_bucket_check(thresholds, args.min_edge_cents)
    if neg_implied:
        print("Negative implied bucket alerts:")
        for k1, k2, implied in neg_implied:
            print(f"  implied bucket [{k1},{k2}) = {implied:.2f}c")

    if args.json_path:
        payload = {
            "event_kxbtc": event_kxbtc,
            "event_kxbtcd": event_kxbtcd,
            "mode": args.mode,
            "fee_model": args.fee_model,
            "min_edge_cents": args.min_edge_cents,
            "missing_bucket_cents": args.missing_bucket_cents,
            "stop_at_stale": args.stop_at_stale,
            "const_payout_min": args.const_payout_min,
            "const_payout_max": args.const_payout_max,
            "buckets": [asdict(b) for b in buckets],
            "thresholds": [asdict(t) for t in thresholds],
            "implied_thresholds": [asdict(row) for row in implied_thresholds],
            "implied_buckets": [asdict(row) for row in implied_buckets],
            "trades": [asdict(c) for c in candidates],
            "alerts": {
                "bucket_sum_deviation": bucket_sum,
                "threshold_monotone": monotone_viol,
                "negative_implied_bucket": neg_implied,
                "parse_warnings": sorted(set(parse_warnings)),
                "skipped_one_sided": sorted(set(skipped_one_sided)),
                "skipped_crossed": sorted(set(skipped_crossed)),
            },
        }
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
