from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from basket import FeeModel
from client import load_client_from_env
from kxbtc_parity_math import (
    Bucket,
    Threshold,
    TradeRecipe,
    compute_bucket_from_thresholds,
    compute_box_trades,
    compute_constant_payout_trades,
    compute_threshold_from_buckets,
)
from kxbtc_parity_parse import classify_market
from normalize import normalize_orderbook

from analyze_kxbtc_parity import (  # reuse helpers (no execution)
    MarketQuote,
    Leg,
    _build_leg_from_spec,
    _derive_other_ticker,
    _quote_from_book,
)


@dataclass(frozen=True)
class MakerPlan:
    trade_type: str
    K: Optional[int]
    delta: Optional[int]
    maker_leg: Leg
    maker_price: int
    edge_if_filled: float
    cross_edge: float
    payout_per_contract: int
    notes: List[str] = field(default_factory=list)
    hedge_legs: List[Leg] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parity maker bot (dry-run planner)")
    p.add_argument("--event_time", help="Event time, e.g., 26FEB2317")
    p.add_argument("--event_kxbtc", help="Full event ticker for base series")
    p.add_argument("--event_kxbtcd", help="Full event ticker for D series")
    p.add_argument("--min_edge_cents", type=float, default=1.0)
    p.add_argument(
        "--maker_buffer_cents",
        type=float,
        default=1.0,
        help="Extra edge buffer required for maker quotes (default 1.0c).",
    )
    p.add_argument(
        "--maker_target_edge_cents",
        type=float,
        default=None,
        help="Absolute edge target for maker-filled plans (overrides min_edge+buffer if set).",
    )
    p.add_argument("--max_trades", type=int, default=10)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument(
        "--require_exact_alignment",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    p.add_argument(
        "--maker_only_buy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only consider BUY legs for maker quotes (default true).",
    )
    p.add_argument(
        "--maker_price_mode",
        choices=["inside", "inside_or_touch"],
        default="inside",
        help="Maker price search mode: inside spread only, or inside then touch bid/ask.",
    )
    return p.parse_args()


def _cross_price(leg: Leg, quote: MarketQuote) -> Optional[int]:
    if leg.side == "YES":
        return quote.yes_ask_px if leg.action == "BUY" else quote.yes_bid_px
    return quote.no_ask_px if leg.action == "BUY" else quote.no_bid_px


def _edge_with_maker(
    legs: List[Leg],
    quotes: Dict[str, MarketQuote],
    maker_idx: int,
    maker_price: int,
    maker_fee: FeeModel,
    taker_fee: FeeModel,
    payout_per_contract: int,
) -> Optional[float]:
    cost = 0.0
    proceeds = 0.0
    for idx, leg in enumerate(legs):
        quote = quotes.get(leg.ticker)
        if quote is None:
            return None
        if idx == maker_idx:
            price = maker_price
            fee_model = maker_fee
        else:
            price = _cross_price(leg, quote)
            fee_model = taker_fee
        if price is None:
            return None
        fee = fee_model.fee_cents(int(price), 1)
        if leg.action == "BUY":
            cost += price + fee
        else:
            proceeds += price - fee
    return proceeds - cost + payout_per_contract


def _max_maker_price_for_buy(leg: Leg, quote: MarketQuote) -> Optional[int]:
    if leg.side == "YES":
        ask = quote.yes_ask_px
        bid = quote.yes_bid_px
    else:
        ask = quote.no_ask_px
        bid = quote.no_bid_px
    if ask is not None and ask > 0:
        return max(0, ask - 1)
    if bid is not None:
        return bid
    return None


def _maker_price_candidates(
    leg: Leg,
    quote: MarketQuote,
    mode: str,
) -> List[int]:
    if leg.side == "YES":
        bid = quote.yes_bid_px
        ask = quote.yes_ask_px
    else:
        bid = quote.no_bid_px
        ask = quote.no_ask_px
    if bid is None or ask is None:
        return []
    if leg.action == "BUY":
        inside = list(range(bid + 1, max(bid + 1, ask)))
        if mode == "inside":
            return inside
        if inside:
            return inside + [bid]
        return [bid]
    # SELL leg
    inside = list(range(ask - 1, min(ask - 1, bid), -1))
    if mode == "inside":
        return inside
    if inside:
        return inside + [ask]
    return [ask]


def main() -> int:
    args = parse_args()
    target_edge = args.min_edge_cents + args.maker_buffer_cents
    if args.maker_target_edge_cents is not None:
        target_edge = args.maker_target_edge_cents

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
    _, kxbtc_markets = client.get_event_markets(event_kxbtc)
    if not kxbtc_markets:
        print(f"Error: No markets found for {event_kxbtc}.", file=sys.stderr)
        return 1
    _, kxbtcd_markets = client.get_event_markets(event_kxbtcd)
    if not kxbtcd_markets:
        print("No matching event found in other series.", file=sys.stderr)
        return 1

    quotes: Dict[str, MarketQuote] = {}
    buckets: List[Bucket] = []
    thresholds: List[Threshold] = []

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
                tradable = False
        if not args.allow_crossed and quote.yes_bid_px is not None and quote.yes_ask_px is not None:
            if quote.yes_bid_px > quote.yes_ask_px:
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
                tradable = False
        if not args.allow_crossed and quote.yes_bid_px is not None and quote.yes_ask_px is not None:
            if quote.yes_bid_px > quote.yes_ask_px:
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

    _, trades_threshold = compute_threshold_from_buckets(
        buckets,
        thresholds,
        args.require_exact_alignment,
        args.min_edge_cents,
        missing_bucket_cents=args.missing_bucket_cents,
        stop_at_stale=args.stop_at_stale,
    )
    _, trades_buckets = compute_bucket_from_thresholds(
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

    trade_recipes: List[TradeRecipe] = (
        trades_threshold + trades_buckets + trades_boxes + trades_constant
    )

    taker_fee = FeeModel("taker")
    maker_fee = FeeModel("maker")
    plans: List[MakerPlan] = []

    for recipe in trade_recipes:
        notes = recipe.notes[:]
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

        # Compute baseline cross edge (taker on all legs).
        base_edge = _edge_with_maker(
            legs,
            quotes,
            maker_idx=-1,
            maker_price=0,
            maker_fee=taker_fee,
            taker_fee=taker_fee,
            payout_per_contract=recipe.payout_per_contract,
        )
        if base_edge is None:
            continue

        for idx, leg in enumerate(legs):
            if args.maker_only_buy and leg.action != "BUY":
                continue
            quote = quotes.get(leg.ticker)
            if quote is None:
                continue
            candidates = _maker_price_candidates(leg, quote, args.maker_price_mode)
            if not candidates:
                continue
            found_price = None
            found_edge = None
            for px in candidates:
                edge = _edge_with_maker(
                    legs,
                    quotes,
                    maker_idx=idx,
                    maker_price=px,
                    maker_fee=maker_fee,
                    taker_fee=taker_fee,
                    payout_per_contract=recipe.payout_per_contract,
                )
                if edge is None:
                    continue
                if edge >= target_edge:
                    found_price = px
                    found_edge = edge
                    break
            if found_price is None or found_edge is None:
                continue
            hedge_legs = [l for j, l in enumerate(legs) if j != idx]
            plans.append(
                MakerPlan(
                    trade_type=recipe.trade_type,
                    K=recipe.K,
                    delta=recipe.delta,
                    maker_leg=leg,
                    maker_price=int(found_price),
                    edge_if_filled=float(found_edge),
                    cross_edge=float(base_edge),
                    payout_per_contract=recipe.payout_per_contract,
                    notes=notes,
                    hedge_legs=hedge_legs,
                )
            )

    plans.sort(key=lambda p: p.edge_if_filled, reverse=True)

    print(f"Event: {event_kxbtc} vs {event_kxbtcd}")
    print(f"Buckets: {len(buckets)} | Thresholds: {len(thresholds)}")
    print(f"Target edge (maker filled): {target_edge:.2f}c")
    print(f"Maker price mode: {args.maker_price_mode}")
    print(f"Plans found: {len(plans)}")
    print()

    if not plans:
        print("No maker plans meet target edge.")
        return 0

    for idx, plan in enumerate(plans[: max(0, args.max_trades)], start=1):
        payout = plan.payout_per_contract
        payout_s = f"{payout}c" if payout else "--"
        print(
            f"{idx}. {plan.trade_type} K={plan.K} Δ={plan.delta} "
            f"maker={plan.maker_leg.action} {plan.maker_leg.side} "
            f"{plan.maker_leg.ticker} px={plan.maker_price:02d}c "
            f"edge_if_filled={plan.edge_if_filled:.2f}c "
            f"cross_edge={plan.cross_edge:.2f}c payout={payout_s}"
        )
        print("   Hedge legs (taker on fill):")
        for leg in plan.hedge_legs:
            q = quotes.get(leg.ticker)
            px = _cross_price(leg, q) if q else None
            px_s = "--" if px is None else f"{px:02d}c"
            print(f"   - {leg.action} {leg.side} {leg.ticker} @ {px_s}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
