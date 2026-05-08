from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Bucket:
    ticker: str
    raw_low: Optional[float]
    raw_high: Optional[float]
    norm_low: Optional[int]
    norm_high: Optional[int]
    yes_mid: Optional[float]


@dataclass(frozen=True)
class Threshold:
    ticker: str
    strike_raw: Optional[float]
    strike_norm: Optional[int]
    yes_mid: Optional[float]


@dataclass(frozen=True)
class LegSpec:
    ticker: str
    action: str  # "BUY" or "SELL"
    side: str  # "YES"
    role: str


@dataclass(frozen=True)
class TradeRecipe:
    trade_type: str
    K: Optional[int]
    delta: Optional[int]
    direction: str
    legs: List[LegSpec]
    implied: Optional[float]
    actual: Optional[float]
    diff: Optional[float]
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    partial_bucket_risk: bool
    alignment: bool
    payout_per_contract: int = 0
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ThresholdImplied:
    ticker: str
    strike_norm: Optional[int]
    strike_raw: Optional[float]
    implied_yes_geK: Optional[float]
    actual_yes: Optional[float]
    diff: Optional[float]
    alignment: bool
    partial_bucket_risk: bool
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class BucketImplied:
    ticker: str
    low: Optional[int]
    high: Optional[int]
    implied_bucket: Optional[float]
    actual_yes: Optional[float]
    diff: Optional[float]
    notes: List[str] = field(default_factory=list)


def _sum_mid(
    buckets: List[Bucket],
    missing_bucket_cents: Optional[float] = None,
) -> Tuple[Optional[float], int]:
    total = 0.0
    missing = 0
    for b in buckets:
        if b.yes_mid is None:
            if missing_bucket_cents is None:
                return None, missing + 1
            total += float(missing_bucket_cents)
            missing += 1
        else:
            total += float(b.yes_mid)
    return total, missing


def bucket_sum_check(
    buckets: List[Bucket],
    missing_bucket_cents: Optional[float] = None,
) -> Tuple[Optional[float], int]:
    total, missing = _sum_mid(buckets, missing_bucket_cents=missing_bucket_cents)
    if total is None:
        return None, missing
    return total - 100.0, missing


def threshold_monotone_check(
    thresholds: List[Threshold],
    min_edge_cents: float,
) -> List[Tuple[int, int, float, float]]:
    items = [
        t
        for t in thresholds
        if t.strike_norm is not None and t.yes_mid is not None
    ]
    items.sort(key=lambda x: x.strike_norm or 0)
    out: List[Tuple[int, int, float, float]] = []
    for i in range(len(items) - 1):
        a = items[i]
        b = items[i + 1]
        if a.strike_norm is None or b.strike_norm is None:
            continue
        if a.yes_mid is None or b.yes_mid is None:
            continue
        if a.yes_mid + min_edge_cents < b.yes_mid:
            out.append((a.strike_norm, b.strike_norm, a.yes_mid, b.yes_mid))
    return out


def negative_implied_bucket_check(
    thresholds: List[Threshold],
    min_edge_cents: float,
) -> List[Tuple[int, int, float]]:
    items = [
        t
        for t in thresholds
        if t.strike_norm is not None and t.yes_mid is not None
    ]
    items.sort(key=lambda x: x.strike_norm or 0)
    out: List[Tuple[int, int, float]] = []
    for i in range(len(items) - 1):
        a = items[i]
        b = items[i + 1]
        if a.strike_norm is None or b.strike_norm is None:
            continue
        if a.yes_mid is None or b.yes_mid is None:
            continue
        implied = a.yes_mid - b.yes_mid
        if implied < -min_edge_cents:
            out.append((a.strike_norm, b.strike_norm, implied))
    return out


def _buckets_with_norm_low_at_or_above(
    buckets: List[Bucket],
    strike_norm: int,
) -> List[Bucket]:
    return [
        b
        for b in buckets
        if b.norm_low is not None and b.norm_low >= strike_norm
    ]


def _buckets_with_raw_low_at_or_above(
    buckets: List[Bucket],
    raw_low: float,
) -> List[Bucket]:
    return [
        b
        for b in buckets
        if b.raw_low is not None and b.raw_low >= raw_low
    ]


def _bucket_low_boundary(b: Bucket) -> Optional[int]:
    if b.norm_low is not None:
        return b.norm_low
    if b.raw_low is None:
        return None
    if abs(b.raw_low - round(b.raw_low)) < 1e-6:
        return int(round(b.raw_low))
    return None


def _bucket_high_boundary(b: Bucket) -> Optional[int]:
    if b.norm_high is not None:
        return b.norm_high
    if b.raw_high is None:
        return None
    frac = abs(b.raw_high - int(b.raw_high))
    if abs(frac - 0.99) < 1e-6:
        return int(b.raw_high) + 1
    if frac < 1e-6:
        return int(round(b.raw_high))
    return None


def _chain_complete_from_raw(
    buckets: List[Bucket],
    start_raw: float,
    *,
    eps: float = 0.02,
) -> bool:
    chain = [
        b
        for b in buckets
        if b.raw_low is not None and b.raw_high is not None and b.raw_low >= start_raw - eps
    ]
    if not chain:
        return False
    chain.sort(key=lambda b: b.raw_low or 0.0)
    first = chain[0]
    if first.raw_low is None or first.raw_low > start_raw + eps:
        return False
    for i in range(len(chain) - 1):
        cur = chain[i]
        nxt = chain[i + 1]
        if cur.raw_high is None:
            return True
        if nxt.raw_low is None:
            return False
        if nxt.raw_low - cur.raw_high > eps:
            return False
    last = chain[-1]
    return last.raw_high is None


def _bucket_chain_from_strike(
    buckets: List[Bucket],
    *,
    strike_raw: Optional[float],
    strike_norm: Optional[int],
    eps: float = 0.02,
) -> Tuple[List[Bucket], bool]:
    candidates: List[Bucket] = []
    if strike_raw is not None:
        for b in buckets:
            if b.raw_low is None:
                continue
            if b.raw_low + eps >= strike_raw:
                candidates.append(b)
    elif strike_norm is not None:
        for b in buckets:
            if b.norm_low is None:
                continue
            if b.norm_low >= strike_norm:
                candidates.append(b)
    if not candidates:
        return [], False
    candidates.sort(key=lambda b: b.raw_low if b.raw_low is not None else float(b.norm_low or 0))
    chain: List[Bucket] = []
    missing_hit = False
    for b in candidates:
        if b.yes_mid is None:
            missing_hit = True
            break
        chain.append(b)
    return chain, missing_hit


def compute_threshold_from_buckets(
    buckets: List[Bucket],
    thresholds: List[Threshold],
    require_exact_alignment: bool,
    min_edge_cents: float,
    missing_bucket_cents: Optional[float] = None,
    stop_at_stale: bool = False,
) -> Tuple[List[ThresholdImplied], List[TradeRecipe]]:
    implied_rows: List[ThresholdImplied] = []
    trades: List[TradeRecipe] = []

    for t in thresholds:
        notes: List[str] = []
        alignment = False
        partial = False
        implied = None
        lower_bound = None
        upper_bound = None

        if t.strike_norm is not None:
            for b in buckets:
                if b.norm_low == t.strike_norm:
                    alignment = True
                    break

        if alignment and t.strike_norm is not None:
            if stop_at_stale:
                basket, missing_hit = _bucket_chain_from_strike(
                    buckets, strike_raw=t.strike_raw, strike_norm=t.strike_norm
                )
                implied = sum(float(b.yes_mid) for b in basket) if basket else 0.0
                if missing_hit:
                    notes.append("stale_barrier")
                    partial = True
            else:
                basket = _buckets_with_norm_low_at_or_above(buckets, t.strike_norm)
                implied, missing = _sum_mid(basket, missing_bucket_cents=missing_bucket_cents)
                if missing > 0:
                    notes.append(f"missing_bucket_assumed={missing_bucket_cents}c")
                    partial = True
        else:
            partial = True
            if t.strike_raw is not None:
                straddle = None
                for b in buckets:
                    if b.raw_low is None or b.raw_high is None:
                        continue
                    if b.raw_low < t.strike_raw < b.raw_high:
                        straddle = b
                        break
                if straddle and straddle.raw_high is not None:
                    lower_buckets = _buckets_with_raw_low_at_or_above(
                        buckets, straddle.raw_high
                    )
                else:
                    lower_buckets = _buckets_with_raw_low_at_or_above(
                        buckets, t.strike_raw
                    )
                lower_bound, missing = _sum_mid(
                    lower_buckets, missing_bucket_cents=missing_bucket_cents
                )
                if missing > 0:
                    notes.append(f"missing_bucket_assumed={missing_bucket_cents}c")
                if straddle and straddle.yes_mid is not None and lower_bound is not None:
                    upper_bound = lower_bound + float(straddle.yes_mid)
                elif lower_bound is not None:
                    upper_bound = lower_bound
                if straddle is None:
                    notes.append("no_straddling_bucket")
            notes.append("partial_bucket_risk")

        actual = t.yes_mid
        diff = None
        if implied is not None and actual is not None:
            diff = implied - actual

        implied_rows.append(
            ThresholdImplied(
                ticker=t.ticker,
                strike_norm=t.strike_norm,
                strike_raw=t.strike_raw,
                implied_yes_geK=implied,
                actual_yes=actual,
                diff=diff,
                alignment=alignment,
                partial_bucket_risk=partial,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                notes=notes[:],
            )
        )

        if actual is None:
            continue

        if alignment and implied is not None:
            if diff is not None and diff >= min_edge_cents:
                if stop_at_stale:
                    basket, _ = _bucket_chain_from_strike(
                        buckets, strike_raw=t.strike_raw, strike_norm=t.strike_norm
                    )
                else:
                    basket = _buckets_with_norm_low_at_or_above(buckets, t.strike_norm)
                if not basket:
                    continue
                leg_specs = [
                    LegSpec(ticker=b.ticker, action="SELL", side="YES", role="bucket")
                    for b in basket
                ]
                leg_specs.append(
                    LegSpec(ticker=t.ticker, action="BUY", side="YES", role="threshold")
                )
                trades.append(
                    TradeRecipe(
                        trade_type="floor_from_buckets",
                        K=t.strike_norm,
                        delta=None,
                        direction="buy_threshold_sell_buckets",
                        legs=leg_specs,
                        implied=implied,
                        actual=actual,
                        diff=diff,
                        lower_bound=None,
                        upper_bound=None,
                        partial_bucket_risk=False,
                        alignment=True,
                        payout_per_contract=0,
                        notes=notes[:],
                    )
                )
            elif diff is not None and diff <= -min_edge_cents:
                if stop_at_stale:
                    basket, _ = _bucket_chain_from_strike(
                        buckets, strike_raw=t.strike_raw, strike_norm=t.strike_norm
                    )
                else:
                    basket = _buckets_with_norm_low_at_or_above(buckets, t.strike_norm)
                if not basket:
                    continue
                leg_specs = [
                    LegSpec(ticker=b.ticker, action="BUY", side="YES", role="bucket")
                    for b in basket
                ]
                leg_specs.append(
                    LegSpec(ticker=t.ticker, action="SELL", side="YES", role="threshold")
                )
                trades.append(
                    TradeRecipe(
                        trade_type="floor_from_buckets",
                        K=t.strike_norm,
                        delta=None,
                        direction="buy_buckets_sell_threshold",
                        legs=leg_specs,
                        implied=implied,
                        actual=actual,
                        diff=diff,
                        lower_bound=None,
                        upper_bound=None,
                        partial_bucket_risk=False,
                        alignment=True,
                        payout_per_contract=0,
                        notes=notes[:],
                    )
                )
        elif not alignment and not require_exact_alignment:
            if t.strike_raw is None:
                continue
            straddle = None
            for b in buckets:
                if b.raw_low is None or b.raw_high is None:
                    continue
                if b.raw_low < t.strike_raw < b.raw_high:
                    straddle = b
                    break
            if straddle and straddle.raw_high is not None:
                lower_buckets = _buckets_with_raw_low_at_or_above(buckets, straddle.raw_high)
            else:
                lower_buckets = _buckets_with_raw_low_at_or_above(buckets, t.strike_raw)
            lower_bound, missing = _sum_mid(
                lower_buckets, missing_bucket_cents=missing_bucket_cents
            )
            if missing > 0:
                notes.append(f"missing_bucket_assumed={missing_bucket_cents}c")
            upper_buckets = list(lower_buckets)
            if straddle is not None:
                upper_buckets.append(straddle)
            if lower_bound is not None and actual is not None:
                if upper_bound is None and straddle is not None and straddle.yes_mid is not None:
                    upper_bound = lower_bound + float(straddle.yes_mid)
                if upper_bound is not None and actual - upper_bound >= min_edge_cents:
                    leg_specs = [
                        LegSpec(ticker=b.ticker, action="BUY", side="YES", role="bucket")
                        for b in upper_buckets
                    ]
                    leg_specs.append(
                        LegSpec(ticker=t.ticker, action="SELL", side="YES", role="threshold")
                    )
                    trades.append(
                        TradeRecipe(
                            trade_type="floor_from_buckets",
                            K=t.strike_norm,
                            delta=None,
                            direction="buy_buckets_sell_threshold",
                            legs=leg_specs,
                            implied=None,
                            actual=actual,
                            diff=None,
                            lower_bound=lower_bound,
                            upper_bound=upper_bound,
                            partial_bucket_risk=True,
                            alignment=False,
                            payout_per_contract=0,
                            notes=notes[:] + ["bound=upper"],
                        )
                    )
                elif lower_bound - actual >= min_edge_cents:
                    leg_specs = [
                        LegSpec(ticker=b.ticker, action="SELL", side="YES", role="bucket")
                        for b in lower_buckets
                    ]
                    leg_specs.append(
                        LegSpec(ticker=t.ticker, action="BUY", side="YES", role="threshold")
                    )
                    trades.append(
                        TradeRecipe(
                            trade_type="floor_from_buckets",
                            K=t.strike_norm,
                            delta=None,
                            direction="buy_threshold_sell_buckets",
                            legs=leg_specs,
                            implied=None,
                            actual=actual,
                            diff=None,
                            lower_bound=lower_bound,
                            upper_bound=upper_bound,
                            partial_bucket_risk=True,
                            alignment=False,
                            payout_per_contract=0,
                            notes=notes[:] + ["bound=lower"],
                        )
                    )

    return implied_rows, trades


def compute_bucket_from_thresholds(
    buckets: List[Bucket],
    thresholds: List[Threshold],
    min_edge_cents: float,
    missing_bucket_cents: Optional[float] = None,
) -> Tuple[List[BucketImplied], List[TradeRecipe]]:
    implied_rows: List[BucketImplied] = []
    trades: List[TradeRecipe] = []
    th_map = {t.strike_norm: t for t in thresholds if t.strike_norm is not None}

    for b in buckets:
        if b.norm_low is None or b.norm_high is None:
            continue
        if b.norm_high <= b.norm_low:
            continue
        t_low = th_map.get(b.norm_low)
        t_high = th_map.get(b.norm_high)
        if not t_low or not t_high:
            continue
        if t_low.yes_mid is None or t_high.yes_mid is None:
            implied_rows.append(
                BucketImplied(
                    ticker=b.ticker,
                    low=b.norm_low,
                    high=b.norm_high,
                    implied_bucket=None,
                    actual_yes=b.yes_mid,
                    diff=None,
                    notes=["missing_threshold_mid"],
                )
            )
            continue
        implied = t_low.yes_mid - t_high.yes_mid
        actual = b.yes_mid
        notes: List[str] = []
        if actual is None and missing_bucket_cents is not None:
            actual = float(missing_bucket_cents)
            notes.append(f"missing_bucket_assumed={missing_bucket_cents}c")
        diff = None
        if actual is not None:
            diff = implied - actual

        implied_rows.append(
            BucketImplied(
                ticker=b.ticker,
                low=b.norm_low,
                high=b.norm_high,
                implied_bucket=implied,
                actual_yes=actual,
                diff=diff,
                notes=notes,
            )
        )

        if actual is None or diff is None:
            continue
        delta = b.norm_high - b.norm_low
        if diff >= min_edge_cents:
            legs = [
                LegSpec(ticker=b.ticker, action="BUY", side="YES", role="bucket"),
                LegSpec(ticker=t_low.ticker, action="SELL", side="YES", role="threshold_low"),
                LegSpec(ticker=t_high.ticker, action="BUY", side="YES", role="threshold_high"),
            ]
            trades.append(
                TradeRecipe(
                    trade_type="bucket_from_floors",
                    K=b.norm_low,
                    delta=delta,
                    direction="buy_bucket_sell_synth",
                    legs=legs,
                    implied=implied,
                    actual=actual,
                    diff=diff,
                    lower_bound=None,
                    upper_bound=None,
                    partial_bucket_risk=False,
                    alignment=True,
                    payout_per_contract=0,
                    notes=[],
                )
            )
        elif diff <= -min_edge_cents:
            legs = [
                LegSpec(ticker=b.ticker, action="SELL", side="YES", role="bucket"),
                LegSpec(ticker=t_low.ticker, action="BUY", side="YES", role="threshold_low"),
                LegSpec(ticker=t_high.ticker, action="SELL", side="YES", role="threshold_high"),
            ]
            trades.append(
                TradeRecipe(
                    trade_type="bucket_from_floors",
                    K=b.norm_low,
                    delta=delta,
                    direction="sell_bucket_buy_synth",
                    legs=legs,
                    implied=implied,
                    actual=actual,
                    diff=diff,
                    lower_bound=None,
                    upper_bound=None,
                    partial_bucket_risk=False,
                    alignment=True,
                    payout_per_contract=0,
                    notes=[],
                )
            )

    return implied_rows, trades


def compute_box_trades(
    buckets: List[Bucket],
    thresholds: List[Threshold],
) -> List[TradeRecipe]:
    bucket_by_low = {
        b.norm_low: b
        for b in buckets
        if b.norm_low is not None and b.yes_mid is not None
    }
    bucket_lows = sorted(bucket_by_low.keys())
    threshold_by_strike = {
        t.strike_norm: t
        for t in thresholds
        if t.strike_norm is not None and t.yes_mid is not None
    }
    strikes = sorted(threshold_by_strike.keys())
    trades: List[TradeRecipe] = []
    if not strikes or not bucket_lows:
        return trades

    for i in range(len(strikes)):
        low = strikes[i]
        t_low = threshold_by_strike.get(low)
        if t_low is None:
            continue
        for j in range(i + 1, len(strikes)):
            high = strikes[j]
            t_high = threshold_by_strike.get(high)
            if t_high is None:
                continue
            bucket_chain = [bucket_by_low[k] for k in bucket_lows if low <= k < high]
            if not bucket_chain:
                continue
            # Ensure there are no gaps in bucket chain.
            bucket_chain.sort(key=lambda b: b.norm_low or 0)
            first_low = _bucket_low_boundary(bucket_chain[0])
            if first_low is None or first_low != low:
                continue
            has_gap = False
            for idx in range(len(bucket_chain) - 1):
                cur = bucket_chain[idx]
                nxt = bucket_chain[idx + 1]
                cur_high = _bucket_high_boundary(cur)
                nxt_low = _bucket_low_boundary(nxt)
                if cur_high is None or nxt_low is None:
                    has_gap = True
                    break
                if nxt_low > cur_high:
                    has_gap = True
                    break
                if nxt_low < cur_high:
                    has_gap = True
                    break
            last_high = _bucket_high_boundary(bucket_chain[-1])
            if last_high is None or last_high != high:
                has_gap = True
            if has_gap:
                continue
            legs = [
                LegSpec(ticker=t_low.ticker, action="BUY", side="NO", role="threshold_low_no"),
                *[
                    LegSpec(ticker=b.ticker, action="BUY", side="YES", role="bucket")
                    for b in bucket_chain
                ],
                LegSpec(ticker=t_high.ticker, action="BUY", side="YES", role="threshold_high_yes"),
            ]
            trades.append(
                TradeRecipe(
                    trade_type="box_from_thresholds",
                    K=low,
                    delta=high - low,
                    direction="buy_box",
                    legs=legs,
                    implied=None,
                    actual=None,
                    diff=None,
                    lower_bound=None,
                    upper_bound=None,
                    partial_bucket_risk=False,
                    alignment=True,
                    payout_per_contract=100,
                    notes=[],
                )
            )
    return trades


def compute_constant_payout_trades(
    buckets: List[Bucket],
    thresholds: List[Threshold],
    *,
    min_payout_dollars: int = 2,
    max_payout_dollars: int = 5,
) -> List[TradeRecipe]:
    bucket_by_low = {
        b.norm_low: b
        for b in buckets
        if b.norm_low is not None and b.yes_mid is not None
    }
    bucket_lows = sorted(bucket_by_low.keys())
    threshold_by_strike = {
        t.strike_norm: t
        for t in thresholds
        if t.strike_norm is not None and t.yes_mid is not None
    }
    strikes = sorted(threshold_by_strike.keys())
    trades: List[TradeRecipe] = []
    if not strikes or not bucket_lows:
        return trades

    min_buckets = max(1, min_payout_dollars - 1)
    max_buckets = max(1, max_payout_dollars - 1)

    for i in range(len(strikes)):
        low = strikes[i]
        t_low = threshold_by_strike.get(low)
        if t_low is None:
            continue
        for j in range(i + 1, len(strikes)):
            high = strikes[j]
            t_high = threshold_by_strike.get(high)
            if t_high is None:
                continue
            bucket_chain = [bucket_by_low[k] for k in bucket_lows if low <= k < high]
            if not bucket_chain:
                continue
            if len(bucket_chain) < min_buckets or len(bucket_chain) > max_buckets:
                continue
            # Ensure there are no gaps in bucket chain.
            bucket_chain.sort(key=lambda b: b.norm_low or 0)
            first_low = _bucket_low_boundary(bucket_chain[0])
            if first_low is None or first_low != low:
                continue
            has_gap = False
            for idx in range(len(bucket_chain) - 1):
                cur = bucket_chain[idx]
                nxt = bucket_chain[idx + 1]
                cur_high = _bucket_high_boundary(cur)
                nxt_low = _bucket_low_boundary(nxt)
                if cur_high is None or nxt_low is None:
                    has_gap = True
                    break
                if nxt_low > cur_high:
                    has_gap = True
                    break
                if nxt_low < cur_high:
                    has_gap = True
                    break
            last_high = _bucket_high_boundary(bucket_chain[-1])
            if last_high is None or last_high != high:
                has_gap = True
            if has_gap:
                continue
            payout_dollars = len(bucket_chain) + 1
            payout_per_contract = payout_dollars * 100
            legs = [
                LegSpec(ticker=t_low.ticker, action="BUY", side="YES", role="threshold_low_yes"),
                *[
                    LegSpec(ticker=b.ticker, action="BUY", side="NO", role="bucket_no")
                    for b in bucket_chain
                ],
                LegSpec(ticker=t_high.ticker, action="BUY", side="NO", role="threshold_high_no"),
            ]
            trades.append(
                TradeRecipe(
                    trade_type="constant_payout",
                    K=low,
                    delta=high - low,
                    direction="buy_const_payout",
                    legs=legs,
                    implied=None,
                    actual=None,
                    diff=None,
                    lower_bound=None,
                    upper_bound=None,
                    partial_bucket_risk=False,
                    alignment=True,
                    payout_per_contract=payout_per_contract,
                    notes=[f"payout={payout_dollars}x"],
                )
            )
    return trades
