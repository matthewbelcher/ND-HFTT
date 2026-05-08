import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class BboRow:
    ts_ms: int
    bid_px: int
    bid_qty: int
    ask_px: int
    ask_qty: int
    spread: int

    @property
    def mid(self) -> float:
        return (self.bid_px + self.ask_px) / 2.0

    @property
    def tob_qty(self) -> float:
        return (self.bid_qty + self.ask_qty) / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Kalshi BBO/trade logs")
    parser.add_argument("--bbo", required=True, help="Path to bbo.csv")
    parser.add_argument("--trades", help="Path to trades.csv (optional)")
    parser.add_argument("--burst-n", type=int, default=5, help="Trades per window for burst count")
    parser.add_argument("--burst-window", type=int, default=10, help="Burst window size in seconds")
    parser.add_argument("--vol-window", type=int, default=10, help="Volatility window in seconds")
    return parser.parse_args()


def weighted_median(values: List[float], weights: List[float]) -> Optional[float]:
    if not values:
        return None
    paired = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(weights)
    if total <= 0:
        return None
    acc = 0.0
    for v, w in paired:
        acc += w
        if acc >= total / 2:
            return v
    return paired[-1][0]


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def load_bbo(path: Path) -> Dict[str, List[BboRow]]:
    by_market: Dict[str, List[BboRow]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            market = row["market_ticker"]
            by_market[market].append(
                BboRow(
                    ts_ms=int(row["timestamp_ms"]),
                    bid_px=int(row["bid_px"]),
                    bid_qty=int(row["bid_qty"]),
                    ask_px=int(row["ask_px"]),
                    ask_qty=int(row["ask_qty"]),
                    spread=int(row["spread"]),
                )
            )
    for m in by_market:
        by_market[m].sort(key=lambda r: r.ts_ms)
    return by_market


def load_trades(path: Optional[Path]) -> Dict[str, List[int]]:
    if not path:
        return {}
    by_market: Dict[str, List[int]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            market = row.get("market_ticker")
            ts = row.get("timestamp_ms")
            if not market or not ts:
                continue
            by_market[market].append(int(ts))
    for m in by_market:
        by_market[m].sort()
    return by_market


def trade_bursts(trade_ts_ms: List[int], n: int, window_s: int) -> int:
    if not trade_ts_ms:
        return 0
    count = 0
    j = 0
    window_ms = window_s * 1000
    for i in range(len(trade_ts_ms)):
        while j < len(trade_ts_ms) and trade_ts_ms[j] - trade_ts_ms[i] <= window_ms:
            j += 1
        if j - i >= n:
            count += 1
    return count


def vol_proxy(rows: List[BboRow], window_s: int) -> Optional[float]:
    if len(rows) < 2:
        return None
    window_ms = window_s * 1000
    deltas: List[float] = []
    j = 0
    for i in range(len(rows)):
        if j < i:
            j = i
        while j < len(rows) and rows[j].ts_ms - rows[i].ts_ms < window_ms:
            j += 1
        if j < len(rows):
            deltas.append(abs(rows[j].mid - rows[i].mid))
    if not deltas:
        return None
    return statistics.median(deltas)


def analyze_market(
    rows: List[BboRow],
    trade_ts_ms: List[int],
    spread_thresholds: List[int],
    burst_n: int,
    burst_window: int,
    vol_window: int,
) -> Dict[str, Optional[float]]:
    if not rows:
        return {}

    spreads = []
    tob_qtys = []
    durations = []
    staleness = []

    total_duration_s = 0.0
    for i in range(len(rows) - 1):
        dt_ms = rows[i + 1].ts_ms - rows[i].ts_ms
        if dt_ms <= 0:
            continue
        dt_s = dt_ms / 1000.0
        total_duration_s += dt_s
        spreads.append(rows[i].spread)
        tob_qtys.append(rows[i].tob_qty)
        durations.append(dt_s)
        staleness.append(dt_s)

    median_spread = weighted_median(spreads, durations)
    median_tob_qty = weighted_median(tob_qtys, durations)
    median_stale = statistics.median(staleness) if staleness else None
    p95_stale = percentile(staleness, 0.95) if staleness else None

    spread_time = {t: 0.0 for t in spread_thresholds}
    if total_duration_s > 0:
        for s, dt in zip(spreads, durations):
            for t in spread_thresholds:
                if s >= t:
                    spread_time[t] += dt

    spread_pct = {
        t: (spread_time[t] / total_duration_s * 100.0) if total_duration_s > 0 else None
        for t in spread_thresholds
    }

    duration_hours = total_duration_s / 3600.0 if total_duration_s > 0 else None
    trade_count = len(trade_ts_ms)
    trades_per_hour = (
        trade_count / duration_hours if duration_hours and duration_hours > 0 else None
    )

    vol = vol_proxy(rows, vol_window)
    bursts = trade_bursts(trade_ts_ms, burst_n, burst_window) if trade_ts_ms else 0

    return {
        "duration_s": total_duration_s,
        "trades_per_hour": trades_per_hour,
        "median_spread": median_spread,
        "median_tob_qty": median_tob_qty,
        "median_stale_s": median_stale,
        "p95_stale_s": p95_stale,
        "vol_proxy_mid": vol,
        "burst_count": float(bursts),
        **{f"pct_spread_ge_{t}c": spread_pct[t] for t in spread_thresholds},
    }


def fmt(v: Optional[float], nd: int = 2) -> str:
    if v is None:
        return "n/a"
    if abs(v) >= 1000:
        return f"{v:,.{nd}f}"
    return f"{v:.{nd}f}"


def main() -> None:
    args = parse_args()
    bbo_path = Path(args.bbo)
    trades_path = Path(args.trades) if args.trades else None

    bbo = load_bbo(bbo_path)
    trades = load_trades(trades_path)

    thresholds = [2, 4, 6, 8, 10]

    for market in sorted(bbo.keys()):
        metrics = analyze_market(
            bbo[market],
            trades.get(market, []),
            thresholds,
            args.burst_n,
            args.burst_window,
            args.vol_window,
        )
        if not metrics:
            continue

        print(f"Market: {market}")
        print(f"Duration (s): {fmt(metrics['duration_s'])}")
        print(f"Trades/hour: {fmt(metrics['trades_per_hour'])}")
        print(f"Median spread (c): {fmt(metrics['median_spread'])}")
        print(f"Median top-of-book qty: {fmt(metrics['median_tob_qty'])}")
        print(f"Median staleness (s): {fmt(metrics['median_stale_s'])}")
        print(f"95p staleness (s): {fmt(metrics['p95_stale_s'])}")
        print(f"Vol proxy median |Δmid| per {args.vol_window}s: {fmt(metrics['vol_proxy_mid'])}")
        print(f"Trade bursts (>= {args.burst_n} in {args.burst_window}s): {fmt(metrics['burst_count'], 0)}")
        for t in thresholds:
            key = f"pct_spread_ge_{t}c"
            print(f"% time spread >= {t}c: {fmt(metrics[key])}")
        print("")


if __name__ == "__main__":
    main()
