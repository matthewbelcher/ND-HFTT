#!/usr/bin/env python3
"""
GBM Market Making Simulator
============================
Combines the augmented GBM model (19 features: OBI + BTC momentum, AUC 0.705)
with a passive fill model — quotes only get filled when a real taker hits them.

Signal:
  P(up|10s) > threshold  → post YES bid 1 tick inside best bid
  P(up|10s) < 1-threshold → post YES ask 1 tick inside best ask

Fill model:
  Passive only. Fill when a negative orderbook delta arrives at our quote price
  and |delta_fp| <= 300 (cancel filter). Fill qty is proportional to our share
  of the level.

Fees: $0 maker (KXBTC15M has zero maker fees).

Usage:
    conda run -n AlgoTrade python analysis/market_maker_sim_gbm.py [csv ...] [opts]
    conda run -n AlgoTrade python analysis/market_maker_sim_gbm.py --threshold 0.58 --qty 10
    conda run -n AlgoTrade python analysis/market_maker_sim_gbm.py --thresholds 0.55,0.58,0.60 --qty 10
"""

import sys, json, argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kalshi_live.orderbook import KalshiOrderBook, compute_ob_from_book
from kalshi_live.features_live import LiveFeatureTracker, FEATS, kalshi_fee

CANCEL_FILTER_QTY = 300  # deltas larger than this are likely cancels, not trades


# ── Data parsing ──────────────────────────────────────────────────────────────

def parse_ts(s) -> pd.Timestamp:
    if s is None:
        return pd.NaT
    try:
        return pd.Timestamp(s, tz="UTC")
    except Exception:
        return pd.NaT


def parse_btc_csv(path: Path) -> List[Tuple[pd.Timestamp, float, str, float]]:
    """
    Parse BTC-TICKER.csv → list of (ts, price, side, size).
    side/size are '' / 0.0 for ticker-only updates (no trade).
    Returns sorted by timestamp.
    """
    ticks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            ch = msg.get("channel", "")
            ts_str = msg.get("timestamp", "")
            try:
                ts = pd.Timestamp(ts_str, tz="UTC")
            except Exception:
                continue

            for evt in msg.get("events", []):
                if ch == "ticker":
                    for t in evt.get("tickers", []):
                        px = t.get("price")
                        if px:
                            ticks.append((ts, float(px), "", 0.0))
                elif ch == "market_trades":
                    for t in evt.get("trades", []):
                        px = t.get("price")
                        sz = t.get("size", 0)
                        side = t.get("side", "")
                        t_ts_str = t.get("time", ts_str)
                        try:
                            t_ts = pd.Timestamp(t_ts_str, tz="UTC")
                        except Exception:
                            t_ts = ts
                        if px:
                            ticks.append((t_ts, float(px), side, float(sz)))
    ticks.sort(key=lambda x: x[0])
    return ticks


# ── Quote state ───────────────────────────────────────────────────────────────

@dataclass
class Quote:
    direction: int        # +1 = YES bid, -1 = YES ask
    price: float
    qty: int
    posted_at: pd.Timestamp
    level_qty_at_post: float


# ── Simulator ─────────────────────────────────────────────────────────────────

class GBMMarketMaker:
    def __init__(self, model, threshold: float = 0.58, qty: int = 10,
                 warmup_s: float = 35.0):
        self.model     = model
        self.threshold = threshold
        self.qty       = qty
        self.warmup_s  = warmup_s

        self.book         = KalshiOrderBook()
        self.feat_tracker = LiveFeatureTracker(sample_rate_hz=1.0)
        self.quote: Optional[Quote] = None
        self.session_start: Optional[pd.Timestamp] = None

        self._last_prob: Optional[float] = None

        # P&L / stats
        self.trades:    List[dict] = []
        self.net_pnl:   float = 0.0
        self.fees:      float = 0.0
        self.n_cancels: int   = 0
        self.n_no_signal: int = 0

    def reset(self):
        self.book.clear()
        self.feat_tracker = LiveFeatureTracker(sample_rate_hz=1.0)
        self.quote        = None
        self.session_start = None
        self._last_prob   = None
        self.trades       = []
        self.net_pnl      = 0.0
        self.fees         = 0.0
        self.n_cancels    = 0
        self.n_no_signal  = 0

    def on_btc_tick(self, ts: pd.Timestamp, price: float,
                    side: str = "", size: float = 0.0):
        wall = ts.timestamp()
        self.feat_tracker.add_btc_price(wall, price)
        if side in ("BUY", "SELL") and size > 0:
            self.feat_tracker.add_btc_trade(wall, side, size)

    def on_snapshot(self, msg: dict):
        self.book.apply_snapshot(msg)

    def on_delta(self, msg: dict, ts: pd.Timestamp):
        if self.session_start is None:
            self.session_start = ts

        # Check fill before updating book so we see pre-delta level qty
        yes_snap, no_snap = self.book.snapshot_tuple()
        delta_fp = float(msg.get("delta_fp", 0))
        price    = float(msg.get("price_dollars", 0))
        side     = msg.get("side", "")
        if delta_fp < 0 and self.quote is not None:
            self._check_fill(side, price, delta_fp, ts, yes_snap, no_snap)

        self.book.apply_delta(msg)

        yes_book, no_book = self.book.snapshot_tuple()
        if not yes_book or not no_book:
            return

        wall = ts.timestamp()
        # Always sample to build rolling history — even during warmup
        row = self.feat_tracker.maybe_sample(
            yes_book, no_book, now_mono=wall, now_wall=wall)

        if row is not None:
            vec = self.feat_tracker.feature_vector(row)
            if vec is not None:
                self._last_prob = float(self.model.predict_proba(vec)[0, 1])

        # Don't post quotes during warmup (need history for mom/obi_vel/toxicity)
        elapsed = (ts - self.session_start).total_seconds()
        if elapsed < self.warmup_s or self._last_prob is None:
            return

        ob  = compute_ob_from_book(yes_book, no_book)
        bid = ob["best_yes_bid"]
        ask = round(1.0 - ob["best_no_bid"], 2) if ob["best_no_bid"] > 0 else 1.0
        self._manage_quote(self._last_prob, bid, ask, ob, ts)

    def _manage_quote(self, prob: float, bid: float, ask: float,
                      ob: dict, ts: pd.Timestamp):
        if prob > self.threshold:
            direction   = +1
            quote_price = round(bid + 0.01, 2)
        elif prob < 1.0 - self.threshold:
            direction   = -1
            quote_price = round(ask - 0.01, 2)
        else:
            self.n_no_signal += 1
            if self.quote:
                self._cancel("no_signal")
            return

        if quote_price <= 0.01 or quote_price >= 0.99:
            if self.quote:
                self._cancel("price_out_of_range")
            return

        if self.quote is not None:
            mid = ob["mid_price"]
            if direction != self.quote.direction:
                self._cancel("direction_flip")
            elif direction == +1 and mid < self.quote.price - 0.005:
                self._cancel("mid_below_bid")
            elif direction == -1 and mid > self.quote.price + 0.005:
                self._cancel("mid_above_ask")
            elif abs(quote_price - self.quote.price) > 0.015:
                self._cancel("reprice")
            else:
                return  # keep existing quote

        # Post new quote
        level_qty = (ob.get("best_yes_bid_qty", 1.0) if direction == +1
                     else ob.get("best_no_bid_qty", 1.0))
        self.quote = Quote(
            direction=direction,
            price=quote_price,
            qty=self.qty,
            posted_at=ts,
            level_qty_at_post=max(1.0, level_qty),
        )

    def _check_fill(self, side: str, price: float, delta_fp: float,
                    ts: pd.Timestamp,
                    yes_snap: dict, no_snap: dict):
        q = self.quote
        abs_delta = abs(delta_fp)
        if abs_delta > CANCEL_FILTER_QTY:
            return  # likely a cancel, not a trade

        if q.direction == +1:
            if side != "yes" or abs(price - q.price) > 0.005:
                return
            level_qty = yes_snap.get(q.price, q.level_qty_at_post)
        else:
            no_price = round(1.0 - q.price, 2)
            if side != "no" or abs(price - no_price) > 0.005:
                return
            level_qty = no_snap.get(no_price, q.level_qty_at_post)

        # Proportional fill: our share of the level
        total     = max(level_qty, 1.0)
        fill_qty  = max(1, round((q.qty / (total + q.qty)) * abs_delta))
        fill_qty  = min(fill_qty, q.qty)
        if fill_qty <= 0:
            return

        # Re-compute mid from snapshot before applying delta
        ob   = compute_ob_from_book(yes_snap, no_snap)
        mid  = ob["mid_price"]
        pnl  = (mid - q.price) * fill_qty if q.direction == +1 else (q.price - mid) * fill_qty
        fee  = kalshi_fee(q.price, fill_qty, "maker")  # $0 on KXBTC15M
        net  = pnl - fee

        self.net_pnl += net
        self.fees    += fee
        self.trades.append({
            "ts":          str(ts),
            "direction":   q.direction,
            "price":       q.price,
            "fill_qty":    fill_qty,
            "mid_at_fill": mid,
            "pnl":         round(pnl, 4),
            "fee":         round(fee, 4),
            "net":         round(net, 4),
            "hold_s":      round((ts - q.posted_at).total_seconds(), 1),
            "prob":        round(self._last_prob, 4) if self._last_prob else None,
        })
        self.quote = None

    def _cancel(self, reason: str):
        self.n_cancels += 1
        self.quote = None

    def summary(self) -> dict:
        n    = len(self.trades)
        wins = sum(1 for t in self.trades if t["net"] > 0)
        return {
            "n_trades":   n,
            "win_rate":   round(wins / n, 3) if n else 0.0,
            "gross_pnl":  round(sum(t["pnl"] for t in self.trades), 4),
            "fees":       round(self.fees, 4),
            "net_pnl":    round(self.net_pnl, 4),
            "per_trade":  round(self.net_pnl / n, 4) if n else 0.0,
            "avg_hold_s": round(sum(t["hold_s"] for t in self.trades) / n, 1) if n else 0.0,
            "n_cancels":  self.n_cancels,
        }


# ── Replay ────────────────────────────────────────────────────────────────────

def replay_session(kalshi_path: Path, sim: GBMMarketMaker) -> Optional[dict]:
    btc_path = kalshi_path.parent / f"BTC-{kalshi_path.name}"
    btc_ticks = parse_btc_csv(btc_path) if btc_path.exists() else []
    btc_idx   = 0

    with open(kalshi_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            typ = msg.get("type")
            m   = msg.get("msg", {})

            if typ == "orderbook_snapshot":
                ts = parse_ts(m.get("ts"))
                while btc_idx < len(btc_ticks) and btc_ticks[btc_idx][0] <= ts:
                    sim.on_btc_tick(*btc_ticks[btc_idx])
                    btc_idx += 1
                sim.on_snapshot(m)

            elif typ == "orderbook_delta":
                ts = parse_ts(m.get("ts"))
                if pd.isna(ts):
                    continue
                while btc_idx < len(btc_ticks) and btc_ticks[btc_idx][0] <= ts:
                    sim.on_btc_tick(*btc_ticks[btc_idx])
                    btc_idx += 1
                sim.on_delta(m, ts)

    s = sim.summary()
    if s["n_trades"] > 0:
        return s
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*",
                        help="Kalshi session CSVs (default: /data/ with trade data)")
    parser.add_argument("--threshold", type=float, default=0.58,
                        help="Single GBM probability threshold (default: 0.58)")
    parser.add_argument("--thresholds", type=str, default="",
                        help="Comma-separated list of thresholds to sweep, e.g. 0.55,0.58,0.60")
    parser.add_argument("--qty", type=int, default=10,
                        help="Contracts per quote (default: 10)")
    parser.add_argument("--model", type=str, default="",
                        help="Path to .joblib model (default: output/ensemble_gbm_10s_btcmom.joblib)")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else ROOT / "output" / "ensemble_gbm_10s_btcmom.joblib"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run: conda run -n AlgoTrade python analysis/ensemble_with_btc_mom.py")
        sys.exit(1)

    artifact = joblib.load(model_path)
    model    = artifact["model"]
    feat_names = artifact.get("feature_names", FEATS)
    print(f"Loaded model: {model_path.name}")
    print(f"Features ({len(feat_names)}): {feat_names}")

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        data_dir = ROOT / "data"
        candidates = sorted(data_dir.glob("KXBTC15M-*.csv"), key=lambda p: p.name)
        paths = []
        for p in candidates:
            if p.name.startswith("BTC-"):
                continue
            try:
                with open(p) as f:
                    sample = "".join(next(f, "") for _ in range(50))
                if '"type":"trade"' in sample or '"type": "trade"' in sample:
                    paths.append(p)
            except Exception:
                pass
        if not paths:
            print("No sessions with trade data found in data/.")
            sys.exit(1)
        print(f"Found {len(paths)} sessions with trade data in data/\n")

    thresholds = [args.threshold]
    if args.thresholds:
        thresholds = [float(t) for t in args.thresholds.split(",")]

    for thr in thresholds:
        print(f"\n{'='*65}")
        print(f"  THRESHOLD = {thr:.2f}  |  qty = {args.qty}")
        print(f"{'='*65}")

        sim = GBMMarketMaker(model, threshold=thr, qty=args.qty)
        all_trades, total_net = [], 0.0
        n_sessions_with_trades = 0

        for path in paths:
            sim.reset()
            result = replay_session(path, sim)
            s = sim.summary()
            name = path.stem

            if s["n_trades"] == 0:
                print(f"  {name:<40}  no trades")
                continue

            n_sessions_with_trades += 1
            total_net += s["net_pnl"]
            all_trades.extend(sim.trades)

            wr  = f"{100*s['win_rate']:.0f}%"
            pnl = f"${s['net_pnl']:+.2f}"
            pt  = f"${s['per_trade']:+.4f}/trade"
            print(f"  {name:<40}  {s['n_trades']:3d} trades  WR={wr:>4}  net={pnl:>7}  {pt}")

        if not all_trades:
            print(f"\n  No trades across all sessions.")
            continue

        n_total = len(all_trades)
        wins    = sum(1 for t in all_trades if t["net"] > 0)
        gross   = sum(t["pnl"] for t in all_trades)
        fees    = sum(t["fee"] for t in all_trades)
        holds   = [t["hold_s"] for t in all_trades]

        print(f"\n  {'─'*60}")
        print(f"  AGGREGATE  {len(paths)} sessions  {n_sessions_with_trades} active")
        print(f"  Trades:        {n_total}")
        print(f"  Win rate:      {100*wins/n_total:.1f}%")
        print(f"  Gross P&L:    ${gross:+.2f}")
        print(f"  Fees:         ${fees:.2f}")
        print(f"  Net P&L:      ${total_net:+.2f}")
        print(f"  Per trade:    ${total_net/n_total:+.4f}")
        print(f"  Avg hold:      {sum(holds)/len(holds):.1f}s")
        print(f"  Median hold:   {sorted(holds)[len(holds)//2]:.1f}s")

        if len(thresholds) == 1:
            probs = [t["prob"] for t in all_trades if t["prob"] is not None]
            if probs:
                print(f"\n  Model confidence at fill:")
                print(f"    mean={sum(probs)/len(probs):.3f}  "
                      f"min={min(probs):.3f}  max={max(probs):.3f}")


if __name__ == "__main__":
    main()
