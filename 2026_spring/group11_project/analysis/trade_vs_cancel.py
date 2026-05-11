#!/usr/bin/env python3
"""
Trade vs Cancel Analysis
========================
Once the collector records the trade channel alongside orderbook_delta,
this script labels every negative book delta at the best bid/ask as either:
  - TRADE  : a trade message arrived at the same price within 50ms
  - CANCEL : no matching trade message

Then it measures what fraction of negative deltas are trades vs cancels,
broken down by features (depth level, BTC momentum, time-in-session, etc.)
to build intuition for how to handle *old* data that has no trade channel.

Usage (run from project root):
    conda run -n AlgoTrade python analysis/trade_vs_cancel.py [csv_file ...]

If no files given, scans bigdata/ for the newest sessions that contain
trade messages.
"""

import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent

# ── Parse a single session CSV into two DataFrames ───────────────────────────

def parse_session(path: Path):
    """Return (deltas_df, trades_df) from a session CSV that has both channels."""
    deltas, trades = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            typ = msg.get("type")
            if typ == "orderbook_delta":
                m = msg.get("msg", {})
                deltas.append({
                    "ts":       m.get("ts"),
                    "price":    float(m.get("price_dollars", 0)),
                    "delta_fp": float(m.get("delta_fp", 0)),
                    "side":     m.get("side"),
                })
            elif typ == "trade":
                m = msg.get("msg", {})
                trades.append({
                    "trade_id":   m.get("trade_id"),
                    "ts_s":       m.get("ts"),       # Unix seconds (ts_ms not sent by API)
                    "yes_price":  float(m.get("yes_price_dollars", 0)),
                    "no_price":   float(m.get("no_price_dollars", 0)),
                    "count_fp":   float(m.get("count_fp", 0)),
                    "taker_side": m.get("taker_side"),
                })

    if not deltas or not trades:
        return None, None

    df_d = pd.DataFrame(deltas)
    df_d["ts"] = pd.to_datetime(df_d["ts"], utc=True)

    df_t = pd.DataFrame(trades)
    # API sends ts in seconds (ts_ms is documented but not currently returned)
    df_t["ts"] = pd.to_datetime(df_t["ts_s"], unit="s", utc=True)

    return df_d, df_t


def build_book_state(deltas_df: pd.DataFrame):
    """Rebuild best bid/ask at each delta event from running book state."""
    book_yes = defaultdict(float)  # price -> qty
    book_no  = defaultdict(float)

    best_bids, best_asks = [], []

    for _, row in deltas_df.iterrows():
        price = row["price"]
        side  = row["side"]
        if side == "yes":
            book_yes[price] = max(0.0, book_yes[price] + row["delta_fp"])
        else:
            book_no[price] = max(0.0, book_no[price] + row["delta_fp"])

        yes_bids = [p for p, q in book_yes.items() if q > 0]
        no_bids  = [p for p, q in book_no.items()  if q > 0]

        best_yes_bid = max(yes_bids) if yes_bids else np.nan
        # YES ask = 1 - best NO bid (complementary sides)
        best_no_bid  = max(no_bids)  if no_bids  else np.nan
        best_yes_ask = round(1.0 - best_no_bid, 2) if not np.isnan(best_no_bid) else np.nan

        best_bids.append(best_yes_bid)
        best_asks.append(best_yes_ask)

    deltas_df = deltas_df.copy()
    deltas_df["best_yes_bid"] = best_bids
    deltas_df["best_yes_ask"] = best_asks
    deltas_df["mid"] = (deltas_df["best_yes_bid"] + deltas_df["best_yes_ask"]) / 2
    return deltas_df


def label_deltas(deltas_df: pd.DataFrame, trades_df: pd.DataFrame,
                 window_ms: float = 50.0):
    """
    For each negative delta at the best bid or ask, label it TRADE or CANCEL.

    Rule: if a trade message exists within ±window_ms ms at the same price,
    it's a TRADE. Otherwise it's a CANCEL.

    Returns a DataFrame of labeled negative-at-top events.
    """
    neg = deltas_df[deltas_df["delta_fp"] < 0].copy()

    # Only events at the current best bid (yes side) or best ask (no side)
    at_top = neg[
        ((neg["side"] == "yes") & (neg["price"] == neg["best_yes_bid"])) |
        ((neg["side"] == "no")  & (neg["price"].round(2) == (1.0 - neg["best_yes_ask"]).round(2)))
    ].copy()

    if at_top.empty:
        return at_top

    labels = []
    t_arr   = trades_df["ts"].values.astype("int64")  # ns
    tp_yes  = trades_df["yes_price"].values
    tp_no   = trades_df["no_price"].values
    ts_side = trades_df["taker_side"].values

    window_ns = int(window_ms * 1e6)

    for _, row in at_top.iterrows():
        t0 = row["ts"].value
        mask = (np.abs(t_arr - t0) <= window_ns)
        nearby = trades_df[mask]

        matched = False
        if not nearby.empty:
            if row["side"] == "yes":
                # Trade hits YES bid when taker_side == "no" at same price
                matched = any(
                    (abs(r["yes_price"] - row["price"]) < 0.005) and r["taker_side"] == "no"
                    for _, r in nearby.iterrows()
                )
            else:
                # Trade hits NO bid (=YES ask) when taker_side == "yes"
                matched = any(
                    (abs(r["no_price"] - row["price"]) < 0.005) and r["taker_side"] == "yes"
                    for _, r in nearby.iterrows()
                )

        labels.append("TRADE" if matched else "CANCEL")

    at_top["label"] = labels
    at_top["abs_delta"] = at_top["delta_fp"].abs()
    return at_top


def analyze(labeled: pd.DataFrame, session_name: str):
    n_total  = len(labeled)
    n_trade  = (labeled["label"] == "TRADE").sum()
    n_cancel = (labeled["label"] == "CANCEL").sum()

    print(f"\n{'='*60}")
    print(f"Session: {session_name}")
    print(f"{'='*60}")
    print(f"  Negative top-of-book deltas: {n_total}")
    print(f"  Labeled TRADE:  {n_trade}  ({100*n_trade/max(n_total,1):.1f}%)")
    print(f"  Labeled CANCEL: {n_cancel} ({100*n_cancel/max(n_total,1):.1f}%)")

    if n_total == 0:
        return

    # By YES side vs NO side
    print("\n  By side:")
    for side in ["yes", "no"]:
        sub = labeled[labeled["side"] == side]
        if sub.empty:
            continue
        t = (sub["label"] == "TRADE").sum()
        print(f"    {side}: {len(sub)} events  {100*t/len(sub):.1f}% trades")

    # By price distance from mid at time of event
    labeled = labeled.copy()
    labeled["dist_from_mid"] = (labeled["price"] - labeled["mid"]).abs().round(2)
    print("\n  Trade rate by distance from mid:")
    for dist in sorted(labeled["dist_from_mid"].unique()):
        sub = labeled[labeled["dist_from_mid"] == dist]
        t = (sub["label"] == "TRADE").sum()
        print(f"    dist={dist:.2f}  n={len(sub):4d}  {100*t/len(sub):.1f}% trades")

    # Size distribution: trades vs cancels
    print("\n  Median abs delta size:")
    for lbl in ["TRADE", "CANCEL"]:
        sub = labeled[labeled["label"] == lbl]
        if not sub.empty:
            print(f"    {lbl}: median={sub['abs_delta'].median():.1f}  "
                  f"mean={sub['abs_delta'].mean():.1f}  "
                  f"p95={sub['abs_delta'].quantile(0.95):.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="Session CSV files to analyze")
    parser.add_argument("--window-ms", type=float, default=1500.0,
                        help="Match window in ms for trade labeling (default: 1500 — "
                             "trade ts has second precision so needs >=500ms + latency)")
    args = parser.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        # Auto-find sessions in bigdata/ that have trade messages
        bigdata = ROOT.parent / "bigdata"
        candidates = sorted(bigdata.glob("KXBTC15M-*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        paths = []
        print(f"Scanning {len(candidates)} sessions for trade messages...")
        for p in candidates[:20]:  # check newest 20
            with open(p) as f:
                sample = "".join(next(f, "") for _ in range(200))
            if '"type":"trade"' in sample or '"type": "trade"' in sample:
                paths.append(p)
                if len(paths) >= 5:
                    break
        if not paths:
            print("No sessions with trade data found. Run the updated collector first.")
            print("Rebuild: cd collector && make && ./build/collector ...")
            sys.exit(0)
        print(f"Found {len(paths)} sessions with trade data.\n")

    all_labeled = []
    for path in paths:
        deltas_df, trades_df = parse_session(path)
        if deltas_df is None:
            print(f"  [skip] {path.name} — no trade messages")
            continue
        deltas_df = build_book_state(deltas_df)
        labeled = label_deltas(deltas_df, trades_df, window_ms=args.window_ms)
        if labeled.empty:
            print(f"  [skip] {path.name} — no negative top-of-book events")
            continue
        analyze(labeled, path.name)
        labeled["session"] = path.stem
        all_labeled.append(labeled)

    if not all_labeled:
        return

    combined = pd.concat(all_labeled, ignore_index=True)
    n = len(combined)
    n_trade = (combined["label"] == "TRADE").sum()
    print(f"\n{'='*60}")
    print(f"AGGREGATE ACROSS {len(all_labeled)} SESSIONS")
    print(f"{'='*60}")
    print(f"  Total top-of-book negative deltas: {n}")
    print(f"  Overall trade rate: {100*n_trade/n:.1f}%")
    print(f"  Overall cancel rate: {100*(n-n_trade)/n:.1f}%")
    print()
    print("  Implication for historical backtest fill assumption:")
    print(f"  If you treat ALL negative top-of-book deltas as fills,")
    print(f"  you overstate fill rate by ~{100*(n-n_trade)/n:.0f}% (the cancel fraction).")


if __name__ == "__main__":
    main()
