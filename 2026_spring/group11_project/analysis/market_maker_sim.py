#!/usr/bin/env python3
"""
Market Making Simulator
=======================
Replays historical Kalshi session CSVs event-by-event and simulates
a passive (maker-only) quoting strategy using OBI + BTC momentum.

Signal (combined):
  OBI level 1   — order book imbalance: best-bid level; IC=+0.20 at 5s
  btc_mom_10s   — BTC price % return over last 10s; IC=+0.14 at 5s
  depth_skew    — total YES depth vs NO depth across ALL price levels; IC=−0.12 at 5s
                  Negative IC: YES-heavy total book → price tends to FALL (supply overhang).
                  Filter: for YES bid, require total NO depth ≥ YES depth (depth_skew ≤ skew_thr).
                  For YES ask, require total YES depth ≥ NO depth (depth_skew ≥ −skew_thr).

BTC momentum plays TWO roles:
  1. Directional: BTC rising → lean YES bid; BTC falling → lean YES ask.
     Combined with OBI, this matches our research finding: when both agree,
     hit rate is 52.5% vs 41.8% when opposed (Δ = +10.7pp across 199 sessions).
  2. Adverse selection cancel: if |btc_mom_10s| > btc_cancel_thr AND BTC
     direction OPPOSES our resting quote → cancel immediately. BTC leads
     Kalshi by ~1-2s (lead-lag finding), so a BTC move against us predicts
     Kalshi is about to reprice against our quote.
     Applied on BOTH entry (pre-fill) and exit (post-fill) phases.

Quoting rules (entry):
  - Strong YES signal: OBI > +obi_thr AND btc_mom >= 0 AND depth_skew ≤ skew_thr
    → post YES bid 1 tick inside spread
  - Strong NO signal:  OBI < -obi_thr AND btc_mom <= 0 AND depth_skew ≥ −skew_thr
    → post YES ask 1 tick inside spread
  - Conflicting on any signal: do not post

Cancel / exit rules:
  - Entry quote: OBI flips, mid crosses, or BTC adverse → cancel
  - Exit (after fill): BTC moves strongly against position → taker exit immediately
  - Stop-loss: mid moves stop_ticks against entry → taker exit immediately
  - max_hold_s exceeded → forced taker exit

Fill model (validated by trade_vs_cancel.py on 3 sessions):
  - 95.9% of negative top-of-book deltas are actual trades
  - Fill triggers: negative delta at our price, |delta_fp| <= CANCEL_THRESHOLD
  - Partial fills proportional to our share of level quantity

Fee model:
  - Maker entry/exit: $0 (Kalshi charges no maker fees on KXBTC15M)
  - Forced taker exit: ceil(0.07 * qty * price * (1-price) * 100) / 100

Usage:
    conda run -n AlgoTrade python analysis/market_maker_sim.py [csv ...] [options]
    conda run -n AlgoTrade python analysis/market_maker_sim.py data/KXBTC15M-26APR200030-30.csv
    conda run -n AlgoTrade python analysis/market_maker_sim.py data/KXBTC15M-26APR*.csv --obi-thr 0.02 --qty 50
"""

import sys
import json
import math
import argparse
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Reuse the book implementation from merge_plot
sys.path.insert(0, str(Path(__file__).parent))
from merge_plot import OrderBook, parse_ts

ROOT = Path(__file__).resolve().parent.parent

TICK             = 0.01
CANCEL_THRESHOLD = 300.0   # delta_fp above this = probable cancel, not trade


# ── BTC momentum tracker ──────────────────────────────────────────────────────

class BtcMomTracker:
    """
    Maintains a rolling window of BTC (timestamp, price) pairs and computes
    btc_mom_10s = (price_now - price_10s_ago) / price_10s_ago.

    Fed from the paired BTC-TICKER.csv ticker channel messages.
    Returns None until at least 10s of history exists.
    """
    def __init__(self, window_s: float = 10.0):
        self.window_s = window_s
        self._prices: deque[tuple[pd.Timestamp, float]] = deque()

    def update(self, ts: pd.Timestamp, price: float):
        self._prices.append((ts, price))
        cutoff = ts - pd.Timedelta(seconds=self.window_s * 3)
        while self._prices and self._prices[0][0] < cutoff:
            self._prices.popleft()

    def momentum(self, now: pd.Timestamp) -> Optional[float]:
        if not self._prices:
            return None
        current = self._prices[-1][1]
        cutoff  = now - pd.Timedelta(seconds=self.window_s)
        past = None
        for ts, px in self._prices:
            if ts <= cutoff:
                past = px
        if past is None or past == 0:
            return None
        return (current - past) / past


def parse_btc_csv(path: Path) -> list[tuple[pd.Timestamp, float]]:
    """Extract (timestamp, btc_mid) pairs from a BTC ticker CSV."""
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
            if msg.get("channel") != "ticker":
                continue
            for ev in msg.get("events", []):
                if ev.get("type") == "snapshot":
                    continue
                for ticker in ev.get("tickers", []):
                    bid = ticker.get("best_bid")
                    ask = ticker.get("best_ask")
                    ts_str = msg.get("timestamp")
                    if bid and ask and ts_str:
                        try:
                            mid = (float(bid) + float(ask)) / 2
                            ts  = pd.Timestamp(ts_str[:26], tz="UTC")
                            ticks.append((ts, mid))
                        except (ValueError, TypeError):
                            continue
    return ticks


# ── Fee helpers ───────────────────────────────────────────────────────────────

def taker_fee(price: float, qty: int) -> float:
    return math.ceil(0.07 * qty * price * (1.0 - price) * 100) / 100

def maker_fee(price: float, qty: int) -> float:
    return 0.0  # no maker fee on KXBTC15M


# ── Position / Quote state ────────────────────────────────────────────────────

@dataclass
class Quote:
    """A resting limit order we've posted."""
    direction: int     # +1 = YES bid (want to buy YES), -1 = YES ask (want to sell YES)
    price: float
    qty: int
    posted_at: pd.Timestamp
    phase: str = "entry"  # "entry" or "exit"


@dataclass
class Fill:
    direction: int
    price: float
    qty: int
    ts: pd.Timestamp
    phase: str
    fee: float = 0.0


@dataclass
class Trade:
    entry: Fill
    exit: Optional[Fill] = None

    @property
    def gross_pnl(self) -> float:
        if self.exit is None:
            return 0.0
        return self.entry.direction * (self.exit.price - self.entry.price) * self.entry.qty

    @property
    def net_pnl(self) -> float:
        if self.exit is None:
            return 0.0
        return self.gross_pnl - self.entry.fee - self.exit.fee

    @property
    def hold_seconds(self) -> float:
        if self.exit is None:
            return 0.0
        return (self.exit.ts - self.entry.ts).total_seconds()


# ── Simulator ─────────────────────────────────────────────────────────────────

class MarketMakerSim:
    def __init__(
        self,
        qty: int = 10,
        obi_thr: float = 0.05,
        max_hold_s: float = 30.0,
        min_spread_ticks: int = 2,
        btc_cancel_thr: float = 0.0001,  # |btc_mom_10s| above this cancels opposing quote
        depth_skew_thr: float = 0.0,     # max depth_skew for YES bid; min -depth_skew for YES ask
                                         # 0.0 = require total book agrees with direction
                                         # 1.0 = disable depth_skew filter
        stop_ticks: int = 0,             # taker stop-loss ticks from entry (0 = disabled)
        verbose: bool = False,
    ):
        self.qty              = qty
        self.obi_thr          = obi_thr
        self.max_hold_s       = max_hold_s
        self.min_spread_ticks = min_spread_ticks
        self.btc_cancel_thr   = btc_cancel_thr
        self.depth_skew_thr   = depth_skew_thr
        self.stop_ticks       = stop_ticks
        self.verbose          = verbose

        self.book    = OrderBook()
        self.btc     = BtcMomTracker(window_s=10.0)
        self.quote:  Optional[Quote] = None
        self.open_trade: Optional[Trade] = None

        self.completed: list[Trade] = []
        self.n_cancelled      = 0
        self.n_btc_cancelled  = 0   # cancels triggered specifically by BTC momentum
        self.n_no_post_btc    = 0   # quotes not posted because BTC opposed OBI
        self.n_skew_filtered  = 0   # quotes not posted because depth_skew opposed direction
        self.n_taker_exits    = 0
        self.n_stopped_out    = 0   # stop-loss taker exits
        self.n_btc_exit       = 0   # BTC-adverse exit after fill
        self.total_fees       = 0.0

    # ── Main event loop ───────────────────────────────────────────────────────

    def on_btc_tick(self, ts: pd.Timestamp, price: float):
        self.btc.update(ts, price)

    def on_snapshot(self, msg: dict):
        self.book.apply_snapshot(msg)

    def on_delta(self, msg: dict, ts: pd.Timestamp):
        side      = msg["side"]
        price     = round(float(msg["price_dollars"]), 2)
        delta_fp  = float(msg["delta_fp"])

        # 1. Check if this delta fills our resting quote (BEFORE updating book)
        if self.quote is not None and delta_fp < 0:
            self._check_fill(side, price, delta_fp, ts)

        # 2. Update book state
        self.book.apply_delta(msg)

        if not self.book.ready:
            return

        bid = self.book.best_yes_bid()
        ask = self.book.best_yes_ask()
        if bid is None or ask is None:
            return

        spread_ticks = round((ask - bid) / TICK)

        # 3. Manage open position timeout
        if self.open_trade is not None and self.quote is not None:
            elapsed = (ts - self.open_trade.entry.ts).total_seconds()
            if elapsed > self.max_hold_s:
                self._force_taker_exit(bid, ask, ts)
                return

        # 4. If flat (no position, no quote): decide whether to post
        if self.open_trade is None and self.quote is None:
            self._maybe_quote(bid, ask, spread_ticks, ts)

        # 5. If holding a position (waiting for exit fill): manage exit quote
        elif self.open_trade is not None and self.quote is not None:
            self._manage_exit_quote(bid, ask, ts)

        # 6. If quote posted but not yet filled: check for cancel conditions
        elif self.open_trade is None and self.quote is not None:
            self._manage_entry_quote(bid, ask, ts)

    # ── Quoting logic ─────────────────────────────────────────────────────────

    def _depth_skew(self) -> float:
        """(yes_total - no_total) / (yes_total + no_total). Positive = YES-heavy."""
        yes_d, no_d = self.book.total_depth()
        total = yes_d + no_d
        return (yes_d - no_d) / total if total > 0 else 0.0

    def _maybe_quote(self, bid: float, ask: float,
                     spread_ticks: int, ts: pd.Timestamp):
        if spread_ticks < self.min_spread_ticks:
            return

        obi     = self.book.obi(depth=1, decay=0.0)
        btc_mom = self.btc.momentum(ts)   # None until 10s of BTC history
        skew    = self._depth_skew()

        if obi is None:
            return

        # BTC momentum directional role:
        #   btc_mom > 0 → BTC rising → confirms YES bid (OBI > 0 side)
        #   btc_mom < 0 → BTC falling → confirms YES ask (OBI < 0 side)
        #   btc_mom is None (warming up) → use OBI alone, no BTC filter
        def btc_agrees(direction: int) -> bool:
            if btc_mom is None:
                return True  # no data yet, don't block on warmup
            if direction == +1:
                return btc_mom >= 0
            else:
                return btc_mom <= 0

        # Depth skew directional role (IC = −0.12):
        #   YES-heavy total book (skew > 0) → supply overhang → price tends to FALL
        #   NO-heavy total book (skew < 0) → price tends to RISE
        #   For YES bid: require skew ≤ depth_skew_thr (book not YES-heavy)
        #   For YES ask: require skew ≥ −depth_skew_thr (book not NO-heavy)
        def skew_agrees(direction: int) -> bool:
            if self.depth_skew_thr >= 1.0:
                return True  # filter disabled
            if direction == +1:
                return skew <= self.depth_skew_thr
            else:
                return skew >= -self.depth_skew_thr

        if obi > self.obi_thr:
            if not btc_agrees(+1):
                self.n_no_post_btc += 1
                return  # OBI says YES but BTC falling — conflicting, skip
            if not skew_agrees(+1):
                self.n_skew_filtered += 1
                return  # OBI says YES but total book YES-heavy — supply overhang, skip
            quote_price = round(bid + TICK, 2)
            if quote_price >= ask:
                return
            self.quote = Quote(direction=+1, price=quote_price,
                               qty=self.qty, posted_at=ts, phase="entry")
            if self.verbose:
                print(f"  QUOTE BID {quote_price:.2f}  OBI={obi:+.3f}  "
                      f"btc_mom={btc_mom:+.5f}  skew={skew:+.3f}  spread={spread_ticks}t  {ts}")

        elif obi < -self.obi_thr:
            if not btc_agrees(-1):
                self.n_no_post_btc += 1
                return  # OBI says NO but BTC rising — conflicting, skip
            if not skew_agrees(-1):
                self.n_skew_filtered += 1
                return  # OBI says NO but total book NO-heavy — skip
            quote_price = round(ask - TICK, 2)
            if quote_price <= bid:
                return
            self.quote = Quote(direction=-1, price=quote_price,
                               qty=self.qty, posted_at=ts, phase="entry")
            if self.verbose:
                print(f"  QUOTE ASK {quote_price:.2f}  OBI={obi:+.3f}  "
                      f"btc_mom={btc_mom:+.5f}  skew={skew:+.3f}  spread={spread_ticks}t  {ts}")

    def _manage_entry_quote(self, bid: float, ask: float, ts: pd.Timestamp):
        """Cancel entry quote if signal flipped, mid crossed, or BTC opposes strongly."""
        obi     = self.book.obi(depth=1, decay=0.0)
        btc_mom = self.btc.momentum(ts)
        if obi is None:
            return
        q = self.quote

        signal_flipped = (q.direction == +1 and obi < -self.obi_thr) or \
                         (q.direction == -1 and obi >  self.obi_thr)

        mid = self.book.mid()
        mid_crossed = mid is not None and (
            (q.direction == +1 and mid < q.price) or
            (q.direction == -1 and mid > q.price)
        )

        # BTC adverse selection cancel:
        # BTC moving strongly against our quote direction = informed flow incoming.
        # Lead-lag: BTC leads Kalshi by ~1-2s, so cancel before Kalshi reprices.
        btc_adverse = (
            btc_mom is not None
            and abs(btc_mom) > self.btc_cancel_thr
            and (
                (q.direction == +1 and btc_mom < 0) or  # we're long YES, BTC falling
                (q.direction == -1 and btc_mom > 0)     # we're short YES, BTC rising
            )
        )

        if signal_flipped or mid_crossed or btc_adverse:
            self.n_cancelled += 1
            if btc_adverse:
                self.n_btc_cancelled += 1
            if self.verbose:
                if btc_adverse:
                    reason = f"BTC adverse (mom={btc_mom:+.5f})"
                elif signal_flipped:
                    reason = "signal flip"
                else:
                    reason = "mid crossed"
                print(f"  CANCEL {reason}  quote={q.price:.2f}  {ts}")
            self.quote = None

    def _manage_exit_quote(self, bid: float, ask: float, ts: pd.Timestamp):
        """Manage exit quote: BTC adverse cancel → taker exit; stop-loss; repost if moved."""
        if self.quote is None or self.open_trade is None:
            return
        q   = self.quote
        ent = self.open_trade.entry
        btc_mom = self.btc.momentum(ts)
        mid = self.book.mid()

        # Stop-loss: if mid has moved stop_ticks against our entry, exit via taker immediately
        if self.stop_ticks > 0 and mid is not None:
            loss_ticks = ent.direction * (ent.price - mid) / TICK
            if loss_ticks >= self.stop_ticks:
                if self.verbose:
                    print(f"  STOP-LOSS  entry={ent.price:.2f}  mid={mid:.4f}  "
                          f"loss={loss_ticks:.1f}t  {ts}")
                self.n_stopped_out += 1
                self._force_taker_exit(bid, ask, ts)
                return

        # BTC adverse exit: if BTC moves strongly against our open position, exit via taker
        # BTC leads Kalshi by ~1-2s, so a strong BTC spike against us = imminent adverse move
        if (btc_mom is not None
                and abs(btc_mom) > self.btc_cancel_thr
                and (
                    (ent.direction == +1 and btc_mom < 0) or  # long YES, BTC falling
                    (ent.direction == -1 and btc_mom > 0)     # short YES, BTC rising
                )):
            if self.verbose:
                print(f"  BTC EXIT  entry={ent.price:.2f}  btc_mom={btc_mom:+.5f}  {ts}")
            self.n_btc_exit += 1
            self._force_taker_exit(bid, ask, ts)
            return

        # Otherwise: adjust exit quote price to stay inside the current spread
        if q.direction == +1:  # we're short, exit = buy YES bid
            target = round(bid + TICK, 2)
            if target >= ask:
                target = bid
        else:  # we're long, exit = sell YES ask
            target = round(ask - TICK, 2)
            if target <= bid:
                target = ask

        if abs(target - q.price) >= TICK:
            self.quote = Quote(direction=q.direction, price=target,
                               qty=q.qty, posted_at=ts, phase="exit")

    # ── Fill detection ────────────────────────────────────────────────────────

    def _check_fill(self, side: str, price: float,
                    delta_fp: float, ts: pd.Timestamp):
        q = self.quote
        if q is None:
            return

        # Check if this delta is at our quoted price and the right side
        if not self._delta_hits_quote(side, price, q):
            return

        abs_delta = abs(delta_fp)

        # Probable cancel filter: large delta with no price movement = cancel
        if abs_delta > CANCEL_THRESHOLD:
            return

        # Level quantity before this delta
        idx   = round(price * 100) - 1
        level_qty = (self.book.yes_qty[idx] if side == "yes"
                     else self.book.no_qty[idx])

        # Proportional fill: our share of what was consumed
        if level_qty <= 0:
            return
        our_share = min(1.0, q.qty / level_qty)
        filled_qty = max(1, round(our_share * abs_delta))
        filled_qty = min(filled_qty, q.qty)

        fee = maker_fee(q.price, filled_qty)
        fill = Fill(direction=q.direction, price=q.price,
                    qty=filled_qty, ts=ts, phase=q.phase, fee=fee)
        self.total_fees += fee

        if self.verbose:
            print(f"  FILL {q.phase.upper()} dir={q.direction:+d}  "
                  f"price={q.price:.2f}  qty={filled_qty}  {ts}")

        if q.phase == "entry":
            self.open_trade = Trade(entry=fill)
            self.quote = None
            # Immediately post the exit quote
            self._post_exit_quote(ts)
        else:
            # Exit fill — trade complete
            self.open_trade.exit = fill
            self.completed.append(self.open_trade)
            self.open_trade = None
            self.quote = None

    def _delta_hits_quote(self, side: str, price: float, q: Quote) -> bool:
        """True if a negative delta at (side, price) would consume our quote."""
        if q.direction == +1:
            # We posted a YES bid → consumed when someone sells YES (yes side delta)
            return side == "yes" and abs(price - q.price) < 0.001
        else:
            # We posted a YES ask → equivalent to NO bid
            # Someone buying YES hits our YES ask → no side delta at (1 - q.price)
            no_equiv = round(1.0 - q.price, 2)
            return side == "no" and abs(price - no_equiv) < 0.001

    def _post_exit_quote(self, ts: pd.Timestamp):
        """After an entry fill, post the exit on the opposite side."""
        ent  = self.open_trade.entry
        bid  = self.book.best_yes_bid()
        ask  = self.book.best_yes_ask()
        if bid is None or ask is None:
            return

        if ent.direction == +1:
            # We bought YES → want to sell YES (post ask)
            exit_price = round(ask - TICK, 2)
            if exit_price <= bid:
                exit_price = ask
            self.quote = Quote(direction=-1, price=exit_price,
                               qty=ent.qty, posted_at=ts, phase="exit")
        else:
            # We sold YES → want to buy YES (post bid)
            exit_price = round(bid + TICK, 2)
            if exit_price >= ask:
                exit_price = bid
            self.quote = Quote(direction=+1, price=exit_price,
                               qty=ent.qty, posted_at=ts, phase="exit")

        if self.verbose:
            print(f"  POST EXIT {exit_price:.2f}  {ts}")

    def _force_taker_exit(self, bid: float, ask: float, ts: pd.Timestamp):
        """Exit position as taker (hit the book) when max_hold_s exceeded."""
        ent = self.open_trade.entry
        if ent.direction == +1:
            exit_price = bid   # sell YES at current bid
        else:
            exit_price = ask   # buy YES at current ask

        fee = taker_fee(exit_price, ent.qty)
        exit_fill = Fill(direction=-ent.direction, price=exit_price,
                         qty=ent.qty, ts=ts, phase="taker_exit", fee=fee)
        self.total_fees  += fee
        self.n_taker_exits += 1
        self.open_trade.exit = exit_fill
        self.completed.append(self.open_trade)
        self.open_trade = None
        self.quote = None

        if self.verbose:
            print(f"  TAKER EXIT {exit_price:.2f}  fee=${fee:.4f}  {ts}")

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        n = len(self.completed)
        if n == 0:
            return {"n_trades": 0}
        gross  = sum(t.gross_pnl for t in self.completed)
        net    = sum(t.net_pnl   for t in self.completed)
        wins   = sum(1 for t in self.completed if t.net_pnl > 0)
        hold_s = [t.hold_seconds for t in self.completed]
        return {
            "n_trades":         n,
            "n_cancelled":      self.n_cancelled,
            "n_btc_cancelled":  self.n_btc_cancelled,
            "n_no_post_btc":    self.n_no_post_btc,
            "n_skew_filtered":  self.n_skew_filtered,
            "n_taker_exits":    self.n_taker_exits,
            "n_stopped_out":    self.n_stopped_out,
            "n_btc_exit":       self.n_btc_exit,
            "win_rate":         round(wins / n, 3),
            "gross_pnl":        round(gross, 4),
            "total_fees":       round(self.total_fees, 4),
            "net_pnl":          round(net, 4),
            "avg_hold_s":       round(np.mean(hold_s), 1),
            "median_hold_s":    round(np.median(hold_s), 1),
            "pnl_per_trade":    round(net / n, 4),
        }


# ── Session replay ────────────────────────────────────────────────────────────

def replay_session(kalshi_path: Path, sim: MarketMakerSim) -> dict:
    # Load BTC ticks from paired file (BTC-TICKER.csv)
    btc_path = kalshi_path.parent / f"BTC-{kalshi_path.name}"
    btc_ticks = parse_btc_csv(btc_path) if btc_path.exists() else []
    btc_idx = 0  # pointer into sorted btc_ticks

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
            if typ == "orderbook_snapshot":
                sim.on_snapshot(msg.get("msg", {}))
            elif typ == "orderbook_delta":
                m  = msg.get("msg", {})
                ts = parse_ts(m["ts"])
                # Feed all BTC ticks up to this Kalshi timestamp
                while btc_idx < len(btc_ticks) and btc_ticks[btc_idx][0] <= ts:
                    sim.on_btc_tick(*btc_ticks[btc_idx])
                    btc_idx += 1
                sim.on_delta(m, ts)

    if not btc_ticks:
        print(f"  [warn] No BTC data found at {btc_path.name} — BTC momentum disabled")

    return sim.summary()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kalshi market making backtest")
    parser.add_argument("files", nargs="*", help="Session CSV files (default: newest in data/)")
    parser.add_argument("--qty",        type=int,   default=10,     help="Contracts per quote")
    parser.add_argument("--obi-thr",    type=float, default=0.05,   help="OBI threshold to post")
    parser.add_argument("--max-hold",   type=float, default=30.0,   help="Seconds before taker exit")
    parser.add_argument("--min-spread", type=int,   default=2,      help="Min spread ticks to post")
    parser.add_argument("--btc-cancel", type=float, default=0.0001,
                        help="|btc_mom_10s| threshold to cancel opposing quote (0=disable)")
    parser.add_argument("--depth-skew-thr", type=float, default=0.0,
                        help="Max depth_skew for YES bid (0=require book agrees, 1=disable)")
    parser.add_argument("--stop-ticks", type=int, default=0,
                        help="Stop-loss ticks from entry price (0=disable)")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        data_dir = ROOT / "data"
        paths = sorted(data_dir.glob("KXBTC15M-26APR*.csv"),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:5]
        if not paths:
            print("No KXBTC15M APR sessions found in data/. Pass file paths explicitly.")
            sys.exit(1)

    all_results = []
    for path in paths:
        sim = MarketMakerSim(
            qty=args.qty, obi_thr=args.obi_thr,
            max_hold_s=args.max_hold, min_spread_ticks=args.min_spread,
            btc_cancel_thr=args.btc_cancel,
            depth_skew_thr=args.depth_skew_thr,
            stop_ticks=args.stop_ticks,
            verbose=args.verbose,
        )
        res = replay_session(path, sim)
        res["session"] = path.stem
        all_results.append((path.stem, sim, res))
        print(f"\n{'='*60}")
        print(f"  {path.stem}")
        print(f"{'='*60}")
        if res.get("n_trades", 0) == 0:
            print("  No completed trades.")
            continue
        print(f"  Trades:           {res['n_trades']}")
        print(f"  Win rate:         {res['win_rate']:.1%}")
        print(f"  Gross P&L:       ${res['gross_pnl']:+.2f}")
        print(f"  Fees:            ${res['total_fees']:.2f}")
        print(f"  Net P&L:         ${res['net_pnl']:+.2f}")
        print(f"  Per trade:       ${res['pnl_per_trade']:+.4f}")
        print(f"  Avg hold:         {res['avg_hold_s']:.1f}s")
        print(f"  Cancels total:    {res['n_cancelled']}")
        print(f"    BTC-triggered:  {res['n_btc_cancelled']}")
        print(f"  Skipped (BTC vs OBI conflict): {res['n_no_post_btc']}")
        print(f"  Skipped (depth skew filter):   {res['n_skew_filtered']}")
        print(f"  Taker exits:      {res['n_taker_exits']}")
        print(f"    Stop-loss:      {res['n_stopped_out']}")
        print(f"    BTC adverse:    {res['n_btc_exit']}")

        # Trade-level breakdown (last 10)
        if sim.completed and args.verbose:
            print(f"\n  Last 10 trades:")
            for t in sim.completed[-10:]:
                tag = "T" if t.exit and t.exit.phase == "taker_exit" else "M"
                print(f"    [{tag}] dir={t.entry.direction:+d}  "
                      f"entry={t.entry.price:.2f}  "
                      f"exit={t.exit.price:.2f if t.exit else '?':>5}  "
                      f"net=${t.net_pnl:+.2f}  hold={t.hold_seconds:.0f}s")

    if len(all_results) > 1:
        total_net = sum(r["net_pnl"] for _, _, r in all_results if "net_pnl" in r)
        total_n   = sum(r.get("n_trades", 0) for _, _, r in all_results)
        print(f"\n{'='*60}")
        print(f"  AGGREGATE: {len(all_results)} sessions  "
              f"{total_n} trades  net=${total_net:+.2f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
