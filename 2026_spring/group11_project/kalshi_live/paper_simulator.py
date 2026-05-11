"""Paper P&L: GBM directional bets on YES mid with fee-aware accounting.

BTC momentum filter (from btc_cross_momentum.py findings):
  When OBI+model and BTC momentum agree on direction, hit rate is 52.5% vs
  41.8% when opposed (Δ = +10.7pp across 198 sessions, 154k 1s bars).
  The filter is applied at position open: if |btc_mom_10s| >= btc_filter_thr
  AND btc direction opposes the model signal, skip the trade.
  Set btc_filter_thr=0.0 to disable (trade on every signal).

Fee model (from Kalshi fee schedule):
  Taker: round_up(0.07  × C × P × (1-P))
  Maker: round_up(0.0175 × C × P × (1-P))
  At P=0.50 mid: taker=$0.0175/contract, maker=$0.004375/contract.
  Round-trip taker cost at mid = $0.035 — needs 4-tick move to break even.
  Round-trip maker cost at mid = $0.00875 — profitable on 1-tick move.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from kalshi_live.features_live import kalshi_fee
from kalshi_live.orderbook import format_order_book_display


@dataclass
class OpenPosition:
    direction: int        # +1 = long YES, -1 = short YES
    qty: int
    entry_mid: float
    entry_fee: float
    opened_at: float
    prob_at_entry: float
    btc_mom_at_entry: float = 0.0


@dataclass
class TradeEvent:
    kind: str             # OPEN or CLOSE
    time: float
    mid: float
    qty: int
    direction: int
    prob: float
    fee: float
    pnl_component: float  # net P&L for CLOSE events; 0 for OPEN
    reason: str
    note: str = ""
    btc_mom: float = 0.0
    btc_filtered: bool = False  # True if trade was skipped by BTC filter


class PaperSimulator:
    """
    Long YES when p(up) > thresh; short YES when p(up) < 1-thresh.
    Close when signal is flat or flips direction.
    BTC momentum filter: skip entry when |btc_mom_10s| >= btc_filter_thr
    and BTC direction opposes the model signal.
    """

    def __init__(
        self,
        threshold: float = 0.58,
        qty: int = 100,
        fee_type: str = "maker",
        btc_filter_thr: float = 0.0001,
    ) -> None:
        self.threshold = threshold
        self.qty = qty
        self.fee_type = fee_type
        self.btc_filter_thr = btc_filter_thr
        self.position: Optional[OpenPosition] = None
        self.realized_pnl: float = 0.0         # net of all closed trades (after fees)
        self.gross_realized_pnl: float = 0.0   # mid-to-mid P&L on closed trades (no fees)
        self.total_fees: float = 0.0
        self.events: List[TradeEvent] = []
        self.n_filtered: int = 0               # trades skipped by BTC filter

    def _signal(self, prob_up: float) -> int:
        if prob_up > self.threshold:
            return 1
        if prob_up < (1.0 - self.threshold):
            return -1
        return 0

    def _close(self, mid: float, ts: float, prob: float, reason: str,
               btc_mom: float = 0.0) -> None:
        if self.position is None:
            return
        p = self.position
        exit_fee = kalshi_fee(mid, self.qty, self.fee_type)
        gross = p.direction * (mid - p.entry_mid) * self.qty
        net   = gross - p.entry_fee - exit_fee
        self.realized_pnl      += net
        self.gross_realized_pnl += gross
        self.total_fees         += exit_fee
        self.events.append(TradeEvent(
            kind="CLOSE", time=ts, mid=mid, qty=self.qty,
            direction=p.direction, prob=prob,
            fee=p.entry_fee + exit_fee, pnl_component=net,
            reason=reason, note=f"held {ts - p.opened_at:.1f}s",
            btc_mom=btc_mom,
        ))
        self.position = None

    def _open(self, direction: int, mid: float, ts: float, prob: float,
              btc_mom: float = 0.0) -> None:
        entry_fee = kalshi_fee(mid, self.qty, self.fee_type)
        self.position = OpenPosition(
            direction=direction, qty=self.qty, entry_mid=mid,
            entry_fee=entry_fee, opened_at=ts, prob_at_entry=prob,
            btc_mom_at_entry=btc_mom,
        )
        self.total_fees += entry_fee
        side = "LONG_YES" if direction == 1 else "SHORT_YES"
        self.events.append(TradeEvent(
            kind="OPEN", time=ts, mid=mid, qty=self.qty,
            direction=direction, prob=prob, fee=entry_fee,
            pnl_component=0.0, reason=side, btc_mom=btc_mom,
        ))

    def on_tick(
        self,
        mid: float,
        prob_up: float,
        ts: float,
        btc_mom: float = 0.0,
    ) -> None:
        d = self._signal(prob_up)
        if self.position is not None:
            if d == 0 or d != self.position.direction:
                self._close(mid, ts, prob_up,
                            "flatten" if d == 0 else "flip", btc_mom)
        if self.position is None and d != 0:
            # BTC momentum filter: skip entry when BTC strongly opposes signal.
            # Only applies when we have a real BTC reading (btc_mom != 0).
            if self.btc_filter_thr > 0 and btc_mom != 0.0:
                btc_dir = 1 if btc_mom > 0 else -1
                if abs(btc_mom) >= self.btc_filter_thr and btc_dir != d:
                    self.n_filtered += 1
                    self.events.append(TradeEvent(
                        kind="OPEN", time=ts, mid=mid, qty=0,
                        direction=d, prob=prob_up, fee=0.0,
                        pnl_component=0.0, reason="FILTERED",
                        btc_mom=btc_mom, btc_filtered=True,
                    ))
                    return
            self._open(d, mid, ts, prob_up, btc_mom)

    def unrealized(self, mid: float) -> float:
        if self.position is None:
            return 0.0
        p = self.position
        return p.direction * (mid - p.entry_mid) * self.qty

    def gross_equity(self, mid: float) -> float:
        return self.gross_realized_pnl + self.unrealized(mid)

    def equity(self, mid: float) -> float:
        return self.realized_pnl + self.unrealized(mid)

    def recent_events(self, n: int = 30) -> List[TradeEvent]:
        return [e for e in self.events[-n:] if not e.btc_filtered]

    def session_summary(self, mid: float) -> Dict:
        """Return a serialisable dict for result logging."""
        closed = [e for e in self.events if e.kind == "CLOSE"]
        n_trades = len(closed)
        wins  = sum(1 for e in closed if e.pnl_component > 0)
        losses= sum(1 for e in closed if e.pnl_component < 0)
        fee_per_trade = self.total_fees / max(n_trades, 1)
        return {
            "n_trades":         n_trades,
            "n_filtered":       self.n_filtered,
            "win_rate":         round(wins / n_trades, 4) if n_trades else 0.0,
            "gross_pnl":        round(self.gross_realized_pnl, 4),
            "total_fees":       round(self.total_fees, 4),
            "net_pnl":          round(self.realized_pnl, 4),
            "unrealized":       round(self.unrealized(mid), 4),
            "net_mtm":          round(self.equity(mid), 4),
            "fee_per_trade":    round(fee_per_trade, 4),
            "fee_type":         self.fee_type,
            "threshold":        self.threshold,
            "btc_filter_thr":   self.btc_filter_thr,
            "qty":              self.qty,
        }


# ── Display helpers ────────────────────────────────────────────────────────────

def _fmt(v: Optional[float], d: int = 4) -> str:
    return "—" if v is None else f"{v:.{d}f}"


def format_status(
    *,
    market: str,
    mid: float,
    prob: float,
    sim: PaperSimulator,
    feat_ok: bool,
    kalshi_ok: bool,
    coinbase_ok: bool,
    kalshi_latency: str = "—",
    coinbase_latency: Optional[str] = None,
    kalshi_yes_bid: Optional[float] = None,
    kalshi_yes_ask: Optional[float] = None,
    coinbase_last_usd: Optional[float] = None,
    btc_mom_10s: Optional[float] = None,
    yes_book: Optional[Dict[float, float]] = None,
    no_book: Optional[Dict[float, float]] = None,
) -> str:
    unreal  = sim.unrealized(mid)
    net_eq  = sim.equity(mid)
    g_eq    = sim.gross_equity(mid)
    pos = "FLAT"
    if sim.position:
        pos = (f"{'LONG' if sim.position.direction == 1 else 'SHORT'} "
               f"{sim.qty} @ {sim.position.entry_mid:.4f}  "
               f"(btc_mom_entry={sim.position.btc_mom_at_entry:+.5f})")

    lat = (
        f"  LATENCY  Kalshi: {kalshi_latency}"
        + (f"  |  Coinbase: {coinbase_latency}" if coinbase_latency else "")
    )
    prices = (
        f"  PRICES   Kalshi YES  bid {_fmt(kalshi_yes_bid)}  "
        f"ask {_fmt(kalshi_yes_ask)}  mid {_fmt(mid)}  |  "
        f"Coinbase USD {_fmt(coinbase_last_usd, 2)}"
    )
    btc_mom_str = _fmt(btc_mom_10s, 5) if btc_mom_10s is not None else "warming"
    fee_note = f"maker=${kalshi_fee(mid,sim.qty,'maker'):.4f}/trade" if mid > 0 else ""

    lines = [
        "=" * 74,
        f"  MARKET   {market}",
        f"  MODEL    mid={mid:.4f}  P(up|10s)={prob:.3f}  "
        f"btc_mom_10s={btc_mom_str}  features={'OK' if feat_ok else 'warming'}",
        prices,
        f"  FEEDS    Kalshi={'up' if kalshi_ok else 'DOWN'}  "
        f"Coinbase={'up' if coinbase_ok else 'DOWN'}",
        lat,
        f"  POSITION {pos}",
        f"  P&L      gross=${g_eq:+.2f}  fees=${sim.total_fees:.2f}  "
        f"net=${net_eq:+.2f}  unrealized=${unreal:+.2f}  "
        f"({fee_note})  filtered={sim.n_filtered}",
    ]
    if yes_book is not None and no_book is not None:
        lines.append("")
        lines.extend(format_order_book_display(yes_book, no_book).splitlines())
    lines.append("-" * 74)
    for ev in sim.recent_events(12):
        lines.append(
            f"  {ev.kind:5s} t={ev.time:.1f} mid={ev.mid:.4f} "
            f"dir={ev.direction:+d} prob={ev.prob:.3f} "
            f"btc={ev.btc_mom:+.5f} fee=${ev.fee:.2f} "
            f"net=${ev.pnl_component:+.2f} [{ev.reason}] {ev.note}"
        )
    lines.append("=" * 74)
    return "\n".join(lines)


def save_session_results(
    sim: PaperSimulator,
    mid: float,
    market: str,
    results_dir: Path,
    extra: Optional[Dict] = None,
) -> Path:
    """Write session summary + full trade log to a timestamped JSON file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = results_dir / f"session_{market}_{ts}.json"

    summary = sim.session_summary(mid)
    if extra:
        summary.update(extra)

    trade_log = [
        {k: v for k, v in asdict(e).items()}
        for e in sim.events
        if not e.btc_filtered   # omit filtered-skips from the log
    ]

    out.write_text(json.dumps({
        "market":    market,
        "timestamp": ts,
        "summary":   summary,
        "trades":    trade_log,
    }, indent=2))
    return out
