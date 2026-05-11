#!/usr/bin/env python3
"""
Parameter sweep for the OBI + BTC momentum market making strategy.

Signals are pre-computed once per session (OBI, BTC mom, spread, depth skew);
each (session, combo) replay just iterates pre-computed arrays — ~8× faster
than re-parsing JSON for every combo.

Usage:
    conda run -n AlgoTrade python analysis/param_sweep.py            # full grid
    conda run -n AlgoTrade python analysis/param_sweep.py --quick    # coarser
    conda run -n AlgoTrade python analysis/param_sweep.py --top 15
"""

import sys, json, math, argparse, itertools
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

TICK             = 0.01
CANCEL_THRESHOLD = 300.0
BTC_WINDOW       = 10.0

# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

GRIDS = {
    "full": {
        "obi_thr":          [0.02, 0.05, 0.10, 0.15, 0.20],
        "btc_cancel_thr":   [0.0, 0.00005, 0.0001, 0.0003, 0.0005],
        "max_hold_s":       [15.0, 30.0, 60.0, 120.0],
        "depth_skew_thr":   [0.0, 0.2, 1.0],   # 1.0 = disabled
        "min_spread_ticks": [1, 2],
    },
    "quick": {
        "obi_thr":          [0.02, 0.05, 0.10, 0.20],
        "btc_cancel_thr":   [0.0, 0.0001, 0.0005],
        "max_hold_s":       [15.0, 30.0, 60.0],
        "depth_skew_thr":   [0.0, 1.0],
        "min_spread_ticks": [1, 2],
    },
}

# ---------------------------------------------------------------------------
# Pre-computed session: signals computed ONCE, reused across all param combos
# ---------------------------------------------------------------------------

@dataclass
class Session:
    stem: str
    # Per delta-event arrays (length = n_events)
    ts:         np.ndarray  # float64 unix seconds
    obi:        np.ndarray  # float32 OBI level-1 BEFORE this delta
    bid:        np.ndarray  # float32 best YES bid BEFORE delta
    ask:        np.ndarray  # float32 best YES ask BEFORE delta
    spread_t:   np.ndarray  # int8  (ask-bid)/TICK, capped 0-127
    depth_skew: np.ndarray  # float32 (yes_total - no_total) / total
    btc_mom:    np.ndarray  # float32 10s BTC momentum (nan = warming up)
    # Raw delta fields for fill detection
    side:       np.ndarray  # uint8  0=yes 1=no
    pidx:       np.ndarray  # uint8  0-based price index (1¢→0, 99¢→98)
    delta:      np.ndarray  # float32


def _parse_btc(path: Path):
    ts_list, px_list = [], []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: msg = json.loads(line)
                except: continue
                if msg.get("channel") != "ticker": continue
                ts_str = msg.get("timestamp")
                if not ts_str: continue
                for ev in msg.get("events", []):
                    if ev.get("type") == "snapshot": continue
                    for t in ev.get("tickers", []):
                        bid = t.get("best_bid"); ask = t.get("best_ask")
                        if bid and ask:
                            try:
                                ts_list.append(pd.Timestamp(ts_str[:26], tz="UTC").timestamp())
                                px_list.append((float(bid) + float(ask)) / 2)
                            except: pass
    except FileNotFoundError: pass
    return np.array(ts_list, dtype=np.float64), np.array(px_list, dtype=np.float64)


def _best_bid_arr(yes_qty):
    for i in range(98, -1, -1):
        if yes_qty[i] > 0: return (i + 1) * 0.01
    return 0.0

def _best_ask_arr(no_qty):
    for i in range(98, -1, -1):
        if no_qty[i] > 0: return round(1.0 - (i + 1) * 0.01, 2)
    return 1.0

def _obi_arr(yes_qty, no_qty):
    yq = nq = 0.0
    for i in range(98, -1, -1):
        if yq == 0 and yes_qty[i] > 0: yq = yes_qty[i]
        if nq == 0 and no_qty[i]  > 0: nq = no_qty[i]
        if yq and nq: break
    t = yq + nq
    return (yq - nq) / t if t > 0 else 0.0

def _skew_arr(yes_qty, no_qty):
    ys = yes_qty.sum(); ns = no_qty.sum(); t = ys + ns
    return (ys - ns) / t if t > 0 else 0.0


def parse_session(kalshi_path: Path) -> Session:
    btc_path = kalshi_path.parent / f"BTC-{kalshi_path.name}"
    btc_ts, btc_px = _parse_btc(btc_path)
    btc_buf: deque = deque()
    btc_idx = 0

    yes_qty = np.zeros(99, dtype=np.float64)
    no_qty  = np.zeros(99, dtype=np.float64)
    ready   = False

    ts_l, obi_l, bid_l, ask_l, st_l, sk_l, bm_l = [], [], [], [], [], [], []
    side_l, pidx_l, delta_l = [], [], []

    with open(kalshi_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: msg = json.loads(line)
            except: continue
            typ = msg.get("type")
            m   = msg.get("msg", {})

            if typ == "orderbook_snapshot":
                yes_qty[:] = 0; no_qty[:] = 0; ready = False
                for key, arr in [("yes_dollars_fp", yes_qty), ("no_dollars_fp", no_qty)]:
                    for pair in m.get(key, []):
                        try:
                            p = float(pair[0]); q = float(pair[1])
                            if 0 < p < 1 and q > 0:
                                arr[int(round(p * 100)) - 1] = q
                        except: pass
                ready = True
                continue

            if typ != "orderbook_delta": continue

            try:
                side_str = m.get("side", "yes")
                price    = float(m.get("price_dollars", 0))
                delt     = float(m.get("delta_fp", 0))
                ts_raw   = m.get("ts")
                if ts_raw is None: continue
                ts = pd.Timestamp(str(ts_raw)[:26], tz="UTC").timestamp()
                pc = int(round(price * 100))
                if pc < 1 or pc > 99: continue
            except: continue

            # Feed BTC ticks up to this ts
            while btc_idx < len(btc_ts) and btc_ts[btc_idx] <= ts:
                btc_buf.append((btc_ts[btc_idx], btc_px[btc_idx]))
                btc_idx += 1
            cutoff3 = ts - BTC_WINDOW * 3
            while btc_buf and btc_buf[0][0] < cutoff3:
                btc_buf.popleft()

            # BTC momentum
            btc_mom_val = float("nan")
            if btc_buf:
                cur = btc_buf[-1][1]; cut = ts - BTC_WINDOW; past = None
                for bts, bpx in btc_buf:
                    if bts <= cut: past = bpx
                if past and past > 0:
                    btc_mom_val = (cur - past) / past

            # Compute signals from CURRENT book state (before applying this delta)
            if ready:
                yb = _best_bid_arr(yes_qty); ya = _best_ask_arr(no_qty)
                obi_val = _obi_arr(yes_qty, no_qty)
                skew_val = _skew_arr(yes_qty, no_qty)
                spr = int(round((ya - yb) / TICK)) if yb > 0 and ya < 1.0 else 0
            else:
                yb = ya = obi_val = skew_val = 0.0; spr = 0

            ts_l.append(ts); obi_l.append(obi_val); bid_l.append(yb); ask_l.append(ya)
            st_l.append(spr); sk_l.append(skew_val); bm_l.append(btc_mom_val)
            side_l.append(0 if side_str == "yes" else 1)
            pidx_l.append(pc - 1)
            delta_l.append(delt)

            # Apply delta to book
            arr2 = yes_qty if side_str == "yes" else no_qty
            arr2[pc - 1] = max(0.0, arr2[pc - 1] + delt)

    return Session(
        stem=kalshi_path.stem,
        ts=np.array(ts_l,    dtype=np.float64),
        obi=np.array(obi_l,  dtype=np.float32),
        bid=np.array(bid_l,  dtype=np.float32),
        ask=np.array(ask_l,  dtype=np.float32),
        spread_t=np.array(st_l, dtype=np.int8),
        depth_skew=np.array(sk_l, dtype=np.float32),
        btc_mom=np.array(bm_l, dtype=np.float32),
        side=np.array(side_l, dtype=np.uint8),
        pidx=np.array(pidx_l, dtype=np.uint8),
        delta=np.array(delta_l, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Fast replay (signals pre-computed; just state machine per event)
# ---------------------------------------------------------------------------

def taker_fee(price: float, qty: int) -> float:
    return math.ceil(0.07 * qty * price * (1.0 - price) * 100) / 100


def replay(sess: Session, params: dict) -> float:
    """Returns net P&L for this session under these params."""
    obi_thr    = params["obi_thr"]
    btc_cancel = params["btc_cancel_thr"]
    max_hold_s = params["max_hold_s"]
    skew_thr   = params["depth_skew_thr"]
    min_spr    = params["min_spread_ticks"]
    qty        = params.get("qty", 10)

    ts_a = sess.ts; obi_a = sess.obi; bid_a = sess.bid; ask_a = sess.ask
    spr_a = sess.spread_t; sk_a = sess.depth_skew; bm_a = sess.btc_mom
    side_a = sess.side; pidx_a = sess.pidx; delta_a = sess.delta
    n = len(ts_a)

    q_dir = 0; q_price = 0.0; q_phase = 0   # 0=entry 1=exit
    pos_dir = 0; pos_price = 0.0; pos_ts = 0.0
    net = 0.0; total_fees = 0.0

    for i in range(n):
        ts    = ts_a[i]; obi = obi_a[i]; bid = bid_a[i]; ask = ask_a[i]
        spr   = spr_a[i]; skew = sk_a[i]; bm = bm_a[i]
        s_id  = side_a[i]; pidx = pidx_a[i]; delt = delta_a[i]
        bm_ok = not math.isnan(bm)

        # Fill check (before delta is applied)
        if q_dir != 0 and delt < 0:
            abs_d = abs(delt)
            if abs_d <= CANCEL_THRESHOLD:
                if q_dir == +1:
                    hits = (s_id == 0 and pidx == int(round(q_price * 100)) - 1)
                else:
                    no_pidx = int(round((1.0 - q_price) * 100)) - 1
                    hits = (s_id == 1 and pidx == no_pidx)
                if hits:
                    if q_phase == 0:   # entry fill
                        pos_dir = q_dir; pos_price = q_price; pos_ts = ts
                        # Post exit immediately
                        if bid > 0 and ask < 1.0:
                            if pos_dir == +1:
                                ep = round(ask - TICK, 2)
                                if ep <= bid: ep = ask
                                q_dir = -1; q_price = ep; q_phase = 1
                            else:
                                ep = round(bid + TICK, 2)
                                if ep >= ask: ep = bid
                                q_dir = +1; q_price = ep; q_phase = 1
                        else:
                            q_dir = 0
                    else:              # exit fill
                        pnl = pos_dir * (q_price - pos_price) * qty
                        net += pnl; pos_dir = 0; q_dir = 0; q_phase = 0

        if bid <= 0 or ask >= 1.0: continue

        # Max hold timeout
        if pos_dir != 0 and ts - pos_ts > max_hold_s:
            ep = bid if pos_dir == +1 else ask
            fee = taker_fee(ep, qty); total_fees += fee
            net += pos_dir * (ep - pos_price) * qty - fee
            pos_dir = 0; q_dir = 0; q_phase = 0
            continue

        if pos_dir == 0 and q_dir == 0:
            # Flat: maybe post
            if spr < min_spr: continue

            def btc_ok(d):
                return (not bm_ok) or (bm >= 0 if d == +1 else bm <= 0)
            def skew_ok(d):
                return skew_thr >= 1.0 or (skew <= skew_thr if d == +1 else skew >= -skew_thr)

            if obi > obi_thr:
                if not btc_ok(+1) or not skew_ok(+1): continue
                qp = round(bid + TICK, 2)
                if qp >= ask: qp = bid
                if bid <= 0.01: continue
                q_dir = +1; q_price = qp; q_phase = 0
            elif obi < -obi_thr:
                if not btc_ok(-1) or not skew_ok(-1): continue
                qp = round(ask - TICK, 2)
                if qp <= bid: qp = ask
                if ask >= 0.99: continue
                q_dir = -1; q_price = qp; q_phase = 0

        elif pos_dir == 0 and q_dir != 0 and q_phase == 0:
            # Entry quote: cancel conditions
            mid = (bid + ask) / 2
            sig_flip = (q_dir == +1 and obi < -obi_thr) or (q_dir == -1 and obi > obi_thr)
            mid_cross = (q_dir == +1 and mid < q_price) or (q_dir == -1 and mid > q_price)
            btc_adv = bm_ok and abs(bm) > btc_cancel and (
                (q_dir == +1 and bm < 0) or (q_dir == -1 and bm > 0))
            if sig_flip or mid_cross or btc_adv:
                q_dir = 0

        elif pos_dir != 0 and q_dir != 0 and q_phase == 1:
            # Exit quote: BTC adverse → taker; else reprice
            btc_adv = bm_ok and abs(bm) > btc_cancel and (
                (pos_dir == +1 and bm < 0) or (pos_dir == -1 and bm > 0))
            if btc_adv:
                ep = bid if pos_dir == +1 else ask
                fee = taker_fee(ep, qty); total_fees += fee
                net += pos_dir * (ep - pos_price) * qty - fee
                pos_dir = 0; q_dir = 0; q_phase = 0
                continue
            # Reprice exit quote
            if pos_dir == +1:
                target = round(ask - TICK, 2)
                if target <= bid: target = ask
            else:
                target = round(bid + TICK, 2)
                if target >= ask: target = bid
            if abs(target - q_price) >= TICK:
                q_price = target

    # Session end: close any residual position
    if pos_dir != 0:
        yb = bid_a[-1] if n > 0 else 0.0
        ya = ask_a[-1] if n > 0 else 1.0
        if yb > 0 and ya < 1.0:
            ep = yb if pos_dir == +1 else ya
            fee = taker_fee(ep, qty); total_fees += fee
            net += pos_dir * (ep - pos_price) * qty - fee

    return net - total_fees


# ---------------------------------------------------------------------------
# Multiprocessing: each worker holds all sessions (sent once via initializer)
# ---------------------------------------------------------------------------

_SESSIONS = None

def _init(sessions):
    global _SESSIONS
    _SESSIONS = sessions

def _work(params):
    pnls = [replay(s, params) for s in _SESSIONS]
    return pnls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def param_key(p):
    return (f"obi={p['obi_thr']:.3f} "
            f"btc_cancel={p['btc_cancel_thr']:.5f} "
            f"hold={p['max_hold_s']:.0f}s "
            f"skew={p['depth_skew_thr']:.1f} "
            f"min_spread={p['min_spread_ticks']}t")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",   action="store_true")
    parser.add_argument("--top",     type=int, default=15)
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = parser.parse_args()

    grid   = GRIDS["quick"] if args.quick else GRIDS["full"]
    keys   = list(grid.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]

    sessions = sorted(DATA_DIR.glob("KXBTC15M-*.csv"))
    sessions = [s for s in sessions if not s.name.startswith("BTC-")]
    if not sessions: print(f"No sessions in {DATA_DIR}"); sys.exit(1)

    print(f"Pre-parsing {len(sessions)} sessions  (workers={args.workers})...", flush=True)
    with Pool(args.workers) as pool:
        parsed = pool.map(parse_session, sessions)
    print(f"Done. {len(combos)} combos × {len(sessions)} sessions = "
          f"{len(combos)*len(sessions):,} replays", flush=True)

    rows = []
    chunk = max(1, len(combos) // 20)
    with Pool(args.workers, initializer=_init, initargs=(parsed,)) as pool:
        for ci, pnl_list in enumerate(pool.imap(_work, combos, chunksize=4)):
            pnls   = np.array(pnl_list)
            std    = pnls.std()
            sharpe = pnls.mean() / std if std > 1e-9 else 0.0
            rows.append({
                "params":           param_key(combos[ci]),
                "total_net":        round(float(pnls.sum()), 4),
                "mean_per_session": round(float(pnls.mean()), 4),
                "std_per_session":  round(float(std), 4),
                "sharpe":           round(sharpe, 3),
                "win_sessions":     round(float((pnls > 0).mean()), 3),
            })
            if (ci + 1) % chunk == 0 or ci == len(combos) - 1:
                print(f"  [{100*(ci+1)/len(combos):5.1f}%]  {ci+1}/{len(combos)}", flush=True)

    df = pd.DataFrame(rows)
    df_s = df.sort_values("sharpe",    ascending=False).reset_index(drop=True)
    df_p = df.sort_values("total_net", ascending=False).reset_index(drop=True)

    print(f"\n{'='*80}\nTOP {args.top} BY SHARPE (most stable)\n{'='*80}")
    for _, row in df_s.head(args.top).iterrows():
        print(f"  Sharpe={row['sharpe']:+.3f}  net=${row['total_net']:+.2f}  "
              f"win={row['win_sessions']:.0%}  | {row['params']}")

    print(f"\n{'='*80}\nTOP {args.top} BY TOTAL NET P&L\n{'='*80}")
    for _, row in df_p.head(args.top).iterrows():
        print(f"  net=${row['total_net']:+.2f}  Sharpe={row['sharpe']:+.3f}  "
              f"win={row['win_sessions']:.0%}  | {row['params']}")

    cur_key = param_key({"obi_thr": 0.05, "btc_cancel_thr": 0.0001,
                         "max_hold_s": 30.0, "depth_skew_thr": 1.0, "min_spread_ticks": 1})
    match = df_s[df_s["params"] == cur_key]
    if not match.empty:
        row = match.iloc[0]
        rs = df_s[df_s["params"] == cur_key].index[0] + 1
        rp = df_p[df_p["params"] == cur_key].index[0] + 1
        print(f"\n{'='*80}\nCURRENT LIVE DEFAULTS\n{'='*80}")
        print(f"  net=${row['total_net']:+.2f}  Sharpe={row['sharpe']:+.3f}  "
              f"win={row['win_sessions']:.0%}  "
              f"rank_sharpe={rs}/{len(combos)}  rank_pnl={rp}/{len(combos)}")
        print(f"  {cur_key}")

    out = ROOT / "output" / "param_sweep_results.csv"
    out.parent.mkdir(exist_ok=True)
    df_s.to_csv(out, index=False)
    print(f"\nResults → {out}")


if __name__ == "__main__":
    main()
