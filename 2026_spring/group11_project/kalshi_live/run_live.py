"""Async Kalshi order book + Coinbase trades + paper GBM simulator."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import websockets

from kalshi_live.coinbase_auth import build_ws_jwt
from kalshi_live.features_live import LiveFeatureTracker
from kalshi_live.kalshi_auth import load_private_key, ws_connect_headers
from kalshi_live.latency_util import (
    coinbase_reference_datetime,
    latency_ms_recv_minus_msg,
    parse_message_ts,
    summarize_samples,
    utc_now,
)
from kalshi_live.market_discovery import describe_discovery, pick_current_btc15m_ticker
from kalshi_live.model_runner import load_10s_model, predict_proba_up
from kalshi_live.orderbook import KalshiOrderBook
from kalshi_live.paper_simulator import PaperSimulator, format_status, save_session_results

KALSHI_WS = "wss://api.elections.kalshi.com/trade-api/ws/v2"
COINBASE_WS = "wss://advanced-trade-ws.coinbase.com"
# Clear viewport + cursor home (ANSI). Works in most modern terminals including Windows Terminal / WSL.
ANSI_CLEAR_SCREEN = "\033[2J\033[H"


def load_kalshi_key_id(project_root: Path) -> str:
    cred = project_root / "secrets" / "kalshi_credentials.json"
    if cred.exists():
        kid = json.loads(cred.read_text()).get("key_id", "").strip()
        if kid:
            return kid
    env = os.environ.get("KALSHI_ACCESS_KEY_ID", "").strip()
    if env:
        return env
    raise FileNotFoundError(
        'Set Kalshi key id: create secrets/kalshi_credentials.json with {"key_id": "..."} '
        "or export KALSHI_ACCESS_KEY_ID"
    )


def _parse_coinbase_trades(data: Dict[str, Any]) -> list[tuple[str, float, Optional[float]]]:
    """Return list of (side, size, price_or_none) from a parsed Coinbase Advanced Trade WS message."""
    out: list[tuple[str, float, Optional[float]]] = []
    if data.get("channel") != "market_trades":
        return out
    for ev in data.get("events") or []:
        for tr in ev.get("trades") or []:
            side = str(tr.get("side", "")).upper()
            try:
                size = float(tr.get("size", 0) or 0)
            except (TypeError, ValueError):
                continue
            px: Optional[float] = None
            raw_p = tr.get("price")
            if raw_p is not None:
                try:
                    px = float(raw_p)
                except (TypeError, ValueError):
                    px = None
            if side in ("BUY", "SELL") and size > 0:
                out.append((side, size, px))
    return out


async def kalshi_reader(
    market_ticker: str,
    pem_path: Path,
    key_id: str,
    book: KalshiOrderBook,
    lock: asyncio.Lock,
    health: Dict[str, Any],
) -> None:
    priv = load_private_key(pem_path)
    headers = ws_connect_headers(priv, key_id)
    sub = json.dumps(
        {
            "id": 1,
            "cmd": "subscribe",
            "params": {"channels": ["orderbook_delta"], "market_ticker": market_ticker},
        }
    )
    while True:
        try:
            async with websockets.connect(
                KALSHI_WS,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=60,
                close_timeout=10,
            ) as ws:
                await ws.send(sub)
                async for raw in ws:
                    health["kalshi_last"] = time.time()
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    typ = data.get("type")
                    msg = data.get("msg") or {}
                    if typ == "error":
                        health["kalshi_error"] = data
                        continue
                    async with lock:
                        if typ == "orderbook_snapshot":
                            book.apply_snapshot(msg)
                        elif typ == "orderbook_delta":
                            recv = utc_now()
                            t_msg = parse_message_ts(msg.get("ts"))
                            if t_msg is not None:
                                lat = latency_ms_recv_minus_msg(t_msg, recv)
                                health["kalshi_lat_last_ms"] = lat
                                health["kalshi_lat_samples"].append(lat)
                            book.apply_delta(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            health["kalshi_error"] = repr(e)
            await asyncio.sleep(2.0)


async def coinbase_reader(
    key_path: Path,
    product_id: str,
    tracker: LiveFeatureTracker,
    health: Dict[str, Any],
) -> None:
    sub_template = {
        "type": "subscribe",
        "product_ids": [product_id],
        "channel": "market_trades",
        "jwt": None,
    }
    while True:
        try:
            token = build_ws_jwt(key_path, ttl_seconds=120)
            sub_template["jwt"] = token
            payload = json.dumps(sub_template)
            async with websockets.connect(
                COINBASE_WS,
                ping_interval=20,
                ping_timeout=60,
                close_timeout=10,
            ) as ws:
                await ws.send(payload)
                t0 = time.time()
                while time.time() - t0 < 100:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    except asyncio.TimeoutError:
                        continue
                    health["coinbase_last"] = time.time()
                    now = time.time()
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    recv = utc_now()
                    msg_dt = coinbase_reference_datetime(data)
                    if msg_dt is not None:
                        lat = latency_ms_recv_minus_msg(msg_dt, recv)
                        health["coinbase_lat_last_ms"] = lat
                        health["coinbase_lat_samples"].append(lat)
                    for side, size, px in _parse_coinbase_trades(data):
                        tracker.add_btc_trade(now, side, size)
                        if px is not None:
                            health["coinbase_last_price"] = px
                            tracker.add_btc_price(now, px)  # feeds btc_mom features
        except asyncio.CancelledError:
            raise
        except Exception as e:
            health["coinbase_error"] = repr(e)
        await asyncio.sleep(1.0)


async def ui_loop(
    market_ticker: str,
    book: KalshiOrderBook,
    lock: asyncio.Lock,
    tracker: LiveFeatureTracker,
    model,
    sim: PaperSimulator,
    health: Dict[str, Any],
    sample_rate_hz: float = 20.0,
    ui_refresh_hz: float = 20.0,
) -> None:
    """Redraw the console status at ``ui_refresh_hz`` (defaults to ``sample_rate_hz``)."""
    print_interval = 1.0 / max(ui_refresh_hz, 1e-6)
    last_print = 0.0
    last_mid: float = 0.0
    last_prob: float = 0.5
    last_btc_mom: float = 0.0
    last_kalshi: Dict[str, Optional[float]] = {
        "yes_bid": None,
        "yes_ask": None,
        "mid": None,
    }
    # Wake at least ~2× per sample so we do not systematically undersample.
    tick = max(0.001, 0.5 / max(sample_rate_hz, 1e-6))
    while True:
        await asyncio.sleep(tick)
        now = time.time()
        async with lock:
            yes_d, no_d = book.snapshot_tuple()
        row = tracker.maybe_sample(yes_d, no_d)
        feat_ok = False
        prob = last_prob
        mid = last_mid
        if row is not None:
            mid = float(row["mid_price"])
            last_mid = mid
            last_kalshi["yes_bid"] = float(row.get("best_yes_bid") or 0.0)
            bn = float(row.get("best_no_bid") or 0.0)
            last_kalshi["yes_ask"] = (1.0 - bn) if bn > 0 else None
            last_kalshi["mid"] = mid
            X = tracker.feature_vector(row)
            if X is not None:
                prob = predict_proba_up(model, X)
                last_prob = prob
                feat_ok = True
                btc_mom = row.get("btc_mom_10s", 0.0) or 0.0
                last_btc_mom = btc_mom
                sim.on_tick(mid, prob, now, btc_mom=btc_mom)

        if now - last_print >= print_interval:
            last_print = now
            async with lock:
                yes_ui, no_ui = book.snapshot_tuple()
            kalshi_ok = (now - float(health.get("kalshi_last", 0))) < 15.0
            coinbase_ok = (now - float(health.get("coinbase_last", 0))) < 30.0
            k_lat = summarize_samples(
                health.get("kalshi_lat_samples") or deque(),
                health.get("kalshi_lat_last_ms"),
            )
            c_lat = summarize_samples(
                health.get("coinbase_lat_samples") or deque(),
                health.get("coinbase_lat_last_ms"),
            )
            coin_lat_disp = None if c_lat == "n/a" else c_lat
            sys.stdout.write(ANSI_CLEAR_SCREEN)
            sys.stdout.flush()
            print(
                format_status(
                    market=market_ticker,
                    mid=mid if mid else 0.0,
                    prob=prob,
                    sim=sim,
                    feat_ok=feat_ok,
                    kalshi_ok=kalshi_ok,
                    coinbase_ok=coinbase_ok,
                    kalshi_latency=k_lat,
                    coinbase_latency=coin_lat_disp,
                    kalshi_yes_bid=last_kalshi["yes_bid"],
                    kalshi_yes_ask=last_kalshi["yes_ask"],
                    coinbase_last_usd=health.get("coinbase_last_price"),
                    btc_mom_10s=last_btc_mom or None,
                    yes_book=yes_ui,
                    no_book=no_ui,
                )
            )
            err_k = health.get("kalshi_error")
            err_c = health.get("coinbase_error")
            if err_k:
                print(f"  [Kalshi error] {err_k}")
            if err_c:
                print(f"  [Coinbase error] {err_c}")


async def run_all(
    *,
    project_root: Path,
    market_ticker: str,
    product_id: str,
    threshold: float,
    qty: int,
    fee_type: str,
    btc_filter_thr: float,
    model_path: Path | None,
    sample_rate_hz: float,
    ui_refresh_hz: float,
) -> None:
    model = load_10s_model(project_root, model_path=model_path)
    print(
        "Loaded 10s GBM (augmented w/ BTC momentum). "
        "SIMULATION ONLY — no orders sent.\n"
    )

    pem = project_root / "secrets" / "TestExample1.pem"
    cdp = project_root / "secrets" / "cdp_api_key.json"
    if not pem.exists():
        raise FileNotFoundError(f"Missing {pem}")
    if not cdp.exists():
        raise FileNotFoundError(f"Missing {cdp}")

    key_id = load_kalshi_key_id(project_root)
    book = KalshiOrderBook()
    lock = asyncio.Lock()
    tracker = LiveFeatureTracker(
        sample_rate_hz=sample_rate_hz, btc_imbalance_window=900.0
    )
    print(
        f"Sampling: {sample_rate_hz:g} Hz  |  "
        f"BTC momentum filter: |btc_mom_10s| >= {btc_filter_thr:.4f}\n"
    )
    sim = PaperSimulator(
        threshold=threshold, qty=qty,
        fee_type=fee_type, btc_filter_thr=btc_filter_thr,
    )
    health: Dict[str, Any] = {
        "kalshi_last": 0.0,
        "coinbase_last": 0.0,
        "kalshi_error": None,
        "coinbase_error": None,
        "kalshi_lat_samples": deque(maxlen=200),
        "coinbase_lat_samples": deque(maxlen=200),
        "kalshi_lat_last_ms": None,
        "coinbase_lat_last_ms": None,
        "coinbase_last_price": None,
    }

    results_dir = project_root / "results"
    try:
        await asyncio.gather(
            kalshi_reader(market_ticker, pem, key_id, book, lock, health),
            coinbase_reader(cdp, product_id, tracker, health),
            ui_loop(
                market_ticker, book, lock, tracker, model, sim, health,
                sample_rate_hz=sample_rate_hz, ui_refresh_hz=ui_refresh_hz,
            ),
        )
    finally:
        # Save session results even if interrupted with Ctrl+C
        mid = sim.position.entry_mid if sim.position else 0.0
        out = save_session_results(
            sim, mid, market_ticker, results_dir,
            extra={"model": str(model_path or "augmented_btcmom"), "product": product_id},
        )
        print(f"\nSession results saved → {out}")


def main(argv: Optional[list] = None) -> None:
    import argparse

    argv = argv if argv is not None else __import__("sys").argv[1:]
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="Live paper simulator (Kalshi + Coinbase + GBM)."
    )
    p.add_argument(
        "--market",
        default=None,
        help="Kalshi market ticker. If omitted, the open KXBTC15M contract is chosen "
        "via GET /markets?series_ticker=…&status=open (soonest close_time).",
    )
    p.add_argument(
        "--series",
        default="KXBTC15M",
        help="Kalshi series ticker used only when --market is omitted (auto-discovery).",
    )
    p.add_argument("--product", default="BTC-USD", help="Coinbase product id")
    p.add_argument(
        "--thresh", type=float, default=0.58, help="Model probability threshold"
    )
    p.add_argument(
        "--qty", type=int, default=100, help="Paper position size (contracts)"
    )
    p.add_argument("--fee", choices=("maker", "taker"), default="maker")
    p.add_argument(
        "--btc-filter",
        type=float, default=0.0001, metavar="THR",
        help="Skip entry when |btc_mom_10s| >= THR and BTC opposes signal "
             "(0.0 = disabled). Default: 0.0001 (~39%% of signals kept, AUC 0.705).",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to ensemble_gbm_10s_live.joblib (default: output/ensemble_gbm_10s_live.joblib).",
    )
    p.add_argument(
        "--sample-hz",
        type=float,
        default=20.0,
        metavar="HZ",
        help="How many times per second to snapshot the book into features (default: 20). "
        "Momentum / MR / toxicity windows use wall-clock seconds, not row counts.",
    )
    p.add_argument(
        "--ui-hz",
        type=float,
        default=None,
        metavar="HZ",
        help="Console status refresh rate (default: same as --sample-hz). "
        "Lower (e.g. 2) for less terminal spam.",
    )
    args = p.parse_args(argv)
    if args.sample_hz <= 0:
        p.error("--sample-hz must be positive")
    ui_refresh_hz = args.ui_hz if args.ui_hz is not None else args.sample_hz
    if ui_refresh_hz <= 0:
        p.error("--ui-hz must be positive")

    if args.market:
        market_ticker = args.market
    else:
        try:
            market_ticker = pick_current_btc15m_ticker(series_ticker=args.series)
        except RuntimeError as e:
            p.error(str(e))
        print(describe_discovery(market_ticker, args.series))

    model_path = args.model_path
    if model_path is not None and not model_path.is_absolute():
        model_path = root / model_path

    try:
        asyncio.run(
            run_all(
                project_root=root,
                market_ticker=market_ticker,
                product_id=args.product,
                threshold=args.thresh,
                qty=args.qty,
                fee_type=args.fee,
                btc_filter_thr=args.btc_filter,
                model_path=model_path,
                sample_rate_hz=args.sample_hz,
                ui_refresh_hz=ui_refresh_hz,
            )
        )
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
