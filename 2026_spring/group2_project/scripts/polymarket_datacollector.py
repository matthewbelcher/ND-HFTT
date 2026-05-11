#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import requests

try:
    import socks
except ImportError:
    socks = None

try:
    from aiohttp_socks import ProxyConnector
except ImportError:
    ProxyConnector = None


GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GEOBLOCK_URL = "https://polymarket.com/api/geoblock"

REST_DELAY = 0.3

INTERVAL_SLUG_PREFIX = {
    5: "btc-updown-5m",
    15: "btc-updown-15m",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("polymarket_collector")


def resolve_proxy(cli_proxy: str | None) -> dict[str, str] | None:
    """
    Determine the proxy dict for `requests`.

    Priority: CLI flag → POLYMARKET_PROXY env → HTTPS_PROXY/ALL_PROXY env.
    Returns a dict like {"http": url, "https": url} or None.
    """
    url = (
        cli_proxy
        or os.environ.get("POLYMARKET_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("ALL_PROXY")
        or os.environ.get("all_proxy")
    )
    if not url:
        return None

    # Validate SOCKS support is installed
    if url.lower().startswith("socks") and socks is None:
        log.error(
            "SOCKS proxy requested but PySocks is not installed.\n"
            "  pip install requests[socks]   # or: pip install pysocks"
        )
        sys.exit(1)

    log.info("Using proxy: %s", _redact_proxy(url))
    return {"http": url, "https": url}


def resolve_proxy_url(cli_proxy: str | None) -> str | None:
    """Return the raw proxy URL string (for aiohttp / WebSocket)."""
    return (
        cli_proxy
        or os.environ.get("POLYMARKET_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("ALL_PROXY")
        or os.environ.get("all_proxy")
        or None
    )


def _redact_proxy(url: str) -> str:
    """Redact credentials from proxy URL for logging."""
    if "@" in url:
        scheme_end = url.index("://") + 3 if "://" in url else 0
        at_pos = url.index("@")
        return url[:scheme_end] + "***:***@" + url[at_pos + 1 :]
    return url


# Module-level proxy config — set once from CLI/env in main()
_PROXIES: dict[str, str] | None = None

# Shared Session with connection pooling to avoid exhausting the proxy
_SESSION: requests.Session | None = None


def _get_session() -> requests.Session:
    """Return shared requests.Session with a small connection pool."""
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4,
            pool_maxsize=8,
            max_retries=0,
        )
        _SESSION.mount("https://", adapter)
        _SESSION.mount("http://", adapter)
    return _SESSION


def _get(url: str, params: dict | None = None, *, retries: int = 4) -> dict | list:
    """GET with simple retry/backoff and optional proxy.  Returns parsed JSON."""
    session = _get_session()
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=20, proxies=_PROXIES)
            # Don't retry 404s — the resource (expired token) won't come back
            if resp.status_code == 404:
                raise RuntimeError(f"404 Not Found: {url}")
            resp.raise_for_status()
            return resp.json()
        except RuntimeError:
            raise  # Re-raise 404 immediately, no retry
        except requests.RequestException as exc:
            # Longer backoff: 2, 4, 8, 16 seconds — proxy needs time to recover
            wait = 2 ** (attempt + 1)
            if attempt < retries - 1:
                log.warning(
                    "GET %s failed (%s). Retry in %ds…",
                    url.split("?")[0],
                    type(exc).__name__,
                    wait,
                )
                time.sleep(wait)
    raise RuntimeError(f"Failed to GET {url} after {retries} attempts")


def check_geoblock() -> dict:
    """
    Hit Polymarket's geoblock endpoint to see if our IP is restricted.
    Returns {"blocked": bool, "ip": str, "country": str, "region": str}.
    """
    try:
        resp = requests.get(GEOBLOCK_URL, timeout=10, proxies=_PROXIES)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.warning("Geoblock check failed: %s", exc)
        return {"blocked": None, "error": str(exc)}


def find_event_by_slug(slug: str) -> dict | None:
    """Return the Gamma event object for a given slug, or None."""
    data = _get(f"{GAMMA_API}/events", params={"slug": slug})
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict) and data.get("slug") == slug:
        return data
    return None


def search_btc_events(
    interval_minutes: int,
    limit: int = 50,
) -> list[dict]:
    """
    Return Gamma event objects for BTC up/down contracts at the given interval.
    Queries both active and closed markets.
    Results are sorted by end date ascending (oldest first).
    """
    prefix = INTERVAL_SLUG_PREFIX.get(interval_minutes)
    if not prefix:
        log.error("Unsupported interval: %d", interval_minutes)
        return []

    # Build search queries
    if interval_minutes == 5:
        search_terms = ["BTC Up Down 5"]
    else:
        search_terms = ["BTC Up Down 15"]

    results: dict[str, dict] = {}  # keyed by slug to deduplicate

    for q_term in search_terms:
        for closed in (False, True):
            params = {
                "q": q_term,
                "closed": str(closed).lower(),
                "limit": limit,
            }
            try:
                data = _get(f"{GAMMA_API}/events", params=params)
            except RuntimeError as exc:
                log.warning("Event search failed: %s", exc)
                continue

            events = data if isinstance(data, list) else data.get("data", [])
            for ev in events:
                slug = ev.get("slug", "")
                if prefix in slug:
                    results[slug] = ev

            time.sleep(REST_DELAY)

    # Sort by end date (proxy for contract timestamp)
    sorted_events = sorted(
        results.values(),
        key=lambda e: e.get("endDate") or e.get("end_date") or "",
    )
    return sorted_events


def generate_recent_slugs(interval_minutes: int, count: int = 20) -> list[str]:
    """
    Generate deterministic slugs for the most recent `count` contract windows.

    Polymarket slugs follow the pattern:
      btc-updown-{interval}m-{unix_ts}
    where unix_ts is the window start rounded down to the interval boundary.
    """
    prefix = INTERVAL_SLUG_PREFIX.get(interval_minutes)
    if not prefix:
        return []

    interval_sec = interval_minutes * 60
    now = int(time.time())
    current_window_start = now - (now % interval_sec)

    slugs = []
    for i in range(count):
        ts = current_window_start - (i * interval_sec)
        slugs.append(f"{prefix}-{ts}")

    return slugs


def extract_markets_from_event(event: dict) -> list[dict]:
    """
    Pull the list of markets out of a Gamma event object.
    Each market has one outcome (Yes/No or Up/Down).
    """
    markets = event.get("markets", [])
    if not markets:
        # Flat event that *is* the market
        markets = [event]
    return markets


def get_token_ids(market: dict) -> list[dict]:
    """
    Return a list of {outcome, token_id, condition_id} from a Gamma market object.
    Polymarket stores tokens as a JSON string or a list.
    """
    tokens_raw = market.get("tokens") or market.get("clobTokenIds") or []
    if isinstance(tokens_raw, str):
        try:
            tokens_raw = json.loads(tokens_raw)
        except json.JSONDecodeError:
            tokens_raw = []

    condition_id = market.get("conditionId") or market.get("condition_id") or ""

    result = []
    outcomes_raw = market.get("outcomes") or market.get("outcomePrices") or []
    if isinstance(outcomes_raw, str):
        try:
            outcomes_raw = json.loads(outcomes_raw)
        except json.JSONDecodeError:
            outcomes_raw = []

    for i, token in enumerate(tokens_raw):
        if isinstance(token, dict):
            token_id = token.get("token_id") or token.get("tokenId") or ""
            outcome = token.get("outcome", f"outcome_{i}")
        else:
            token_id = str(token)
            outcome = outcomes_raw[i] if i < len(outcomes_raw) else f"outcome_{i}"
        if token_id:
            result.append(
                {
                    "outcome": outcome,
                    "token_id": token_id,
                    "condition_id": condition_id,
                }
            )

    return result


def fetch_price_history(
    token_id: str,
    fidelity: int = 1,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> list[dict]:
    """
    Fetch price history from the CLOB API.
    Returns a list of {"t": unix_timestamp, "p": price_float}.
    fidelity : bucket size in minutes (1 = finest available)
    """
    params: dict = {
        "market": token_id,
        "interval": "all",
        "fidelity": fidelity,
    }
    if start_ts:
        params["startTs"] = start_ts
    if end_ts:
        params["endTs"] = end_ts

    try:
        data = _get(f"{CLOB_API}/prices-history", params=params)
    except RuntimeError as exc:
        log.warning("Price history failed for token %s: %s", token_id[:12], exc)
        return []

    history = data.get("history") or data.get("data") or []
    return history


def fetch_current_price(token_id: str) -> float | None:
    """Return the current mid-point price for a token, or None on error."""
    try:
        data = _get(f"{CLOB_API}/midpoint", params={"token_id": token_id})
        return float(data.get("mid", 0)) or None
    except (RuntimeError, ValueError):
        return None


def fetch_trades(condition_id: str, limit: int = 5000) -> list[dict]:
    """
    Fetch individual matched trades from the public Data API.
    Uses conditionId (not tokenId).  Paginates via limit/offset.
    """
    if not condition_id:
        return []

    all_trades: list[dict] = []
    page_size = min(limit, 500)
    offset = 0

    while len(all_trades) < limit:
        params: dict = {
            "market": condition_id,
            "limit": page_size,
            "offset": offset,
        }
        try:
            data = _get(f"{DATA_API}/trades", params=params)
        except RuntimeError as exc:
            log.warning(
                "Trades fetch failed for condition %s: %s", condition_id[:12], exc
            )
            break

        batch = data if isinstance(data, list) else data.get("data") or []
        all_trades.extend(batch)
        time.sleep(REST_DELAY)

        if len(batch) < page_size or offset + page_size >= 10_000:
            break
        offset += page_size

    return all_trades[:limit]


def write_history_csv(rows: list[dict], output_path: Path) -> int:
    """Write price-history rows to CSV. Returns number of rows written."""
    columns = [
        "interval_minutes",
        "event_slug",
        "outcome",
        "token_id",
        "start_date",
        "end_date",
        "status",
        "timestamp_utc",
        "price",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_trades_csv(rows: list[dict], output_path: Path) -> int:
    """Write individual trade records to CSV. Returns number of rows written."""
    columns = [
        "interval_minutes",
        "event_slug",
        "outcome",
        "token_id",
        "start_date",
        "end_date",
        "status",
        "trade_id",
        "timestamp_utc",
        "price",
        "side",
        "size",
        "maker_address",
        "taker_address",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def open_orderbook_csv(output_path: Path) -> tuple[csv.DictWriter, Any]:
    """Open the orderbook CSV and write header."""
    columns = [
        "timestamp_utc",
        "interval_minutes",
        "event_slug",
        "outcome",
        "token_id",
        "best_bid",
        "best_ask",
        "mid",
        "spread",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(output_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    fh.flush()
    return writer, fh


def write_live_csv_header(output_path: Path) -> tuple[csv.DictWriter, Any]:
    """Open the live CSV and write the header."""
    columns = [
        "interval_minutes",
        "event_slug",
        "outcome",
        "token_id",
        "timestamp_utc",
        "event_type",
        "price",
        "side",
        "size",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(output_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    fh.flush()
    return writer, fh


def detect_interval_from_slug(slug: str) -> int:
    """Detect interval minutes from a slug string."""
    if "updown-5m" in slug or "up-down-5m" in slug.lower():
        return 5
    if "updown-15m" in slug or "up-down-15m" in slug.lower():
        return 15
    return 0


def _discover_events(
    event_slug: str | None,
    interval_minutes: int,
    lookback: int,
) -> list[dict]:
    """Discover events to process, in order (oldest first)."""
    if event_slug:
        log.info("Looking up event: %s", event_slug)
        ev = find_event_by_slug(event_slug)
        if ev is None:
            log.error("Event not found for slug: %s", event_slug)
            sys.exit(1)
        events = [ev]

        # Also search for related past contracts
        detected = detect_interval_from_slug(event_slug) or interval_minutes
        log.info(
            "Searching for related past %dm contracts (lookback=%d)…",
            detected,
            lookback,
        )
        series = search_btc_events(detected, limit=lookback + 10)
        seen_slugs = {event_slug}
        for past_ev in series[-lookback:]:
            if past_ev.get("slug") not in seen_slugs:
                events.append(past_ev)
                seen_slugs.add(past_ev.get("slug"))
        return events

    # Auto-discover
    log.info(
        "Searching for BTC Up/Down %dm contracts (lookback=%d)…",
        interval_minutes,
        lookback,
    )
    events = search_btc_events(interval_minutes, limit=lookback + 10)

    if not events:
        # Fall back to slug generation — try recent windows directly
        log.info("Gamma search returned no results; trying generated slugs…")
        generated = generate_recent_slugs(interval_minutes, count=lookback)
        for slug in generated:
            try:
                ev = find_event_by_slug(slug)
            except RuntimeError as exc:
                log.warning("Slug lookup failed for %s: %s", slug, exc)
                ev = None
            if ev:
                events.append(ev)
            time.sleep(REST_DELAY)
            if len(events) >= lookback:
                break

    if not events:
        log.error("No BTC %dm events found. Try --event-slug.", interval_minutes)
        sys.exit(1)

    log.info("Found %d contracts.", len(events))
    return events


def _discover_events_multi(
    event_slug: str | None,
    intervals: list[int],
    lookback: int,
) -> list[tuple[int, dict]]:
    """
    Discover events for one or more intervals.
    Returns list of (interval_minutes, event_dict).
    """
    result: list[tuple[int, dict]] = []
    seen_slugs: set[str] = set()

    for iv in intervals:
        events = _discover_events(event_slug, iv, lookback)
        for ev in events:
            slug = ev.get("slug", "")
            if slug not in seen_slugs:
                detected_iv = detect_interval_from_slug(slug) or iv
                result.append((detected_iv, ev))
                seen_slugs.add(slug)

    # Sort by end date
    result.sort(key=lambda x: x[1].get("endDate") or x[1].get("end_date") or "")
    return result


def collect_history(
    event_slug: str | None,
    intervals: list[int],
    lookback: int,
    fidelity: int,
    output_dir: str,
) -> tuple[list[dict], list[Path]]:
    """
    Fetch aggregated price-history snapshots for Up + Down tokens across
    all requested intervals.

    Returns (active_token_metadata, [output_paths]).
    """
    events_with_iv = _discover_events_multi(event_slug, intervals, lookback)
    active_tokens: list[dict] = []
    rows_by_interval: dict[int, list[dict]] = {}

    for iv, ev in events_with_iv:
        slug = ev.get("slug", "unknown")
        status = (
            "active"
            if ev.get("active")
            else ("closed" if ev.get("closed") else "resolved")
        )
        start_date = ev.get("startDate") or ev.get("start_date") or ""
        end_date = ev.get("endDate") or ev.get("end_date") or ""

        markets = extract_markets_from_event(ev)
        log.info("  %-55s  %s  (%d market(s))", slug, status, len(markets))

        for mkt in markets:
            token_pairs = get_token_ids(mkt)
            if not token_pairs:
                log.warning("    No token IDs found in market: %s", mkt.get("id", "?"))
                continue

            for tp in token_pairs:
                outcome = tp["outcome"]
                token_id = tp["token_id"]

                log.info(
                    "    [%dm] Fetching price history: %-6s  token=%s…",
                    iv,
                    outcome,
                    token_id[:12],
                )
                history = fetch_price_history(token_id, fidelity=fidelity)
                time.sleep(REST_DELAY)

                if not history:
                    mid = fetch_current_price(token_id)
                    if mid is not None:
                        history = [{"t": int(time.time()), "p": mid}]
                    time.sleep(REST_DELAY)

                for point in history:
                    ts = point.get("t") or point.get("timestamp") or 0
                    price = point.get("p") or point.get("price") or 0
                    ts_utc = (
                        datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
                        if ts
                        else ""
                    )
                    rows_by_interval.setdefault(iv, []).append(
                        {
                            "interval_minutes": iv,
                            "event_slug": slug,
                            "outcome": outcome,
                            "token_id": token_id,
                            "start_date": start_date,
                            "end_date": end_date,
                            "status": status,
                            "timestamp_utc": ts_utc,
                            "price": price,
                        }
                    )

                if status == "active":
                    active_tokens.append(
                        {
                            "interval_minutes": iv,
                            "event_slug": slug,
                            "outcome": outcome,
                            "token_id": token_id,
                        }
                    )

    # Write one CSV per interval
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_paths: list[Path] = []
    for iv, rows in rows_by_interval.items():
        path = Path(output_dir) / f"polymarket_btc{iv}m_history_{ts_tag}.csv"
        n = write_history_csv(rows, path)
        log.info("Wrote %d history rows → %s", n, path)
        output_paths.append(path)

    return active_tokens, output_paths


def collect_trades(
    event_slug: str | None,
    intervals: list[int],
    lookback: int,
    trades_limit: int,
    output_dir: str,
) -> list[dict]:
    """
    Fetch individual matched trades for Up + Down tokens across intervals.
    Returns the active-token metadata list.
    """
    events_with_iv = _discover_events_multi(event_slug, intervals, lookback)
    active_tokens: list[dict] = []
    rows_by_interval: dict[int, list[dict]] = {}

    for iv, ev in events_with_iv:
        slug = ev.get("slug", "unknown")
        status = (
            "active"
            if ev.get("active")
            else ("closed" if ev.get("closed") else "resolved")
        )
        start_date = ev.get("startDate") or ev.get("start_date") or ""
        end_date = ev.get("endDate") or ev.get("end_date") or ""

        markets = extract_markets_from_event(ev)
        log.info("  %-55s  %s  (%d market(s))", slug, status, len(markets))

        for mkt in markets:
            token_pairs = get_token_ids(mkt)
            if not token_pairs:
                continue

            condition_id = token_pairs[0].get("condition_id", "") if token_pairs else ""
            token_id_to_outcome = {tp["token_id"]: tp["outcome"] for tp in token_pairs}

            log.info(
                "    [%dm] Fetching trades:  condition=%s…",
                iv,
                condition_id[:12] if condition_id else "?",
            )
            trades = fetch_trades(condition_id, limit=trades_limit)
            log.info("      → %d trades", len(trades))

            for t in trades:
                raw_ts = t.get("timestamp") or t.get("created_at") or 0
                if isinstance(raw_ts, (int, float)) and raw_ts:
                    ts_utc = datetime.fromtimestamp(
                        int(raw_ts), tz=timezone.utc
                    ).isoformat()
                else:
                    ts_utc = str(raw_ts)

                asset = t.get("asset") or t.get("token_id") or ""
                outcome = token_id_to_outcome.get(asset, asset[:8] if asset else "")
                proxy_wallet = t.get("proxyWallet") or ""
                trade_id = (
                    t.get("id") or f"{condition_id[:8]}_{raw_ts}_{proxy_wallet[:8]}"
                )

                rows_by_interval.setdefault(iv, []).append(
                    {
                        "interval_minutes": iv,
                        "event_slug": slug,
                        "outcome": outcome,
                        "token_id": asset,
                        "start_date": start_date,
                        "end_date": end_date,
                        "status": status,
                        "trade_id": trade_id,
                        "timestamp_utc": ts_utc,
                        "price": t.get("price") or "",
                        "side": t.get("side") or "",
                        "size": t.get("size") or "",
                        "maker_address": t.get("maker_address") or proxy_wallet,
                        "taker_address": t.get("taker_address") or "",
                    }
                )

            for tp in token_pairs:
                if status == "active":
                    active_tokens.append(
                        {
                            "interval_minutes": iv,
                            "event_slug": slug,
                            "outcome": tp["outcome"],
                            "token_id": tp["token_id"],
                            "condition_id": condition_id,
                        }
                    )

    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    for iv, rows in rows_by_interval.items():
        path = Path(output_dir) / f"polymarket_btc{iv}m_trades_{ts_tag}.csv"
        n = write_trades_csv(rows, path)
        log.info("Wrote %d trade rows → %s", n, path)

    return active_tokens


async def _create_ws_session(proxy_url: str | None) -> aiohttp.ClientSession:
    """Create an aiohttp session, optionally through a SOCKS/HTTP proxy."""
    if proxy_url and ProxyConnector is not None:
        connector = ProxyConnector.from_url(proxy_url)
        return aiohttp.ClientSession(connector=connector)
    elif proxy_url and ProxyConnector is None:
        log.warning(
            "aiohttp-socks not installed — WebSocket will NOT use proxy.\n"
            "  pip install aiohttp-socks"
        )
    return aiohttp.ClientSession()


async def stream_live_ticks(
    active_tokens: list[dict],
    output_dir: str,
    shutdown_event: asyncio.Event,
    proxy_url: str | None = None,
    intervals: list[int] | None = None,
) -> None:
    """
    Subscribe to live trade events via WebSocket and write to CSV.

    Periodically reconnects with fresh token subscriptions so we don't
    waste the session streaming dead (expired) 5m/15m contract tokens.
    """
    if not active_tokens and not intervals:
        log.warning("No active tokens found — skipping live stream.")
        return

    if not intervals:
        intervals = sorted({t.get("interval_minutes", 15) for t in active_tokens})

    # Filename setup
    iv_label = "_".join(str(i) for i in sorted(intervals)) if intervals else "mix"
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    live_path = Path(output_dir) / f"polymarket_btc{iv_label}m_live_{ts_tag}.csv"
    writer, fh = write_live_csv_header(live_path)
    log.info("Live output: %s", live_path)

    resubscribe_interval = 60.0
    backoff = 1

    # Use passed-in tokens initially, then always refresh from current window
    current_tokens = list(active_tokens) if active_tokens else []

    while not shutdown_event.is_set():
        # Refresh active tokens for the current window
        try:
            fresh = _discover_current_active_tokens(intervals)
            if fresh:
                current_tokens = fresh
        except Exception as exc:
            log.warning("WS token refresh failed: %s", exc)

        if not current_tokens:
            log.warning("No active tokens for WS; retrying in 30s…")
            await asyncio.sleep(30)
            continue

        token_map = {t["token_id"]: t for t in current_tokens}
        asset_ids = list(token_map.keys())

        sub_payload = json.dumps(
            {
                "auth": {},
                "type": "Market",
                "assets_ids": asset_ids,
            }
        )

        try:
            async with await _create_ws_session(proxy_url) as session:
                async with session.ws_connect(WS_URL, heartbeat=20) as ws:
                    await ws.send_str(sub_payload)
                    log.info(
                        "WebSocket subscribed to %d tokens (will refresh in %.0fs)",
                        len(asset_ids),
                        resubscribe_interval,
                    )
                    backoff = 1

                    # Create a task that sets a flag after resubscribe_interval
                    resub_deadline = time.time() + resubscribe_interval

                    async for msg in ws:
                        if shutdown_event.is_set():
                            break
                        # Time to resubscribe with fresh tokens?
                        if time.time() >= resub_deadline:
                            log.info("Resubscribing WebSocket with fresh tokens…")
                            break

                        if msg.type == aiohttp.WSMsgType.ERROR:
                            raise RuntimeError(f"WS error frame: {ws.exception()}")
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue

                        try:
                            data = json.loads(msg.data)
                        except json.JSONDecodeError:
                            continue

                        events_list = data if isinstance(data, list) else [data]
                        for event in events_list:
                            etype = event.get("event_type") or event.get("type") or ""
                            # Only keep trade events — skip orderbook snapshots
                            # (those are covered by the orderbook poller)
                            if etype not in (
                                "trade",
                                "last_trade_price",
                                "price_change",
                            ):
                                continue

                            asset_id = event.get("asset_id") or ""
                            # Skip events for tokens we didn't subscribe to
                            # (filters out book events keyed by condition_id)
                            if not asset_id or asset_id not in token_map:
                                continue
                            meta = token_map[asset_id]

                            price = (
                                event.get("price")
                                or event.get("last_trade_price")
                                or ""
                            )
                            side = event.get("side", "")
                            size = event.get("size", "")
                            raw_ts = event.get("timestamp") or int(time.time())
                            try:
                                # Polymarket timestamps are sometimes in milliseconds
                                raw_ts_int = int(raw_ts)
                                if raw_ts_int > 10_000_000_000:
                                    raw_ts_int = raw_ts_int // 1000
                                ts_utc = datetime.fromtimestamp(
                                    raw_ts_int, tz=timezone.utc
                                ).isoformat()
                            except (ValueError, TypeError):
                                ts_utc = datetime.now(timezone.utc).isoformat()

                            row = {
                                "interval_minutes": meta.get("interval_minutes", ""),
                                "event_slug": meta.get("event_slug", ""),
                                "outcome": meta.get("outcome", ""),
                                "token_id": asset_id,
                                "timestamp_utc": ts_utc,
                                "event_type": etype,
                                "price": price,
                                "side": side,
                                "size": size,
                            }
                            writer.writerow(row)
                            fh.flush()

        except Exception as exc:
            if shutdown_event.is_set():
                break
            log.warning(
                "WS error (%s). Reconnecting in %ds…", type(exc).__name__, backoff
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

    fh.close()
    log.info("Live stream ended.")


def _discover_current_active_tokens(intervals: list[int]) -> list[dict]:
    """
    Discover tokens for the CURRENTLY active contract windows.
    Uses deterministic slug generation to find the live contract,
    falling back to Gamma search.

    Called repeatedly by the live poller to refresh stale tokens.
    """
    active: list[dict] = []
    for iv in intervals:
        interval_sec = iv * 60
        now = int(time.time())
        current_window_start = now - (now % interval_sec)
        prefix = INTERVAL_SLUG_PREFIX.get(iv, f"btc-updown-{iv}m")
        slug = f"{prefix}-{current_window_start}"

        ev = None
        try:
            ev = find_event_by_slug(slug)
        except RuntimeError:
            pass
        if not ev:
            # Try next window (might already be listed)
            next_ts = current_window_start + interval_sec
            slug = f"{prefix}-{next_ts}"
            try:
                ev = find_event_by_slug(slug)
            except RuntimeError:
                pass
        if not ev:
            continue

        status = (
            "active"
            if ev.get("active")
            else ("closed" if ev.get("closed") else "resolved")
        )
        if status != "active":
            continue

        for mkt in extract_markets_from_event(ev):
            for tp in get_token_ids(mkt):
                condition_id = tp.get("condition_id", "")
                active.append(
                    {
                        "interval_minutes": iv,
                        "event_slug": slug,
                        "outcome": tp["outcome"],
                        "token_id": tp["token_id"],
                        "condition_id": condition_id,
                    }
                )

    return active


async def poll_orderbook(
    active_tokens: list[dict],
    output_dir: str,
    poll_interval: float,
    shutdown_event: asyncio.Event,
    intervals: list[int] | None = None,
) -> None:
    """
    Poll the CLOB orderbook at regular intervals for each active token.

    Auto-refreshes the token list every cycle so that when a 5m/15m
    contract expires, we pick up the next one instead of 404-ing.
    """
    if not active_tokens and not intervals:
        return

    # Infer intervals from the initial token list if not provided
    if not intervals:
        intervals = sorted({t.get("interval_minutes", 15) for t in active_tokens})

    ivs = sorted(intervals)
    iv_label = "_".join(str(i) for i in ivs) if ivs else "mix"

    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    ob_path = Path(output_dir) / f"polymarket_btc{iv_label}m_orderbook_{ts_tag}.csv"
    writer, fh = open_orderbook_csv(ob_path)
    log.info("Orderbook output: %s  (poll every %.0fs)", ob_path, poll_interval)

    # Track which tokens have 404'd so we don't spam logs
    dead_tokens: set[str] = set()
    last_refresh = 0.0
    refresh_interval = 60.0

    current_tokens = list(active_tokens)

    while not shutdown_event.is_set():
        # Periodically refresh the active token list
        now = time.time()
        if now - last_refresh > refresh_interval:
            try:
                fresh = _discover_current_active_tokens(intervals)
                if fresh:
                    current_tokens = fresh
                    dead_tokens.clear()
                    log.info(
                        "Refreshed active tokens: %d token(s) across %s",
                        len(fresh),
                        [t["event_slug"] for t in fresh[:2]],
                    )
            except Exception as exc:
                log.warning("Token refresh failed: %s", exc)
            last_refresh = now

        for tok in current_tokens:
            if shutdown_event.is_set():
                break
            token_id = tok["token_id"]
            if token_id in dead_tokens:
                continue

            try:
                data = _get(f"{CLOB_API}/book", params={"token_id": token_id})
            except RuntimeError:
                log.debug("Token %s expired (404), skipping.", token_id[:16])
                dead_tokens.add(token_id)
                continue

            bids = data.get("bids") or []
            asks = data.get("asks") or []

            def _price(entry: dict | list) -> float:
                if isinstance(entry, dict):
                    return float(entry.get("price") or entry.get("p") or 0)
                return float(entry[0]) if entry else 0.0

            best_bid = _price(bids[0]) if bids else 0.0
            best_ask = _price(asks[0]) if asks else 0.0
            mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0
            spread = best_ask - best_bid if best_bid and best_ask else 0.0

            ts_utc = datetime.now(timezone.utc).isoformat()
            writer.writerow(
                {
                    "timestamp_utc": ts_utc,
                    "interval_minutes": tok.get("interval_minutes", ""),
                    "event_slug": tok.get("event_slug", ""),
                    "outcome": tok.get("outcome", ""),
                    "token_id": token_id,
                    "best_bid": f"{best_bid:.4f}",
                    "best_ask": f"{best_ask:.4f}",
                    "mid": f"{mid:.4f}",
                    "spread": f"{spread:.4f}",
                }
            )
            fh.flush()

        await asyncio.sleep(poll_interval)

    fh.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Polymarket BTC Up/Down tick data (5m + 15m).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--event-slug",
        default=None,
        metavar="SLUG",
        help=(
            "Polymarket event slug (e.g. btc-updown-5m-1775263800). "
            "Defaults to auto-searching for the latest contracts."
        ),
    )
    parser.add_argument(
        "--interval",
        default="15",
        metavar="{5,15,all}",
        help=(
            "Contract interval: 5 (5-minute), 15 (15-minute), or all. " "Default: 15"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path.cwd()),
        metavar="DIR",
        help="Directory to write CSV files (default: current directory)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=20,
        metavar="N",
        help="Number of past contracts to fetch history for (default: 20)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="After fetching history/trades, stream live ticks via WebSocket",
    )
    parser.add_argument(
        "--trades-only",
        action="store_true",
        help="Skip price-history; only fetch individual CLOB trade records",
    )
    parser.add_argument(
        "--fidelity",
        type=int,
        default=1,
        metavar="MINUTES",
        help="Price history bucket size in minutes (default: 1)",
    )
    parser.add_argument(
        "--trades-limit",
        type=int,
        default=5000,
        metavar="N",
        help="Max individual trades to fetch per market (default: 5000)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Orderbook poll frequency in seconds when --live (default: 5)",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        metavar="URL",
        help=(
            "SOCKS5 or HTTP(S) proxy URL. Examples:\n"
            "  socks5://user:pass@host:1080\n"
            "  http://host:8080\n"
            "Also reads POLYMARKET_PROXY, HTTPS_PROXY, or ALL_PROXY env vars."
        ),
    )
    parser.add_argument(
        "--check-geo",
        action="store_true",
        help="Check Polymarket geoblock status for your IP before fetching",
    )
    return parser.parse_args()


def parse_intervals(raw: str) -> list[int]:
    """Parse the --interval argument into a list of interval minutes."""
    raw = raw.strip().lower()
    if raw == "all":
        return [5, 15]
    try:
        val = int(raw)
        if val not in (5, 15):
            log.error("Unsupported interval %d — use 5, 15, or all", val)
            sys.exit(1)
        return [val]
    except ValueError:
        log.error("Invalid --interval value: %s", raw)
        sys.exit(1)


async def async_main() -> None:
    global _PROXIES
    args = parse_args()

    # proxy setup
    _PROXIES = resolve_proxy(args.proxy)
    proxy_url = resolve_proxy_url(args.proxy)

    # geoblock check
    if args.check_geo:
        geo = check_geoblock()
        blocked = geo.get("blocked")
        ip_addr = geo.get("ip", "?")
        country = geo.get("country", "?")
        region = geo.get("region", "?")

        if blocked is True:
            log.warning(
                "   GEOBLOCKED — IP %s (%s/%s) is restricted by Polymarket.\n"
                "   Read-only API calls may still work, but consider:\n"
                "     - Using --proxy with a non-US SOCKS5/HTTP proxy\n"
                "     - Setting up WireGuard/OpenVPN on this server\n"
                "     - Using POLYMARKET_PROXY environment variable",
                ip_addr,
                country,
                region,
            )
        elif blocked is False:
            log.info(
                "SUCCESS - Not geoblocked — IP %s (%s/%s)", ip_addr, country, region
            )
        else:
            log.warning("Geoblock check inconclusive: %s", geo)

    # Parse Intervals
    intervals = parse_intervals(args.interval)
    log.info("Intervals: %s", [f"{i}m" for i in intervals])

    # Historical Pricing Snapshots
    if not args.trades_only:
        active_tokens, _ = collect_history(
            event_slug=args.event_slug,
            intervals=intervals,
            lookback=args.lookback,
            fidelity=args.fidelity,
            output_dir=args.output_dir,
        )
    else:
        active_tokens = []

    # Individual CLOB trades
    trade_active_tokens = collect_trades(
        event_slug=args.event_slug,
        intervals=intervals,
        lookback=args.lookback,
        trades_limit=args.trades_limit,
        output_dir=args.output_dir,
    )

    # Merge active-token lists
    seen = {t["token_id"] for t in active_tokens}
    for t in trade_active_tokens:
        if t["token_id"] not in seen:
            active_tokens.append(t)
            seen.add(t["token_id"])

    # Live straem + orderbook poll
    if args.live:
        shutdown_event = asyncio.Event()
        try:
            await asyncio.gather(
                stream_live_ticks(
                    active_tokens,
                    args.output_dir,
                    shutdown_event,
                    proxy_url=proxy_url,
                    intervals=intervals,
                ),
                poll_orderbook(
                    active_tokens,
                    args.output_dir,
                    args.poll_interval,
                    shutdown_event,
                    intervals=intervals,
                ),
            )
        except KeyboardInterrupt:
            print("\nShutdown requested.")
            shutdown_event.set()


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
