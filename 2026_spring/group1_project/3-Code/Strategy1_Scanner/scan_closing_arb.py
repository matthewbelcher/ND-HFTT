#!/usr/bin/env python3
"""
Scan Kalshi events closing/settling in the next N days and test each event
with a known exact YES-winner count for full-set arbitrage. Events with unknown
winner counts are skipped unless configured or overridden.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

# Max seconds to wait for arb computation per event; prevents random hangs (e.g. deep orderbooks)
ARB_COMPUTE_TIMEOUT_SEC = 3

# Ensure Arbitrage dir is on path so full_set_arb can be imported from project root or Arbitrage/
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# Import arb logic from full_set_arb (one-way; we do not need full_set_arb to import us)
from full_set_arb import (
    fetch_event_and_orderbooks_from_api,
    run_full_set_arb,
)
from resolution import DEFAULT_RESOLUTION_CONFIG, infer_resolution_spec, load_resolution_overrides

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# File of event tickers that had arb last run; these are tested first on next run
ARB_FOUND_FILENAME = "arb_found.txt"


def load_priority_tickers(filename: str = ARB_FOUND_FILENAME) -> list[str]:
    """Load event tickers from file (one per line, # = comment). Returns list of non-empty tickers."""
    path = os.path.join(_SCRIPT_DIR, filename)
    if not os.path.isfile(path):
        return []
    tickers: list[str] = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip().split("#")[0].strip()
                if line:
                    tickers.append(line)
    except OSError:
        pass
    return tickers


def save_arb_found_tickers(
    tickers: list[str],
    filename: str = ARB_FOUND_FILENAME,
    merge_with_existing: bool = True,
) -> None:
    """Write event tickers to file (one per line) for next run's priority list.
    If merge_with_existing, loads existing tickers and only adds new ones (no duplicates).
    """
    path = os.path.join(_SCRIPT_DIR, filename)
    existing: list[str] = []
    if merge_with_existing:
        existing = load_priority_tickers(filename)
    seen: set[str] = set()
    combined: list[str] = []
    for t in existing:
        t = (t or "").strip()
        if t and t not in seen:
            seen.add(t)
            combined.append(t)
    for t in tickers:
        t = (t or "").strip()
        if t and t not in seen:
            seen.add(t)
            combined.append(t)
    try:
        with open(path, "w") as f:
            f.write("# Event tickers with NO arb (tested first on next run)\n")
            for t in combined:
                f.write(t + "\n")
    except OSError:
        pass


def load_amount_from_file(filename: str = "amount.txt") -> float | None:
    """
    Read USD amount from a file in the script directory (e.g. amount.txt with line amount=100).
    Returns None if file missing or no valid amount= line.
    """
    path = os.path.join(_SCRIPT_DIR, filename)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    if key.strip().lower() == "amount":
                        return float(val.strip())
    except (OSError, ValueError):
        pass
    return None


# --- Local API helpers (avoid circular imports; used only for GET /events) ---

KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def _http_get(url: str, timeout: int = 15) -> dict[str, Any] | None:
    """GET URL and return JSON. Returns None on failure."""
    if _HAS_REQUESTS:
        try:
            r = requests.get(url, headers={"accept": "application/json"}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"Request failed {url}: {e}", file=sys.stderr)
            return None
    try:
        from urllib.request import Request, urlopen
        req = Request(url, headers={"accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"Request failed {url}: {e}", file=sys.stderr)
        return None


def fetch_events_page(cursor: str | None = None, limit: int = 200) -> dict[str, Any] | None:
    """
    Fetch one page of open events with nested markets.
    Returns {"events": [...], "cursor": "..."} or None.
    """
    params = [
        "with_nested_markets=true",
        "status=open",
        f"limit={limit}",
    ]
    if cursor:
        params.append(f"cursor={cursor}")
    url = f"{KALSHI_API_BASE}/events?{'&'.join(params)}"
    data = _http_get(url)
    if data is None:
        return None
    return {
        "events": data.get("events", []),
        "cursor": data.get("cursor") or "",
    }


def _parse_close_time(close_time_str: str | None) -> datetime | None:
    """Parse ISO close_time to UTC datetime. Handles Z suffix."""
    if not close_time_str:
        return None
    s = (close_time_str or "").strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def iter_events_closing_in_days(days: int = 10):
    """
    Fetch open events (paginated) and yield each event that has at least
    one market whose close_time is in [now_utc, now_utc + days].
    Yields event dicts (with nested markets) as they are found.
    """
    now_utc = datetime.now(timezone.utc)
    end_utc = now_utc + timedelta(days=days)
    cursor: str | None = None

    while True:
        page = fetch_events_page(cursor=cursor)
        if page is None:
            break
        events = page.get("events") or []
        for ev in events:
            markets = ev.get("markets") or []
            if not markets:
                continue
            for m in markets:
                ct = _parse_close_time(m.get("close_time"))
                if ct is None:
                    continue
                if now_utc <= ct <= end_utc:
                    yield ev
                    break
        cursor = page.get("cursor") or ""
        if not cursor:
            break
        time.sleep(0.1)


def _default_days_until_april_first() -> int:
    """Default scan window: from today through April 1 (next occurrence)."""
    today = date.today()
    year = today.year
    if (today.month, today.day) > (4, 1):
        year += 1
    april_first = date(year, 4, 1)
    days = (april_first - today).days + 1  # +1 so we include all of April 1
    return max(1, min(days, 365))  # clamp to [1, 365]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan mutually exclusive events closing in the next N days for full-set arbitrage.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        metavar="N",
        help="Events closing within the next N days (default: through April 1)",
    )
    parser.add_argument(
        "--fee-mode",
        choices=["taker", "maker"],
        default="taker",
        help="Fee mode for arb calculation (default taker)",
    )
    parser.add_argument(
        "--limit-events",
        type=int,
        default=None,
        metavar="N",
        help="Max number of events with known exact winner counts to check (default no limit)",
    )
    parser.add_argument(
        "--n-yes",
        type=int,
        default=None,
        dest="n_yes",
        metavar="M",
        help="Exactly M markets resolve YES (for non-mutually-exclusive multi-winner events). YES payout=M, NO payout=k-M.",
    )
    parser.add_argument(
        "--exact-yes",
        type=int,
        default=None,
        dest="n_yes",
        metavar="M",
        help="Same as --n-yes: use for events with exactly M winners (e.g. top 3).",
    )
    parser.add_argument(
        "--resolution-config",
        type=str,
        default=os.path.join(_SCRIPT_DIR, DEFAULT_RESOLUTION_CONFIG),
        help="JSON file with event/series yes_count overrides.",
    )
    parser.add_argument(
        "--no-inferred-resolution",
        action="store_true",
        help="Only scan Kalshi mutually_exclusive events and explicit config/CLI winner counts.",
    )
    args = parser.parse_args()
    if args.days is None:
        args.days = _default_days_until_april_first()
    resolution_config = load_resolution_overrides(args.resolution_config)
    allow_inferred_resolution = not args.no_inferred_resolution

    amount_usd = load_amount_from_file()
    if amount_usd is not None and amount_usd > 0:
        print(f"Max amount (from amount.txt): ${amount_usd:.2f} USD - we use as much as maximizes profit (up to this cap) per event.")
    else:
        print("No amount.txt (or invalid amount=). Profit shown per set only; set amount=100 in amount.txt for budget-based profit.")
    # Event tickers to skip (e.g. known to hang or not of interest)
    skip_event_tickers: set[str] = {"KXSPOTIFYALBUMW-26FEB26"}

    arb_opportunities: list[tuple[str, str, bool, bool, float | None, float | None, bool, int, int, int, dict]] = []
    already_checked: set[str] = set()
    skipped_unknown_resolution = 0
    skipped_invalid_resolution = 0

    # 1) Test priority events first (tickers that had arb last run)
    priority_tickers = load_priority_tickers()
    if priority_tickers:
        print(f"Priority: testing {len(priority_tickers)} event(s) from {ARB_FOUND_FILENAME} first...")
        for event_ticker in priority_tickers:
            event_ticker = (event_ticker or "").strip()
            if not event_ticker or event_ticker in skip_event_tickers:
                continue
            already_checked.add(event_ticker)
            event_dict, tradeable, orderbooks_by_ticker = fetch_event_and_orderbooks_from_api(event_ticker)
            title = (event_dict.get("title") or event_dict.get("sub_title") or "(no title)")
            print(f"(priority) {event_ticker} - {title}")
            if not tradeable:
                print("  -> No tradeable markets")
                print("---")
                continue
            spec = infer_resolution_spec(
                event_dict,
                tradeable,
                override_yes_count=args.n_yes,
                config=resolution_config,
                allow_inferred=allow_inferred_resolution,
            )
            if not spec.is_exact:
                print(f"  -> Skipped (winner count unknown: {spec.reason})")
                print("---")
                if spec.source == "invalid":
                    skipped_invalid_resolution += 1
                else:
                    skipped_unknown_resolution += 1
                continue
            print(f"  Winner count: exactly {spec.yes_count} YES ({spec.source})")
            print("  Computing arb...", flush=True)
            result_holder: list[dict] = []

            def _run_arb_pri() -> None:
                result_holder.append(
                    run_full_set_arb(
                        event_ticker=event_ticker,
                        event=event_dict,
                        markets=tradeable,
                        orderbooks_by_ticker=orderbooks_by_ticker,
                        fee_mode=args.fee_mode,
                        contracts_per_set=1,
                        budget_dollars=amount_usd,
                        n_yes_resolves=spec.yes_count,
                    )
                )

            arb_thread = threading.Thread(target=_run_arb_pri, daemon=True)
            arb_thread.start()
            arb_thread.join(timeout=ARB_COMPUTE_TIMEOUT_SEC)
            if arb_thread.is_alive():
                print(f"  -> Skipped (arb timed out after {ARB_COMPUTE_TIMEOUT_SEC}s)", flush=True)
                print("---")
                continue
            if not result_holder:
                print("  -> No result", flush=True)
                print("---")
                continue
            result = result_holder[0]
            result["resolution_source"] = spec.source
            result["resolution_reason"] = spec.reason
            summary = result.get("summary") or {}
            if summary.get("error"):
                print(f"  -> {summary['error']}")
                print("---")
                continue
            arb_yes = summary.get("arb_exists_yes", False)
            arb_no = summary.get("arb_exists_no", False)
            if arb_yes or arb_no:
                partial = result.get("partial_cover", False)
                k_yes = result.get("k_yes", result.get("k", 0))
                k_no = result.get("k_no", result.get("k", 0))
                k_total = result.get("k_total", max(k_yes, k_no))
                arb_opportunities.append((
                    event_ticker,
                    title,
                    arb_yes,
                    arb_no,
                    summary.get("profit_per_set_yes_worst"),
                    summary.get("profit_per_set_no_worst"),
                    partial,
                    k_yes,
                    k_no,
                    k_total,
                    result,
                ))
                if arb_no:
                    save_arb_found_tickers([event_ticker], merge_with_existing=True)
            else:
                print("  -> No arb")
            print("---")
        if priority_tickers:
            print("")

    print(f"Scanning events closing in the next {args.days} days (known exact winner counts only)...")
    count = 0  # number of analyzable events examined (limit applies to this)
    for ev in iter_events_closing_in_days(days=args.days):
        event_ticker = ev.get("event_ticker") or ""
        title = ev.get("title") or ev.get("sub_title") or "(no title)"

        initial_markets = ev.get("markets") if isinstance(ev.get("markets"), list) else []
        initial_spec = infer_resolution_spec(
            ev,
            initial_markets,
            override_yes_count=args.n_yes,
            config=resolution_config,
            allow_inferred=allow_inferred_resolution,
        )
        if not initial_spec.is_exact:
            if initial_spec.source == "invalid":
                skipped_invalid_resolution += 1
            else:
                skipped_unknown_resolution += 1
            continue
        if event_ticker in skip_event_tickers:
            continue
        if event_ticker in already_checked:
            continue

        if args.limit_events is not None and count >= args.limit_events:
            break
        count += 1
        progress = f"{count}/{args.limit_events}" if args.limit_events is not None else str(count)
        print(f"({progress}) Resolvable event: {event_ticker} - {title}")
        print(f"  Winner count: exactly {initial_spec.yes_count} YES ({initial_spec.source})")

        # Use same data source as full_set_arb --fetch so results match (full event + all markets + orderbooks)
        event_dict, tradeable, orderbooks_by_ticker = fetch_event_and_orderbooks_from_api(event_ticker)
        if not tradeable:
            continue
        spec = infer_resolution_spec(
            event_dict,
            tradeable,
            override_yes_count=args.n_yes,
            config=resolution_config,
            allow_inferred=allow_inferred_resolution,
        )
        if not spec.is_exact:
            print(f"  -> Skipped after fetch (winner count unknown: {spec.reason})")
            print("---")
            if spec.source == "invalid":
                skipped_invalid_resolution += 1
            else:
                skipped_unknown_resolution += 1
            continue
        if spec.yes_count != initial_spec.yes_count:
            print(f"  -> Skipped (winner count changed after fetch: {initial_spec.yes_count} -> {spec.yes_count})")
            print("---")
            skipped_invalid_resolution += 1
            continue

        print("  Computing arb...", flush=True)
        result_holder: list[dict] = []

        def _run_arb() -> None:
            result_holder.append(
                run_full_set_arb(
                    event_ticker=event_ticker,
                    event=event_dict,
                    markets=tradeable,
                    orderbooks_by_ticker=orderbooks_by_ticker,
                    fee_mode=args.fee_mode,
                    contracts_per_set=1,
                    budget_dollars=amount_usd,
                    n_yes_resolves=spec.yes_count,
                )
            )

        arb_thread = threading.Thread(target=_run_arb, daemon=True)
        arb_thread.start()
        arb_thread.join(timeout=ARB_COMPUTE_TIMEOUT_SEC)
        if arb_thread.is_alive():
            print(f"  -> Skipped (arb timed out after {ARB_COMPUTE_TIMEOUT_SEC}s)", flush=True)
            print("---")
            continue
        if not result_holder:
            print("  -> No result", flush=True)
            print("---")
            continue

        result = result_holder[0]
        result["resolution_source"] = spec.source
        result["resolution_reason"] = spec.reason
        summary = result.get("summary") or {}
        if summary.get("error"):
            print(f"  -> {summary['error']}")
            print("---")
            continue
        arb_yes = summary.get("arb_exists_yes", False)
        arb_no = summary.get("arb_exists_no", False)
        if arb_yes or arb_no:
            partial = result.get("partial_cover", False)
            k_yes = result.get("k_yes", result.get("k", 0))
            k_no = result.get("k_no", result.get("k", 0))
            k_total = result.get("k_total", max(k_yes, k_no))
            arb_opportunities.append((
                event_ticker,
                title,
                arb_yes,
                arb_no,
                summary.get("profit_per_set_yes_worst"),
                summary.get("profit_per_set_no_worst"),
                partial,
                k_yes,
                k_no,
                k_total,
                result,
            ))
            if arb_no:
                save_arb_found_tickers([event_ticker], merge_with_existing=True)
        else:
            print("  -> No arb")
        print("---")

    print(f"Done. Checked {count} event(s) with known exact winner counts.")
    if skipped_unknown_resolution or skipped_invalid_resolution:
        print(
            f"Skipped {skipped_unknown_resolution} event(s) with unknown winner counts "
            f"and {skipped_invalid_resolution} with invalid/conflicting counts."
        )
    # Save tickers that had NO arb so they are tested first next run
    no_arb_tickers = [ticker for ticker, _, _, arb_no, *_ in arb_opportunities if arb_no]
    if no_arb_tickers:
        save_arb_found_tickers(no_arb_tickers)
    return 0


if __name__ == "__main__":
    sys.exit(main())

