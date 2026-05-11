#!/usr/bin/env python3
"""Fetch a Kalshi event by ticker (with nested markets) and write JSON to a file."""

import argparse
import json
import sys

try:
    import requests
except ImportError:
    requests = None

API_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def fetch_event(event_ticker: str, with_nested_markets: bool = True) -> dict | None:
    url = f"{API_BASE}/events/{event_ticker}"
    if with_nested_markets:
        url += "?with_nested_markets=true"
    if requests:
        try:
            r = requests.get(url, headers={"accept": "application/json"}, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"Request failed: {e}", file=sys.stderr)
            return None
    from urllib.request import Request, urlopen
    try:
        req = Request(url, headers={"accept": "application/json"})
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Kalshi event data and write to file.")
    parser.add_argument("event_ticker", help="Event ticker (e.g. KXWOFREESKI-XTAER26MEDAL)")
    parser.add_argument("-o", "--output", default=None, help="Output file (default: <event_ticker>_event.json)")
    parser.add_argument("--no-markets", action="store_true", help="Do not include nested markets")
    args = parser.parse_args()
    ticker = args.event_ticker.strip().upper()
    out_path = args.output or f"{ticker}_event.json"

    data = fetch_event(ticker, with_nested_markets=not args.no_markets)
    if data is None:
        return 1
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
