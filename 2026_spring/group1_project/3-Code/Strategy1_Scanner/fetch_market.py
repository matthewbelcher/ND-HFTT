#!/usr/bin/env python3
"""Fetch a Kalshi market by ticker and write JSON to a file."""

import argparse
import json
import sys

try:
    import requests
except ImportError:
    requests = None

API_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def fetch_market(ticker: str) -> dict | None:
    url = f"{API_BASE}/markets/{ticker}"
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
    parser = argparse.ArgumentParser(description="Fetch Kalshi market data and write to file.")
    parser.add_argument("ticker", help="Market ticker (e.g. KXWOFREESKI-XTAER26MEDAL-CHN)")
    parser.add_argument("-o", "--output", default=None, help="Output file (default: <ticker>.json)")
    args = parser.parse_args()
    ticker = args.ticker.strip().upper()
    out_path = args.output or f"{ticker}.json"

    data = fetch_market(ticker)
    if data is None:
        return 1
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
