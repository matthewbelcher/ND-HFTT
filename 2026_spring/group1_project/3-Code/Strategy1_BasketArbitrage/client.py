from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

from kalshi_common import load_private_key_pem, sign_pss_base64, now_ms

TRADE_API_V2 = "/trade-api/v2"


def kalshi_rest_headers(key_id: str, private_key, method: str, path: str) -> Dict[str, str]:
    ts_ms = str(now_ms())
    path_to_sign = path.split("?", 1)[0]
    msg = ts_ms + method.upper() + path_to_sign
    sig = sign_pss_base64(private_key, msg)
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": sig,
    }


@dataclass
class _CacheItem:
    expires_at: float
    value: dict


class KalshiClient:
    def __init__(
        self,
        key_id: str,
        private_key,
        base_url: Optional[str] = None,
        cache_seconds: int = 0,
        retries: int = 2,
        backoff_s: float = 0.4,
    ) -> None:
        if not base_url:
            base_url = os.getenv(
                "KALSHI_TRADE_API_BASE",
                os.getenv("KALSHI_BASE_URL", "https://demo-api.kalshi.co"),
            )
        self.base_url = base_url
        self.key_id = key_id
        self.private_key = private_key
        self.cache_seconds = max(0, int(cache_seconds))
        self.retries = max(0, int(retries))
        self.backoff_s = max(0.0, float(backoff_s))
        self._cache: Dict[str, _CacheItem] = {}

    def _get(self, path: str) -> dict:
        cache_key = path
        if self.cache_seconds > 0:
            cached = self._cache.get(cache_key)
            if cached and cached.expires_at > time.time():
                return cached.value

        url = self.base_url + path
        headers = kalshi_rest_headers(self.key_id, self.private_key, "GET", path)
        req = urllib.request.Request(url, headers=headers, method="GET")
        last_err: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
                value = json.loads(data.decode("utf-8")) if data else {}
                if self.cache_seconds > 0:
                    self._cache[cache_key] = _CacheItem(
                        expires_at=time.time() + self.cache_seconds,
                        value=value,
                    )
                return value
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(self.backoff_s * (2 ** attempt))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionResetError) as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(self.backoff_s * (2 ** attempt))
                    continue
                raise

        if last_err:
            raise last_err
        return {}

    def _post(self, path: str, body: dict) -> dict:
        url = self.base_url + path
        headers = kalshi_rest_headers(self.key_id, self.private_key, "POST", path)
        headers["Content-Type"] = "application/json"
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, headers=headers, data=payload, method="POST")
        last_err: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
                return json.loads(data.decode("utf-8")) if data else {}
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(self.backoff_s * (2 ** attempt))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionResetError) as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(self.backoff_s * (2 ** attempt))
                    continue
                raise

        if last_err:
            raise last_err
        return {}

    def _get_paginated(self, path: str, list_key: str, limit: int = 200) -> List[dict]:
        out: List[dict] = []
        cursor = None
        while True:
            params = {"limit": str(limit)}
            if cursor:
                params["cursor"] = cursor
            qs = urllib.parse.urlencode(params)
            sep = "&" if "?" in path else "?"
            resp = self._get(f"{path}{sep}{qs}")
            items = resp.get(list_key) or []
            if isinstance(items, list):
                out.extend(items)
            cursor = resp.get("cursor")
            if not cursor:
                break
        return out

    def get_event_markets(self, event_ticker: str) -> Tuple[dict, List[dict]]:
        path = f"{TRADE_API_V2}/events/{urllib.parse.quote(event_ticker)}"
        resp = self._get(path)
        event = resp.get("event", {}) if isinstance(resp, dict) else {}
        markets = []
        if isinstance(resp, dict):
            markets = resp.get("markets") or event.get("markets") or []
        if not isinstance(markets, list):
            markets = []
        return event, markets

    def get_orderbook(self, market_ticker: str, depth: int = 1) -> dict:
        params = {"depth": str(max(1, int(depth)))}
        qs = urllib.parse.urlencode(params)
        path = f"{TRADE_API_V2}/markets/{urllib.parse.quote(market_ticker)}/orderbook?{qs}"
        resp = self._get(path)
        return resp if isinstance(resp, dict) else {}

    def place_orders_batch(self, orders: List[dict]) -> dict:
        if not orders:
            return {}
        path = f"{TRADE_API_V2}/portfolio/orders/batched"
        return self._post(path, {"orders": orders})

    def cancel_order(self, order_id: str) -> dict:
        path = f"{TRADE_API_V2}/portfolio/orders/{order_id}/cancel"
        return self._post(path, {})

    def get_positions(self, event_ticker: str, subaccount: Optional[int] = None) -> List[dict]:
        params = {"event_ticker": event_ticker, "count_filter": "position"}
        if subaccount is not None:
            params["subaccount"] = str(subaccount)
        qs = urllib.parse.urlencode(params)
        path = f"{TRADE_API_V2}/portfolio/positions?{qs}"
        return self._get_paginated(path, "market_positions", limit=200)

    def get_orders(
        self,
        event_ticker: str,
        status: str = "resting",
        subaccount: Optional[int] = None,
    ) -> List[dict]:
        params = {"event_ticker": event_ticker, "status": status}
        if subaccount is not None:
            params["subaccount"] = str(subaccount)
        qs = urllib.parse.urlencode(params)
        path = f"{TRADE_API_V2}/portfolio/orders?{qs}"
        return self._get_paginated(path, "orders", limit=200)

    def get_fills(
        self,
        ticker: str,
        min_ts: Optional[int] = None,
        subaccount: Optional[int] = None,
    ) -> List[dict]:
        params: Dict[str, str] = {"ticker": ticker}
        if min_ts is not None:
            params["min_ts"] = str(min_ts)
        if subaccount is not None:
            params["subaccount"] = str(subaccount)
        qs = urllib.parse.urlencode(params)
        path = f"{TRADE_API_V2}/portfolio/fills?{qs}"
        return self._get_paginated(path, "fills", limit=200)


def load_client_from_env(
    *,
    cache_seconds: int = 0,
    retries: int = 2,
    backoff_s: float = 0.4,
) -> KalshiClient:
    base_url = os.getenv(
        "KALSHI_DATA_API_BASE",
        os.getenv("KALSHI_BASE_URL", "https://demo-api.kalshi.co"),
    )
    key_id = os.getenv("KALSHI_KEY_ID") or os.getenv("KALSHI_TRADE_KEY_ID")
    pem_path = os.getenv("KALSHI_PRIVATE_KEY_PEM") or os.getenv("KALSHI_TRADE_PRIVATE_KEY_PEM")
    if not key_id or not pem_path:
        raise RuntimeError(
            "Missing credentials. Set KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PEM."
        )
    private_key = load_private_key_pem(pem_path)
    return KalshiClient(
        key_id=key_id,
        private_key=private_key,
        base_url=base_url,
        cache_seconds=cache_seconds,
        retries=retries,
        backoff_s=backoff_s,
    )


def load_trade_client_from_env(
    *,
    cache_seconds: int = 0,
    retries: int = 2,
    backoff_s: float = 0.4,
) -> KalshiClient:
    key_id = os.getenv("KALSHI_TRADE_KEY_ID") or os.getenv("KALSHI_KEY_ID")
    pem_path = os.getenv("KALSHI_TRADE_PRIVATE_KEY_PEM") or os.getenv("KALSHI_PRIVATE_KEY_PEM")
    if not key_id or not pem_path:
        raise RuntimeError(
            "Missing trade credentials. Set KALSHI_TRADE_KEY_ID and KALSHI_TRADE_PRIVATE_KEY_PEM."
        )
    base_url = os.getenv(
        "KALSHI_TRADE_API_BASE",
        os.getenv("KALSHI_BASE_URL", "https://demo-api.kalshi.co"),
    )
    private_key = load_private_key_pem(pem_path)
    return KalshiClient(
        key_id=key_id,
        private_key=private_key,
        base_url=base_url,
        cache_seconds=cache_seconds,
        retries=retries,
        backoff_s=backoff_s,
    )
