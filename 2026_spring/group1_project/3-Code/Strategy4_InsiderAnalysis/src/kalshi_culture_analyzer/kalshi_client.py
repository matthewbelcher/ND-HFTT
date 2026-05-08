from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlencode

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from .config import BackoffConfig, Config


@dataclass
class KalshiAuth:
    key_id: str
    private_key_path: str

    def __post_init__(self) -> None:
        key_bytes = open(self.private_key_path, "rb").read()
        self._private_key = serialization.load_pem_private_key(key_bytes, password=None)

    def sign(self, timestamp_ms: int, method: str, path: str) -> str:
        message = f"{timestamp_ms}{method.upper()}{path}".encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")


class KalshiClient:
    def __init__(
        self,
        base_url: str,
        backoff: BackoffConfig,
        auth: Optional[KalshiAuth] = None,
        timeout: float = 20.0,
    ) -> None:
        self.base_url = normalize_base_url(base_url)
        self.backoff = backoff
        self.auth = auth
        self.timeout = timeout
        self.session = requests.Session()

    def request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.auth:
            ts = int(time.time() * 1000)
            signature = self.auth.sign(ts, method, path)
            headers.update(
                {
                    "KALSHI-ACCESS-KEY": self.auth.key_id,
                    "KALSHI-ACCESS-TIMESTAMP": str(ts),
                    "KALSHI-ACCESS-SIGNATURE": signature,
                }
            )

        retries = 0
        while True:
            try:
                resp = self.session.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                if retries >= self.backoff.max_retries:
                    raise
                sleep_seconds = min(self.backoff.base_seconds * (2**retries), self.backoff.max_seconds)
                time.sleep(sleep_seconds)
                retries += 1
                continue

            if resp.status_code in (429, 500, 502, 503, 504):
                if retries >= self.backoff.max_retries:
                    resp.raise_for_status()
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    sleep_seconds = float(retry_after)
                else:
                    sleep_seconds = min(self.backoff.base_seconds * (2**retries), self.backoff.max_seconds)
                time.sleep(sleep_seconds)
                retries += 1
                continue

            resp.raise_for_status()
            if not resp.text:
                return {}
            return resp.json()

    def get_event(self, event_ticker: str, with_nested_markets: bool = False) -> Dict[str, Any]:
        params = {"with_nested_markets": str(with_nested_markets).lower()} if with_nested_markets else None
        return self.request("GET", f"/events/{event_ticker}", params=params)

    def get_events(self, cursor: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self.request("GET", "/events", params=params)

    def get_markets(
        self,
        event_ticker: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 200,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        return self.request("GET", "/markets", params=params)

    def get_market(self, ticker: str) -> Dict[str, Any]:
        return self.request("GET", f"/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int = 1) -> Dict[str, Any]:
        return self.request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})

    def get_trades(
        self,
        ticker: str,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        cursor: Optional[str] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"ticker": ticker, "limit": limit}
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts
        if cursor:
            params["cursor"] = cursor
        return self.request("GET", "/markets/trades", params=params)

    def get_historical_cutoff(self) -> Dict[str, Any]:
        return self.request("GET", "/historical/cutoff")

    def get_historical_markets(
        self,
        event_ticker: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if cursor:
            params["cursor"] = cursor
        return self.request("GET", "/historical/markets", params=params)

    def get_historical_market(self, ticker: str) -> Dict[str, Any]:
        return self.request("GET", f"/historical/markets/{ticker}")

    def get_historical_fills(
        self,
        ticker: Optional[str] = None,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        cursor: Optional[str] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts
        if cursor:
            params["cursor"] = cursor
        return self.request("GET", "/historical/fills", params=params)


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/trade-api/v2"):
        return base_url
    return base_url + "/trade-api/v2"


def build_client_from_env(backoff: BackoffConfig) -> KalshiClient:
    data_base = os.getenv("KALSHI_DATA_API_BASE") or os.getenv("KALSHI_BASE_URL") or "https://demo-api.kalshi.co"
    key_id = os.getenv("KALSHI_KEY_ID")
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PEM")
    auth = None
    if key_id and key_path:
        auth = KalshiAuth(key_id=key_id, private_key_path=key_path)
    return KalshiClient(base_url=data_base, backoff=backoff, auth=auth)


def build_client_from_config(config: Config) -> KalshiClient:
    base = config.api.data_api_base or "https://demo-api.kalshi.co"
    key_id = config.auth.key_id or config.auth.trade_key_id
    key_path = config.auth.private_key_pem or config.auth.trade_private_key_pem
    auth = KalshiAuth(key_id=key_id, private_key_path=key_path) if key_id and key_path else None
    return KalshiClient(base_url=base, backoff=config.backoff, auth=auth)

