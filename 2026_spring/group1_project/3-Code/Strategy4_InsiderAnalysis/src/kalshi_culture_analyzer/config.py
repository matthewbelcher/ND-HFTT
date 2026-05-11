from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import os
import yaml

from .utils import parse_duration_to_seconds


@dataclass
class PollingConfig:
    poll_minutes: float = 10.0
    adaptive: bool = False
    adaptive_min_minutes: float = 2.0
    adaptive_max_minutes: float = 15.0
    adaptive_trade_burst_count: int = 5
    adaptive_window_minutes: int = 30


@dataclass
class AggressiveConfig:
    price_tolerance: float = 0.005
    burst_window_minutes: int = 60
    burst_trade_count: int = 3
    burst_contracts: int = 200


@dataclass
class LargeTradeConfig:
    absolute_contracts: int = 500
    pct_trailing_volume: float = 0.20
    pct_top_book: float = 0.30
    trailing_volume_window: str = "7d"

    def trailing_volume_seconds(self) -> int:
        return parse_duration_to_seconds(self.trailing_volume_window)


@dataclass
class StepChangeConfig:
    step_change_dollars: float = 0.10
    step_window_hours: int = 24
    step_aggressive_contracts: int = 200


@dataclass
class FavoriteConfig:
    favorite_high: float = 0.65
    favorite_min_days: int = 7


@dataclass
class ScoringConfig:
    score_threshold_flag: float = 70.0
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "aggressive_burst": 30.0,
            "large_aggressive": 25.0,
            "step_change": 25.0,
            "sustained_favorite": 20.0,
        }
    )


@dataclass
class BackoffConfig:
    max_retries: int = 5
    base_seconds: float = 1.0
    max_seconds: float = 30.0


@dataclass
class ApiConfig:
    data_api_base: Optional[str] = None
    trade_api_base: Optional[str] = None


@dataclass
class AuthConfig:
    key_id: Optional[str] = None
    private_key_pem: Optional[str] = None
    trade_key_id: Optional[str] = None
    trade_private_key_pem: Optional[str] = None


@dataclass
class Config:
    polling: PollingConfig = field(default_factory=PollingConfig)
    aggressive: AggressiveConfig = field(default_factory=AggressiveConfig)
    large_trade: LargeTradeConfig = field(default_factory=LargeTradeConfig)
    step_change: StepChangeConfig = field(default_factory=StepChangeConfig)
    favorite: FavoriteConfig = field(default_factory=FavoriteConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    backoff: BackoffConfig = field(default_factory=BackoffConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)

    @staticmethod
    def from_yaml(path: Optional[str]) -> "Config":
        cfg = Config()
        if not path:
            cfg._load_from_env()
            return cfg

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

        def assign(dc, values: Dict[str, Any]):
            for k, v in values.items():
                if hasattr(dc, k):
                    setattr(dc, k, v)

        assign(cfg.polling, data.get("polling", {}))
        assign(cfg.aggressive, data.get("aggressive", {}))
        assign(cfg.large_trade, data.get("large_trade", {}))
        assign(cfg.step_change, data.get("step_change", {}))
        assign(cfg.favorite, data.get("favorite", {}))
        assign(cfg.scoring, data.get("scoring", {}))
        assign(cfg.backoff, data.get("backoff", {}))
        assign(cfg.api, data.get("api", {}))
        assign(cfg.auth, data.get("auth", {}))

        cfg._load_from_env()
        return cfg

    def _load_from_env(self) -> None:
        # API bases
        self.api.data_api_base = self.api.data_api_base or os.getenv("KALSHI_DATA_API_BASE")
        self.api.trade_api_base = self.api.trade_api_base or os.getenv("KALSHI_TRADE_API_BASE")
        # Shared base override
        shared = os.getenv("KALSHI_BASE_URL")
        if shared and not self.api.data_api_base:
            self.api.data_api_base = shared
        if shared and not self.api.trade_api_base:
            self.api.trade_api_base = shared

        # Auth
        self.auth.key_id = self.auth.key_id or os.getenv("KALSHI_KEY_ID")
        self.auth.private_key_pem = self.auth.private_key_pem or os.getenv("KALSHI_PRIVATE_KEY_PEM")
        self.auth.trade_key_id = self.auth.trade_key_id or os.getenv("KALSHI_TRADE_KEY_ID")
        self.auth.trade_private_key_pem = (
            self.auth.trade_private_key_pem or os.getenv("KALSHI_TRADE_PRIVATE_KEY_PEM")
        )


