from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Tuple

from .config import Config


def config_hash(config: Config) -> str:
    payload = {
        "aggressive": config.aggressive.__dict__,
        "large_trade": config.large_trade.__dict__,
        "step_change": config.step_change.__dict__,
        "favorite": config.favorite.__dict__,
        "scoring": config.scoring.__dict__,
    }
    data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:12]


def evaluate_rules(
    market_ticker: str,
    features: Dict[str, Any],
    event_features: Dict[str, Any],
    config: Config,
) -> Tuple[float, Dict[str, Any], str]:
    triggered: Dict[str, Any] = {}
    score = 0.0

    # Rule: Aggressive YES burst
    if (
        features.get("aggressive_burst_count", 0) >= config.aggressive.burst_trade_count
        and features.get("aggressive_burst_contracts", 0) >= config.aggressive.burst_contracts
    ):
        triggered["aggressive_burst"] = {
            "count": features.get("aggressive_burst_count"),
            "contracts": features.get("aggressive_burst_contracts"),
            "window_minutes": config.aggressive.burst_window_minutes,
        }
        score += config.scoring.weights.get("aggressive_burst", 0.0)

    # Rule: Large aggressive YES trade
    max_trade = features.get("max_aggressive_yes_trade", 0.0)
    trailing_volume = features.get("trailing_volume", 0.0) or 0.0
    yes_ask_size = features.get("yes_ask_size", 0.0) or 0.0
    cond_abs = max_trade >= config.large_trade.absolute_contracts
    cond_pct_volume = trailing_volume > 0 and (max_trade / trailing_volume) >= config.large_trade.pct_trailing_volume
    cond_pct_book = yes_ask_size > 0 and (max_trade / yes_ask_size) >= config.large_trade.pct_top_book
    if cond_abs or cond_pct_volume or cond_pct_book:
        triggered["large_aggressive"] = {
            "max_trade": max_trade,
            "trailing_volume": trailing_volume,
            "yes_ask_size": yes_ask_size,
            "abs_threshold": config.large_trade.absolute_contracts,
            "pct_trailing_volume": config.large_trade.pct_trailing_volume,
            "pct_top_book": config.large_trade.pct_top_book,
        }
        score += config.scoring.weights.get("large_aggressive", 0.0)

    # Rule: Step-change with aggressive flow
    if (
        features.get("mid_change_window", 0.0) >= config.step_change.step_change_dollars
        and features.get("aggressive_step_contracts", 0.0) >= config.step_change.step_aggressive_contracts
    ):
        triggered["step_change"] = {
            "mid_change": features.get("mid_change_window"),
            "step_window_hours": config.step_change.step_window_hours,
            "aggressive_contracts": features.get("aggressive_step_contracts"),
        }
        score += config.scoring.weights.get("step_change", 0.0)

    # Rule: Sustained favorite
    if (
        event_features.get("favorite_streak_days") is not None
        and event_features.get("favorite_streak_days") >= config.favorite.favorite_min_days
        and features.get("yes_mid", 0.0) >= config.favorite.favorite_high
    ):
        triggered["sustained_favorite"] = {
            "favorite_streak_days": event_features.get("favorite_streak_days"),
            "favorite_high": config.favorite.favorite_high,
        }
        score += config.scoring.weights.get("sustained_favorite", 0.0)

    score = min(score, 100.0)

    explanation_parts = []
    for key, detail in triggered.items():
        explanation_parts.append(f"{key}: {detail}")
    explanation = "; ".join(explanation_parts) if explanation_parts else "no rules triggered"
    return score, triggered, explanation

