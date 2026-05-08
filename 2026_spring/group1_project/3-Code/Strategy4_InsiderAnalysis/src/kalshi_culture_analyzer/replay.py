from __future__ import annotations

from typing import Dict

from .config import Config
from .features import compute_event_favorites, compute_market_features
from .rules import config_hash, evaluate_rules
from .storage import Storage


def replay_scores(db_path: str, config: Config) -> None:
    storage = Storage(db_path)
    cfg_hash = config_hash(config)

    events = storage.get_events()
    for event in events:
        event_ticker = event["event_ticker"]
        markets = storage.get_markets_for_event(event_ticker)
        if not markets:
            continue

        ts_set = set()
        for market in markets:
            snaps = storage.get_snapshots(market["market_ticker"])
            ts_set.update([s["ts"] for s in snaps])

        ts_list = sorted(ts_set)
        streak_days: Dict[str, float] = {m["market_ticker"]: 0.0 for m in markets}
        last_fav_ts: Dict[str, int] = {m["market_ticker"]: 0 for m in markets}
        last_fav_state: Dict[str, bool] = {m["market_ticker"]: False for m in markets}

        for ts in ts_list:
            favorites = compute_event_favorites(storage, event_ticker, ts)
            for market in markets:
                ticker = market["market_ticker"]
                is_fav = favorites.get(ticker, {}).get("is_favorite", 0.0) == 1.0
                if is_fav:
                    if last_fav_state.get(ticker):
                        delta_days = (ts - last_fav_ts.get(ticker, ts)) / (1000 * 3600 * 24)
                        streak_days[ticker] += max(delta_days, 0.0)
                    else:
                        streak_days[ticker] = 0.0
                    last_fav_ts[ticker] = ts
                else:
                    streak_days[ticker] = 0.0
                    last_fav_ts[ticker] = ts
                last_fav_state[ticker] = is_fav

                event_features = {**favorites.get(ticker, {}), "favorite_streak_days": streak_days[ticker]}

                features = compute_market_features(storage, ticker, ts, config)
                if features:
                    storage.insert_features(ticker, ts, "market", features)
                    storage.insert_features(ticker, ts, "event", event_features)
                    score, rules, explanation = evaluate_rules(ticker, features, event_features, config)
                    storage.insert_score(ticker, ts, cfg_hash, score, rules, explanation)

