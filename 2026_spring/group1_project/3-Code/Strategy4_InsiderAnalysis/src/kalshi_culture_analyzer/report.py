from __future__ import annotations

from typing import Optional

from .config import Config
from .features import compute_event_favorites
from .storage import Storage
from .utils import parse_since_to_ms


def _result_is_yes(value: Optional[str]) -> bool:
    if not value:
        return False
    return str(value).strip().lower() in {"yes", "y", "true", "1"}


def generate_report(db_path: str, event_ticker: Optional[str], top_k: int, since: Optional[str], config: Config) -> None:
    storage = Storage(db_path)
    since_ms = parse_since_to_ms(since) if since else None
    scores = storage.get_latest_scores(since_ms)

    if event_ticker:
        market_rows = storage.get_markets_for_event(event_ticker)
        market_set = {m["market_ticker"] for m in market_rows}
        scores = [s for s in scores if s["market_ticker"] in market_set]

    scores_sorted = sorted(scores, key=lambda r: r["score"], reverse=True)

    print("Top suspicious markets:")
    for row in scores_sorted[:top_k]:
        print(f"- {row['market_ticker']} score={row['score']:.1f} ts={row['ts']}")

    if not event_ticker:
        return

    # Event-level evaluation
    markets = storage.get_markets_for_event(event_ticker)
    if not markets:
        print("No markets found for event.")
        return

    winner_ticker = None
    for market in markets:
        result = market["result"]
        if _result_is_yes(result):
            winner_ticker = market["market_ticker"]
            break

    if not winner_ticker:
        print("Winner not resolved yet (no YES result in markets table).")
        return

    # Favorite checks at open/mid/pre-close
    snapshots = {m["market_ticker"]: storage.get_snapshots(m["market_ticker"]) for m in markets}
    all_ts = sorted({snap["ts"] for snaps in snapshots.values() for snap in snaps})
    if not all_ts:
        print("No snapshots available for event.")
        return

    open_ts = all_ts[0]
    mid_ts = all_ts[len(all_ts) // 2]
    pre_close_ts = all_ts[-1]

    favorites_open = compute_event_favorites(storage, event_ticker, open_ts)
    favorites_mid = compute_event_favorites(storage, event_ticker, mid_ts)
    favorites_pre = compute_event_favorites(storage, event_ticker, pre_close_ts)

    print("Winner/favorite checks:")
    print(f"- winner: {winner_ticker}")
    print(f"- favorite at open: {favorites_open.get(winner_ticker, {}).get('is_favorite', 0) == 1.0}")
    print(f"- favorite at mid: {favorites_mid.get(winner_ticker, {}).get('is_favorite', 0) == 1.0}")
    print(f"- favorite pre-close: {favorites_pre.get(winner_ticker, {}).get('is_favorite', 0) == 1.0}")

    # Precision/recall of flagged markets
    latest_scores = {row["market_ticker"]: row["score"] for row in scores_sorted}
    flagged = {m for m, sc in latest_scores.items() if sc >= config.scoring.score_threshold_flag}

    precision = 0.0
    recall = 0.0
    if flagged:
        precision = 1.0 / len(flagged) if winner_ticker in flagged else 0.0
    recall = 1.0 if winner_ticker in flagged else 0.0

    print("Flagged market accuracy:")
    print(f"- flagged count: {len(flagged)}")
    print(f"- precision: {precision:.2f}")
    print(f"- recall: {recall:.2f}")

    # Winner rank
    ranked = sorted(latest_scores.items(), key=lambda x: x[1], reverse=True)
    rank = next((i + 1 for i, (t, _) in enumerate(ranked) if t == winner_ticker), None)
    if rank is not None:
        print(f"- winner score rank: {rank} / {len(ranked)}")

