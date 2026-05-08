from __future__ import annotations

import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .utils import safe_json_dumps


class Storage:
    def __init__(self, path: str) -> None:
        self.path = path
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_ticker TEXT PRIMARY KEY,
                title TEXT,
                category TEXT,
                series_ticker TEXT,
                status TEXT,
                tags_json TEXT,
                raw_json TEXT
            );

            CREATE TABLE IF NOT EXISTS markets (
                market_ticker TEXT PRIMARY KEY,
                event_ticker TEXT,
                title TEXT,
                status TEXT,
                close_time TEXT,
                result TEXT,
                settlement_value REAL,
                raw_json TEXT,
                FOREIGN KEY(event_ticker) REFERENCES events(event_ticker)
            );

            CREATE TABLE IF NOT EXISTS snapshots (
                market_ticker TEXT,
                ts INTEGER,
                snapshot_type TEXT,
                yes_bid REAL,
                yes_ask REAL,
                no_bid REAL,
                no_ask REAL,
                mid REAL,
                spread REAL,
                volume REAL,
                open_interest REAL,
                yes_bid_size REAL,
                yes_ask_size REAL,
                raw_json TEXT,
                PRIMARY KEY (market_ticker, ts, snapshot_type)
            );

            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                market_ticker TEXT,
                ts INTEGER,
                price REAL,
                count REAL,
                taker_side TEXT,
                is_inferred INTEGER,
                raw_json TEXT
            );

            CREATE TABLE IF NOT EXISTS features (
                market_ticker TEXT,
                ts INTEGER,
                feature_set TEXT,
                features_json TEXT,
                PRIMARY KEY (market_ticker, ts, feature_set)
            );

            CREATE TABLE IF NOT EXISTS scores (
                market_ticker TEXT,
                ts INTEGER,
                config_hash TEXT,
                score REAL,
                rules_json TEXT,
                explanation TEXT,
                PRIMARY KEY (market_ticker, ts, config_hash)
            );

            CREATE TABLE IF NOT EXISTS outcomes (
                market_ticker TEXT PRIMARY KEY,
                result TEXT,
                settlement_value REAL,
                resolved_ts INTEGER
            );

            CREATE TABLE IF NOT EXISTS collector_state (
                market_ticker TEXT PRIMARY KEY,
                last_trade_ts INTEGER,
                last_snapshot_ts INTEGER,
                last_cursor TEXT
            );
            """
        )
        self.conn.commit()

    def upsert_event(self, event_ticker: str, data: Dict[str, Any], tags: Optional[List[str]] = None) -> None:
        self.conn.execute(
            """
            INSERT INTO events (event_ticker, title, category, series_ticker, status, tags_json, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_ticker) DO UPDATE SET
                title=excluded.title,
                category=excluded.category,
                series_ticker=excluded.series_ticker,
                status=excluded.status,
                tags_json=excluded.tags_json,
                raw_json=excluded.raw_json
            """,
            (
                event_ticker,
                data.get("title"),
                data.get("category"),
                data.get("series_ticker"),
                data.get("status"),
                safe_json_dumps(tags or []),
                safe_json_dumps(data),
            ),
        )
        self.conn.commit()

    def upsert_market(self, market_ticker: str, event_ticker: str, data: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO markets (market_ticker, event_ticker, title, status, close_time, result, settlement_value, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market_ticker) DO UPDATE SET
                event_ticker=excluded.event_ticker,
                title=excluded.title,
                status=excluded.status,
                close_time=excluded.close_time,
                result=excluded.result,
                settlement_value=excluded.settlement_value,
                raw_json=excluded.raw_json
            """,
            (
                market_ticker,
                event_ticker,
                data.get("title"),
                data.get("status"),
                data.get("close_time"),
                data.get("result"),
                data.get("settlement_value"),
                safe_json_dumps(data),
            ),
        )
        self.conn.commit()

    def insert_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO snapshots
            (market_ticker, ts, snapshot_type, yes_bid, yes_ask, no_bid, no_ask, mid, spread,
             volume, open_interest, yes_bid_size, yes_ask_size, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot["market_ticker"],
                snapshot["ts"],
                snapshot.get("snapshot_type", "bbo"),
                snapshot.get("yes_bid"),
                snapshot.get("yes_ask"),
                snapshot.get("no_bid"),
                snapshot.get("no_ask"),
                snapshot.get("mid"),
                snapshot.get("spread"),
                snapshot.get("volume"),
                snapshot.get("open_interest"),
                snapshot.get("yes_bid_size"),
                snapshot.get("yes_ask_size"),
                safe_json_dumps(snapshot.get("raw", {})),
            ),
        )
        self.conn.commit()

    def insert_trade(self, trade: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO trades
            (trade_id, market_ticker, ts, price, count, taker_side, is_inferred, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade["trade_id"],
                trade["market_ticker"],
                trade["ts"],
                trade.get("price"),
                trade.get("count"),
                trade.get("taker_side"),
                1 if trade.get("is_inferred") else 0,
                safe_json_dumps(trade.get("raw", {})),
            ),
        )
        self.conn.commit()

    def insert_features(self, market_ticker: str, ts: int, feature_set: str, features: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO features (market_ticker, ts, feature_set, features_json)
            VALUES (?, ?, ?, ?)
            """,
            (market_ticker, ts, feature_set, safe_json_dumps(features)),
        )
        self.conn.commit()

    def insert_score(
        self, market_ticker: str, ts: int, config_hash: str, score: float, rules: Dict[str, Any], explanation: str
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO scores (market_ticker, ts, config_hash, score, rules_json, explanation)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (market_ticker, ts, config_hash, score, safe_json_dumps(rules), explanation),
        )
        self.conn.commit()

    def upsert_outcome(self, market_ticker: str, result: str, settlement_value: Optional[float], resolved_ts: int) -> None:
        self.conn.execute(
            """
            INSERT INTO outcomes (market_ticker, result, settlement_value, resolved_ts)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(market_ticker) DO UPDATE SET
                result=excluded.result,
                settlement_value=excluded.settlement_value,
                resolved_ts=excluded.resolved_ts
            """,
            (market_ticker, result, settlement_value, resolved_ts),
        )
        self.conn.commit()

    def get_markets_for_event(self, event_ticker: str) -> List[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM markets WHERE event_ticker = ?", (event_ticker,))
        return cur.fetchall()

    def get_events(self) -> List[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM events")
        return cur.fetchall()

    def get_latest_snapshot(self, market_ticker: str) -> Optional[sqlite3.Row]:
        cur = self.conn.execute(
            """
            SELECT * FROM snapshots WHERE market_ticker = ? ORDER BY ts DESC LIMIT 1
            """,
            (market_ticker,),
        )
        return cur.fetchone()

    def get_snapshot_before(self, market_ticker: str, ts_ms: int) -> Optional[sqlite3.Row]:
        cur = self.conn.execute(
            """
            SELECT * FROM snapshots WHERE market_ticker = ? AND ts <= ? ORDER BY ts DESC LIMIT 1
            """,
            (market_ticker, ts_ms),
        )
        return cur.fetchone()

    def get_snapshots(self, market_ticker: str, since_ms: Optional[int] = None) -> List[sqlite3.Row]:
        if since_ms:
            cur = self.conn.execute(
                "SELECT * FROM snapshots WHERE market_ticker = ? AND ts >= ? ORDER BY ts ASC",
                (market_ticker, since_ms),
            )
        else:
            cur = self.conn.execute(
                "SELECT * FROM snapshots WHERE market_ticker = ? ORDER BY ts ASC",
                (market_ticker,),
            )
        return cur.fetchall()

    def get_trades(self, market_ticker: str, since_ms: Optional[int] = None) -> List[sqlite3.Row]:
        if since_ms:
            cur = self.conn.execute(
                "SELECT * FROM trades WHERE market_ticker = ? AND ts >= ? ORDER BY ts ASC",
                (market_ticker, since_ms),
            )
        else:
            cur = self.conn.execute(
                "SELECT * FROM trades WHERE market_ticker = ? ORDER BY ts ASC",
                (market_ticker,),
            )
        return cur.fetchall()

    def get_latest_scores(self, since_ms: Optional[int] = None) -> List[sqlite3.Row]:
        if since_ms:
            cur = self.conn.execute(
                """
                SELECT s1.* FROM scores s1
                JOIN (
                    SELECT market_ticker, MAX(ts) AS max_ts FROM scores WHERE ts >= ? GROUP BY market_ticker
                ) s2
                ON s1.market_ticker = s2.market_ticker AND s1.ts = s2.max_ts
                """,
                (since_ms,),
            )
        else:
            cur = self.conn.execute(
                """
                SELECT s1.* FROM scores s1
                JOIN (
                    SELECT market_ticker, MAX(ts) AS max_ts FROM scores GROUP BY market_ticker
                ) s2
                ON s1.market_ticker = s2.market_ticker AND s1.ts = s2.max_ts
                """
            )
        return cur.fetchall()

    def update_collector_state(
        self, market_ticker: str, last_trade_ts: Optional[int] = None, last_snapshot_ts: Optional[int] = None, last_cursor: Optional[str] = None
    ) -> None:
        cur = self.conn.execute("SELECT market_ticker FROM collector_state WHERE market_ticker = ?", (market_ticker,))
        exists = cur.fetchone() is not None
        if exists:
            self.conn.execute(
                """
                UPDATE collector_state
                SET last_trade_ts = COALESCE(?, last_trade_ts),
                    last_snapshot_ts = COALESCE(?, last_snapshot_ts),
                    last_cursor = COALESCE(?, last_cursor)
                WHERE market_ticker = ?
                """,
                (last_trade_ts, last_snapshot_ts, last_cursor, market_ticker),
            )
        else:
            self.conn.execute(
                """
                INSERT INTO collector_state (market_ticker, last_trade_ts, last_snapshot_ts, last_cursor)
                VALUES (?, ?, ?, ?)
                """,
                (market_ticker, last_trade_ts, last_snapshot_ts, last_cursor),
            )
        self.conn.commit()

    def get_collector_state(self, market_ticker: str) -> Optional[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM collector_state WHERE market_ticker = ?", (market_ticker,))
        return cur.fetchone()

    def list_markets(self) -> List[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM markets")
        return cur.fetchall()

    def list_open_markets(self) -> List[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM markets WHERE status NOT IN ('resolved', 'settled', 'final')")
        return cur.fetchall()

