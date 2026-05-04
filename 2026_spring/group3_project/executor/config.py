import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    db_dsn: str
    db_min_pool_size: int
    db_max_pool_size: int
    poll_interval_ms: int
    market_scan_limit: int
    lease_seconds: int
    instance_id: str


def load_settings() -> Settings:
    return Settings(
        db_dsn=os.getenv(
            "EXECUTOR_DB_DSN",
            "postgresql://postgres:postgres@127.0.0.1:5432/postgres?sslmode=require",
        ),
        db_min_pool_size=int(os.getenv("EXECUTOR_DB_MIN_POOL", "1")),
        db_max_pool_size=int(os.getenv("EXECUTOR_DB_MAX_POOL", "10")),
        poll_interval_ms=int(os.getenv("EXECUTOR_POLL_INTERVAL_MS", "1000")),
        market_scan_limit=int(os.getenv("EXECUTOR_MARKET_SCAN_LIMIT", "50")),
        lease_seconds=int(os.getenv("EXECUTOR_LEASE_SECONDS", "10")),
        instance_id=os.getenv("EXECUTOR_INSTANCE_ID", "executor-local"),
    )
