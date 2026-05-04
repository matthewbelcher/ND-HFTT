import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    db_dsn: str
    db_min_pool_size: int
    db_max_pool_size: int
    poll_interval_ms: int
    market_scan_limit: int
    order_scan_limit: int
    lease_seconds: int
    instance_id: str
    instance_index: int
    total_instances: int
    requeue_interval_ms: int
    metrics_host: str
    metrics_port: int


def load_settings() -> Settings:
    instance_index = max(0, int(os.getenv("MATCHMAKER_INSTANCE_INDEX", "0")))
    total_instances = max(1, int(os.getenv("MATCHMAKER_TOTAL_INSTANCES", "1")))
    if instance_index >= total_instances:
        instance_index = 0

    return Settings(
        db_dsn=os.getenv(
            "MATCHMAKER_DB_DSN",
            "postgresql://postgres:postgres@127.0.0.1:5432/postgres?sslmode=require",
        ),
        db_min_pool_size=int(os.getenv("MATCHMAKER_DB_MIN_POOL", "1")),
        db_max_pool_size=int(os.getenv("MATCHMAKER_DB_MAX_POOL", "10")),
        poll_interval_ms=int(os.getenv("MATCHMAKER_POLL_INTERVAL_MS", "500")),
        market_scan_limit=int(os.getenv("MATCHMAKER_MARKET_SCAN_LIMIT", "50")),
        order_scan_limit=int(os.getenv("MATCHMAKER_ORDER_SCAN_LIMIT", "2000")),
        lease_seconds=int(os.getenv("MATCHMAKER_LEASE_SECONDS", "5")),
        instance_id=os.getenv("MATCHMAKER_INSTANCE_ID", "matchmaker-local"),
        instance_index=instance_index,
        total_instances=total_instances,
        requeue_interval_ms=int(os.getenv("MATCHMAKER_REQUEUE_INTERVAL_MS", "5000")),
        metrics_host=os.getenv("MATCHMAKER_METRICS_HOST", "0.0.0.0"),
        metrics_port=int(os.getenv("MATCHMAKER_METRICS_PORT", "0")),
    )
