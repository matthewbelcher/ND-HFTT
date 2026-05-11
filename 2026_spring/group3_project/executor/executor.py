import asyncio
from datetime import datetime, timezone

from config import load_settings
from db import ExecutionError, ExecutionRepository, create_pool


async def main() -> None:
    settings = load_settings()
    pool = await create_pool(
        dsn=settings.db_dsn,
        min_size=settings.db_min_pool_size,
        max_size=settings.db_max_pool_size,
    )
    repo = ExecutionRepository(pool)
    await repo.ensure_schema()

    print(
        f"[{datetime.now(timezone.utc).isoformat()}] executor started "
        f"instance_id={settings.instance_id}"
    )

    try:
        while True:
            candidates = await repo.list_settlement_candidates(settings.market_scan_limit)
            if not candidates:
                await asyncio.sleep(settings.poll_interval_ms / 1000)
                continue

            settled_total = 0
            for c in candidates:
                acquired = await repo.try_acquire_market_lease(
                    market_id=c.market_id,
                    owner_id=settings.instance_id,
                    lease_seconds=settings.lease_seconds,
                )
                if not acquired:
                    continue

                try:
                    settled = await repo.execute_settlement(c.market_id, settings.instance_id)
                except ExecutionError as exc:
                    print(
                        f"[{datetime.now(timezone.utc).isoformat()}] "
                        f"settlement error market_id={c.market_id}: {exc}"
                    )
                    continue

                if settled > 0:
                    settled_total += settled
                    print(
                        f"[{datetime.now(timezone.utc).isoformat()}] "
                        f"market_settled market_id={c.market_id} accounts={settled}"
                    )

            if settled_total > 0:
                print(
                    f"[{datetime.now(timezone.utc).isoformat()}] "
                    f"settled_accounts_total={settled_total} markets_scanned={len(candidates)}"
                )

            await asyncio.sleep(settings.poll_interval_ms / 1000)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
