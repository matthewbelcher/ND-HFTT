import asyncio
import random
import uuid

import asyncpg

from models import SettlementCandidate


# Must agree with client/client-listener/db.py ACCOUNT_CASH_BUCKETS.
ACCOUNT_CASH_BUCKETS = 16


def _bucket_for_uuid(value: str) -> int:
    try:
        return int(uuid.UUID(value).int) % ACCOUNT_CASH_BUCKETS
    except Exception:
        return abs(hash(value)) % ACCOUNT_CASH_BUCKETS


class ExecutionError(Exception):
    pass


class RetryableOCCError(Exception):
    pass


def _is_occ_error(exc: Exception) -> bool:
    return isinstance(exc, asyncpg.PostgresError) and exc.sqlstate in {"40001", "40P01"}


async def create_pool(dsn: str, min_size: int, max_size: int) -> asyncpg.Pool:
    async def _dsql_safe_reset(_conn: asyncpg.Connection) -> None:
        # Aurora DSQL does not support pg_advisory_unlock_all().
        return None

    return await asyncpg.create_pool(
        dsn=dsn,
        min_size=min_size,
        max_size=max_size,
        reset=_dsql_safe_reset,
    )


class ExecutionRepository:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def ensure_schema(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS market_settlement_runs (
                    market_id UUID PRIMARY KEY,
                    status TEXT NOT NULL CHECK (status IN ('IN_PROGRESS', 'COMPLETED', 'FAILED')),
                    owner_id TEXT NOT NULL,
                    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    completed_at TIMESTAMPTZ,
                    settled_accounts BIGINT NOT NULL DEFAULT 0,
                    error_text TEXT,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX ASYNC IF NOT EXISTS market_settlement_runs_status_idx
                ON market_settlement_runs(status, updated_at)
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_market_leases (
                    market_id UUID PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    lease_expires_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX ASYNC IF NOT EXISTS execution_market_leases_exp_idx
                ON execution_market_leases(lease_expires_at)
                """
            )

    async def list_settlement_candidates(self, limit: int) -> list[SettlementCandidate]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT m.market_id, mr.outcome
                FROM markets m
                JOIN market_resolutions mr ON mr.market_id = m.market_id
                LEFT JOIN market_settlement_runs msr ON msr.market_id = m.market_id
                LEFT JOIN execution_market_leases l ON l.market_id = m.market_id
                WHERE m.status = 'RESOLVED'
                  AND (m.resolve_time IS NULL OR m.resolve_time <= now())
                  AND (msr.market_id IS NULL OR msr.status != 'COMPLETED')
                ORDER BY
                  CASE
                    WHEN l.market_id IS NULL OR l.lease_expires_at < now() THEN 0
                    ELSE 1
                  END,
                  COALESCE(l.updated_at, TIMESTAMPTZ 'epoch') ASC,
                  COALESCE(m.resolve_time, mr.resolved_at) ASC,
                  mr.resolved_at ASC
                LIMIT $1
                """,
                limit,
            )
        return [
            SettlementCandidate(market_id=str(r["market_id"]), outcome=str(r["outcome"]))
            for r in rows
        ]

    async def try_acquire_market_lease(
        self,
        market_id: str,
        owner_id: str,
        lease_seconds: int,
    ) -> bool:
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                async with self.pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        INSERT INTO execution_market_leases (
                            market_id,
                            owner_id,
                            lease_expires_at,
                            updated_at
                        )
                        VALUES ($1, $2, now() + ($3::text || ' seconds')::interval, now())
                        ON CONFLICT (market_id)
                        DO UPDATE
                        SET owner_id = EXCLUDED.owner_id,
                            lease_expires_at = EXCLUDED.lease_expires_at,
                            updated_at = now()
                        WHERE execution_market_leases.lease_expires_at < now()
                           OR execution_market_leases.owner_id = EXCLUDED.owner_id
                        RETURNING owner_id
                        """,
                        market_id,
                        owner_id,
                        f"{lease_seconds} seconds",
                    )
                    return bool(row and row["owner_id"] == owner_id)
            except asyncpg.PostgresError as exc:
                if not _is_occ_error(exc):
                    raise
                if attempt == max_attempts:
                    return False
                delay = (0.02 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.02)
                await asyncio.sleep(delay)
        return False

    async def execute_settlement(self, market_id: str, owner_id: str) -> int:
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                return await self._execute_settlement_once(market_id, owner_id)
            except RetryableOCCError:
                if attempt == max_attempts:
                    raise ExecutionError("database conflict after retries")
                delay = (0.02 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.02)
                await asyncio.sleep(delay)

    async def _execute_settlement_once(self, market_id: str, owner_id: str) -> int:
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    row = await conn.fetchrow(
                        """
                        SELECT m.market_id, m.status, m.resolve_time, mr.outcome
                        FROM markets m
                        JOIN market_resolutions mr ON mr.market_id = m.market_id
                        WHERE m.market_id = $1
                        """,
                        market_id,
                    )
                    if not row:
                        return 0
                    if row["status"] != "RESOLVED":
                        return 0
                    if row["resolve_time"] is not None:
                        now_ts = await conn.fetchval("SELECT now()")
                        if row["resolve_time"] > now_ts:
                            return 0

                    run_row = await conn.fetchrow(
                        """
                        INSERT INTO market_settlement_runs (
                            market_id, status, owner_id, started_at, updated_at
                        )
                        VALUES ($1, 'IN_PROGRESS', $2, now(), now())
                        ON CONFLICT (market_id)
                        DO UPDATE
                        SET status = 'IN_PROGRESS',
                            owner_id = EXCLUDED.owner_id,
                            updated_at = now(),
                            error_text = NULL
                        WHERE market_settlement_runs.status != 'COMPLETED'
                        RETURNING market_id
                        """,
                        market_id,
                        owner_id,
                    )
                    if not run_row:
                        return 0

                    positions = await conn.fetch(
                        """
                        SELECT account_id, yes_shares, no_shares, locked_yes_shares, locked_no_shares
                        FROM positions
                        WHERE market_id = $1
                          AND (yes_shares > 0 OR no_shares > 0 OR locked_yes_shares > 0 OR locked_no_shares > 0)
                        """,
                        market_id,
                    )

                    settled_accounts = 0
                    outcome = str(row["outcome"])

                    for p in positions:
                        account_id = str(p["account_id"])
                        yes_total = int(p["yes_shares"]) + int(p["locked_yes_shares"])
                        no_total = int(p["no_shares"]) + int(p["locked_no_shares"])

                        if outcome == "YES":
                            payout = yes_total * 100
                        elif outcome == "NO":
                            payout = no_total * 100
                        else:
                            # Project-level approximation for cancelled markets.
                            payout = (yes_total + no_total) * 50

                        inserted = await conn.fetchrow(
                            """
                            INSERT INTO market_settlements (
                                settlement_id, market_id, account_id, winning_side,
                                payout_cents, yes_shares_settled, no_shares_settled
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            ON CONFLICT (market_id, account_id) DO NOTHING
                            RETURNING settlement_id
                            """,
                            str(uuid.uuid4()),
                            market_id,
                            account_id,
                            outcome,
                            payout,
                            yes_total,
                            no_total,
                        )
                        if not inserted:
                            continue

                        # Bucket keyed by market_id so concurrent settlements
                        # for different markets target different rows.
                        payout_bucket = _bucket_for_uuid(market_id)
                        await conn.execute(
                            """
                            INSERT INTO account_cash_buckets (
                                account_id, bucket_id, available_cash_cents, locked_cash_cents
                            )
                            VALUES ($1, $2, $3, 0)
                            ON CONFLICT (account_id, bucket_id) DO UPDATE
                            SET available_cash_cents = account_cash_buckets.available_cash_cents + EXCLUDED.available_cash_cents,
                                updated_at = now()
                            """,
                            account_id,
                            payout_bucket,
                            payout,
                        )

                        await conn.execute(
                            """
                            UPDATE positions
                            SET yes_shares = 0,
                                no_shares = 0,
                                locked_yes_shares = 0,
                                locked_no_shares = 0,
                                updated_at = now()
                            WHERE account_id = $1
                              AND market_id = $2
                            """,
                            account_id,
                            market_id,
                        )

                        await conn.execute(
                            """
                            INSERT INTO ledger_entries (
                                entry_id, account_id, market_id,
                                cash_delta_cents, locked_cash_delta_cents,
                                yes_share_delta, no_share_delta,
                                locked_yes_delta, locked_no_delta,
                                reason, notes
                            )
                            VALUES ($1, $2, $3, $4, 0, $5, $6, $7, $8, 'MARKET_SETTLEMENT', $9)
                            """,
                            str(uuid.uuid4()),
                            account_id,
                            market_id,
                            payout,
                            -yes_total,
                            -no_total,
                            -int(p["locked_yes_shares"]),
                            -int(p["locked_no_shares"]),
                            f"Settlement for market {market_id} outcome={outcome}",
                        )

                        settled_accounts += 1

                    await conn.execute(
                        """
                        UPDATE market_settlement_runs
                        SET status = 'COMPLETED',
                            completed_at = now(),
                            settled_accounts = $2,
                            updated_at = now(),
                            error_text = NULL
                        WHERE market_id = $1
                        """,
                        market_id,
                        settled_accounts,
                    )

                    return settled_accounts
        except asyncpg.PostgresError as exc:
            if _is_occ_error(exc):
                raise RetryableOCCError from exc
            raise ExecutionError(f"database error: {exc}") from exc
