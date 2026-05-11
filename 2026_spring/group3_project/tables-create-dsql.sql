-- Aurora DSQL does not support foreign keys; relationships are enforced in app logic.
-- Aurora DSQL requires CREATE INDEX ASYNC; all indexes here use ASYNC.
-- =========================================================
-- Prediction Market Exchange Schema
-- v1 starter schema
-- =========================================================

-- =========================================================
-- 1. Accounts
-- =========================================================

CREATE TABLE IF NOT EXISTS accounts (
    account_id UUID PRIMARY KEY,
    external_user_id TEXT UNIQUE,
    username TEXT UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'SUSPENDED', 'CLOSED')),

    available_cash_cents BIGINT NOT NULL DEFAULT 0 CHECK (available_cash_cents >= 0),
    locked_cash_cents BIGINT NOT NULL DEFAULT 0 CHECK (locked_cash_cents >= 0),

    password_hash TEXT,
    password_salt TEXT,
    password_iterations INTEGER CHECK (password_iterations IS NULL OR password_iterations > 0),
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE accounts ADD COLUMN IF NOT EXISTS password_hash TEXT;
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS password_salt TEXT;
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS password_iterations INTEGER;
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS is_admin BOOLEAN;

CREATE INDEX ASYNC IF NOT EXISTS accounts_status_idx ON accounts(status);
CREATE INDEX ASYNC IF NOT EXISTS accounts_username_idx ON accounts(username);


-- =========================================================
-- 1a. Account auth sessions
-- =========================================================

CREATE TABLE IF NOT EXISTS account_auth_sessions (
    session_id UUID PRIMARY KEY,
    account_id UUID NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ASYNC IF NOT EXISTS account_auth_sessions_account_idx ON account_auth_sessions(account_id);
CREATE INDEX ASYNC IF NOT EXISTS account_auth_sessions_expires_idx ON account_auth_sessions(expires_at);


-- =========================================================
-- 2. Markets
-- =========================================================

CREATE TABLE IF NOT EXISTS markets (
    market_id UUID PRIMARY KEY,
    slug TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    description TEXT,

    status TEXT NOT NULL CHECK (
        status IN ('DRAFT', 'ACTIVE', 'HALTED', 'CLOSED', 'RESOLVED', 'CANCELLED')
    ),

    tick_size_cents BIGINT NOT NULL DEFAULT 1 CHECK (tick_size_cents > 0),
    min_price_cents BIGINT NOT NULL DEFAULT 1 CHECK (min_price_cents >= 0),
    max_price_cents BIGINT NOT NULL DEFAULT 99 CHECK (max_price_cents > min_price_cents),

    close_time TIMESTAMPTZ,
    resolve_time TIMESTAMPTZ,

    created_by UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CHECK (min_price_cents <= max_price_cents),
    CHECK (max_price_cents <= 100)
);

CREATE INDEX ASYNC IF NOT EXISTS markets_status_idx ON markets(status);
CREATE INDEX ASYNC IF NOT EXISTS markets_close_time_idx ON markets(close_time);
CREATE INDEX ASYNC IF NOT EXISTS markets_resolve_time_idx ON markets(resolve_time);


-- =========================================================
-- 3. Market resolutions
-- One row per resolved/cancelled market outcome
-- =========================================================

CREATE TABLE IF NOT EXISTS market_resolutions (
    resolution_id UUID PRIMARY KEY,
    market_id UUID NOT NULL UNIQUE,

    outcome TEXT NOT NULL CHECK (outcome IN ('YES', 'NO', 'CANCELLED')),
    resolved_by UUID,
    notes TEXT,

    resolved_at TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- =========================================================
-- 4. Positions
-- Snapshot table for fast reads
-- =========================================================

CREATE TABLE IF NOT EXISTS positions (
    account_id UUID NOT NULL,
    market_id UUID NOT NULL,

    yes_shares BIGINT NOT NULL DEFAULT 0 CHECK (yes_shares >= 0),
    no_shares BIGINT NOT NULL DEFAULT 0 CHECK (no_shares >= 0),

    locked_yes_shares BIGINT NOT NULL DEFAULT 0 CHECK (locked_yes_shares >= 0),
    locked_no_shares BIGINT NOT NULL DEFAULT 0 CHECK (locked_no_shares >= 0),

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (account_id, market_id)
);

CREATE INDEX ASYNC IF NOT EXISTS positions_market_idx ON positions(market_id);


-- =========================================================
-- 4a. Account cash buckets
-- Sharded cash holders to break the per-account OCC hotspot.
-- The bucket count must agree across listener / matcher / executor.
-- =========================================================

CREATE TABLE IF NOT EXISTS account_cash_buckets (
    account_id UUID NOT NULL,
    bucket_id INT NOT NULL CHECK (bucket_id >= 0),

    available_cash_cents BIGINT NOT NULL DEFAULT 0 CHECK (available_cash_cents >= 0),
    locked_cash_cents BIGINT NOT NULL DEFAULT 0 CHECK (locked_cash_cents >= 0),

    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (account_id, bucket_id)
);

CREATE INDEX ASYNC IF NOT EXISTS account_cash_buckets_account_idx
    ON account_cash_buckets(account_id);


-- =========================================================
-- 5. Orders
-- Limit orders only for v1
-- YES / NO side for prediction markets
-- =========================================================

CREATE TABLE IF NOT EXISTS orders (
    request_id UUID PRIMARY KEY,

    global_seq BIGINT GENERATED ALWAYS AS IDENTITY (
        START WITH 1
        INCREMENT BY 1
        CACHE 1
    ),

    account_id UUID NOT NULL,
    market_id UUID NOT NULL,

    side TEXT NOT NULL CHECK (side IN ('YES', 'NO')),
    order_type TEXT NOT NULL CHECK (order_type IN ('LIMIT')),
    time_in_force TEXT NOT NULL CHECK (time_in_force IN ('GTC', 'IOC', 'FOK')),

    qty BIGINT NOT NULL CHECK (qty > 0),
    remaining_qty BIGINT NOT NULL CHECK (remaining_qty >= 0 AND remaining_qty <= qty),

    -- YES price in cents, e.g. 57 means $0.57 YES
    price_cents BIGINT NOT NULL CHECK (price_cents >= 0 AND price_cents <= 100),

    ingress_ts_ns BIGINT NOT NULL,

    status TEXT NOT NULL CHECK (
        status IN ('ACCEPTED', 'OPEN', 'PARTIALLY_FILLED', 'FILLED', 'CANCELLED', 'REJECTED')
    ),

    reject_reason TEXT,

    -- Which account_cash_buckets row holds the locked cash for this order.
    -- NULL on legacy rows that pre-date the bucketed cash design.
    lock_bucket_id INT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE orders ADD COLUMN IF NOT EXISTS lock_bucket_id INT;

CREATE UNIQUE INDEX ASYNC IF NOT EXISTS orders_global_seq_uq ON orders(global_seq);
CREATE INDEX ASYNC IF NOT EXISTS orders_market_seq_idx ON orders(market_id, global_seq);
CREATE INDEX ASYNC IF NOT EXISTS orders_account_seq_idx ON orders(account_id, global_seq);
CREATE INDEX ASYNC IF NOT EXISTS orders_market_status_price_idx ON orders(market_id, status, price_cents);
CREATE INDEX ASYNC IF NOT EXISTS orders_market_side_status_price_idx ON orders(market_id, side, status, price_cents);


-- =========================================================
-- 6. Order cancels
-- Separate audit/event table for cancellation requests
-- =========================================================

CREATE TABLE IF NOT EXISTS order_cancels (
    cancel_id UUID PRIMARY KEY,
    order_id UUID NOT NULL,
    account_id UUID NOT NULL,

    cancel_seq BIGINT GENERATED ALWAYS AS IDENTITY (
        START WITH 1
        INCREMENT BY 1
        CACHE 1
    ),

    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX ASYNC IF NOT EXISTS order_cancels_seq_uq ON order_cancels(cancel_seq);
CREATE INDEX ASYNC IF NOT EXISTS order_cancels_order_idx ON order_cancels(order_id);


-- =========================================================
-- 7. Trades
-- One row per execution
-- =========================================================

CREATE TABLE IF NOT EXISTS trades (
    trade_id UUID PRIMARY KEY,

    match_seq BIGINT GENERATED ALWAYS AS IDENTITY (
        START WITH 1
        INCREMENT BY 1
        CACHE 1
    ),

    market_id UUID NOT NULL,

    resting_order_id UUID NOT NULL,
    aggressing_order_id UUID NOT NULL,

    qty BIGINT NOT NULL CHECK (qty > 0),
    yes_price_cents BIGINT NOT NULL CHECK (yes_price_cents >= 0 AND yes_price_cents <= 100),

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX ASYNC IF NOT EXISTS trades_match_seq_uq ON trades(match_seq);
CREATE INDEX ASYNC IF NOT EXISTS trades_market_seq_idx ON trades(market_id, match_seq);
CREATE INDEX ASYNC IF NOT EXISTS trades_resting_order_idx ON trades(resting_order_id);
CREATE INDEX ASYNC IF NOT EXISTS trades_aggressing_order_idx ON trades(aggressing_order_id);


-- =========================================================
-- 8. Trade parties
-- Useful to explicitly record who got what in each trade
-- =========================================================

CREATE TABLE IF NOT EXISTS trade_parties (
    trade_id UUID NOT NULL,
    account_id UUID NOT NULL,

    role TEXT NOT NULL CHECK (role IN ('BUYER', 'SELLER')),
    side_acquired TEXT NOT NULL CHECK (side_acquired IN ('YES', 'NO')),

    qty BIGINT NOT NULL CHECK (qty > 0),
    cash_delta_cents BIGINT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (trade_id, account_id)
);

CREATE INDEX ASYNC IF NOT EXISTS trade_parties_account_idx ON trade_parties(account_id);


-- =========================================================
-- 9. Cash transactions
-- Deposits / withdrawals with external payment refs
-- =========================================================

CREATE TABLE IF NOT EXISTS cash_transactions (
    cash_txn_id UUID PRIMARY KEY,
    account_id UUID NOT NULL,

    type TEXT NOT NULL CHECK (type IN ('DEPOSIT', 'WITHDRAWAL')),
    amount_cents BIGINT NOT NULL CHECK (amount_cents > 0),

    status TEXT NOT NULL CHECK (
        status IN ('PENDING', 'COMPLETED', 'FAILED', 'CANCELLED')
    ),

    external_ref TEXT,
    notes TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX ASYNC IF NOT EXISTS cash_transactions_account_idx ON cash_transactions(account_id, created_at);
CREATE INDEX ASYNC IF NOT EXISTS cash_transactions_status_idx ON cash_transactions(status);


-- =========================================================
-- 10. Ledger entries
-- Append-only accounting truth
-- =========================================================

CREATE TABLE IF NOT EXISTS ledger_entries (
    entry_id UUID PRIMARY KEY,

    account_id UUID NOT NULL,
    market_id UUID,
    order_id UUID,
    trade_id UUID,
    cash_txn_id UUID,

    cash_delta_cents BIGINT NOT NULL DEFAULT 0,
    locked_cash_delta_cents BIGINT NOT NULL DEFAULT 0,

    yes_share_delta BIGINT NOT NULL DEFAULT 0,
    no_share_delta BIGINT NOT NULL DEFAULT 0,

    locked_yes_delta BIGINT NOT NULL DEFAULT 0,
    locked_no_delta BIGINT NOT NULL DEFAULT 0,

    reason TEXT NOT NULL CHECK (
        reason IN (
            'DEPOSIT',
            'WITHDRAWAL',
            'ORDER_LOCK',
            'ORDER_UNLOCK',
            'TRADE_EXECUTION',
            'MARKET_SETTLEMENT',
            'FEE',
            'ADJUSTMENT'
        )
    ),

    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ASYNC IF NOT EXISTS ledger_entries_account_idx ON ledger_entries(account_id, created_at);
CREATE INDEX ASYNC IF NOT EXISTS ledger_entries_market_idx ON ledger_entries(market_id, created_at);
CREATE INDEX ASYNC IF NOT EXISTS ledger_entries_trade_idx ON ledger_entries(trade_id);
CREATE INDEX ASYNC IF NOT EXISTS ledger_entries_order_idx ON ledger_entries(order_id);
CREATE INDEX ASYNC IF NOT EXISTS ledger_entries_cash_txn_idx ON ledger_entries(cash_txn_id);


-- =========================================================
-- 11. Market settlements
-- Per-account settlement result after market resolution
-- =========================================================

CREATE TABLE IF NOT EXISTS market_settlements (
    settlement_id UUID PRIMARY KEY,

    market_id UUID NOT NULL,
    account_id UUID NOT NULL,

    winning_side TEXT NOT NULL CHECK (winning_side IN ('YES', 'NO', 'CANCELLED')),
    payout_cents BIGINT NOT NULL CHECK (payout_cents >= 0),

    yes_shares_settled BIGINT NOT NULL DEFAULT 0 CHECK (yes_shares_settled >= 0),
    no_shares_settled BIGINT NOT NULL DEFAULT 0 CHECK (no_shares_settled >= 0),

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (market_id, account_id)
);

CREATE INDEX ASYNC IF NOT EXISTS market_settlements_account_idx ON market_settlements(account_id);


-- =========================================================
-- 12. Matchmaker leases
-- Lightweight ownership to avoid duplicate matching work
-- =========================================================

CREATE TABLE IF NOT EXISTS matchmaker_market_leases (
    market_id UUID PRIMARY KEY,
    owner_id TEXT NOT NULL,
    lease_expires_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ASYNC IF NOT EXISTS matchmaker_market_leases_exp_idx
    ON matchmaker_market_leases(lease_expires_at);

-- =========================================================
-- 12a. Match work queue
-- Submit-time signal that a market may have a fresh cross.
-- Replaces the matchmaker's full JOIN+GROUP BY scan over `orders`
-- with O(markets-just-touched) work per poll.
-- =========================================================

CREATE TABLE IF NOT EXISTS match_work_queue (
    market_id UUID PRIMARY KEY,
    queued_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ASYNC IF NOT EXISTS match_work_queue_queued_idx
    ON match_work_queue(queued_at);

-- =========================================================
-- 13. Settlement runs
-- Global settlement state by market for executor nodes
-- =========================================================

CREATE TABLE IF NOT EXISTS market_settlement_runs (
    market_id UUID PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('IN_PROGRESS', 'COMPLETED', 'FAILED')),
    owner_id TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    settled_accounts BIGINT NOT NULL DEFAULT 0,
    error_text TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ASYNC IF NOT EXISTS market_settlement_runs_status_idx
    ON market_settlement_runs(status, updated_at);

-- =========================================================
-- 14. Executor leases
-- Lightweight ownership to avoid duplicate settlement work
-- =========================================================

CREATE TABLE IF NOT EXISTS execution_market_leases (
    market_id UUID PRIMARY KEY,
    owner_id TEXT NOT NULL,
    lease_expires_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ASYNC IF NOT EXISTS execution_market_leases_exp_idx
    ON execution_market_leases(lease_expires_at);


-- =========================================================
-- Optional helpful comments / notes
-- =========================================================
-- price_cents always refers to YES price.
-- implied NO price = 100 - YES price.
--
-- Example:
-- yes_price_cents = 57 means YES costs $0.57 and NO costs $0.43.
--
-- For v1, enforce market-specific price bounds in application logic:
--   markets.min_price_cents <= orders.price_cents <= markets.max_price_cents
--
-- Snapshot tables:
--   accounts.available_cash_cents / locked_cash_cents
--   positions.*
-- should be updated transactionally alongside ledger_entries.
--
-- Ledger is the append-only source of truth.
