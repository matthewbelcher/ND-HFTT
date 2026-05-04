-- =========================================================
-- Prediction Market Exchange Schema
-- v1 starter schema
-- =========================================================

-- =========================================================
-- 1. Accounts
-- =========================================================

CREATE TABLE accounts (
    account_id UUID PRIMARY KEY,
    external_user_id TEXT UNIQUE,
    username TEXT UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'SUSPENDED', 'CLOSED')),

    available_cash_cents BIGINT NOT NULL DEFAULT 0 CHECK (available_cash_cents >= 0),
    locked_cash_cents BIGINT NOT NULL DEFAULT 0 CHECK (locked_cash_cents >= 0),

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX accounts_status_idx ON accounts(status);


-- =========================================================
-- 2. Markets
-- =========================================================

CREATE TABLE markets (
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

    created_by UUID REFERENCES accounts(account_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CHECK (min_price_cents <= max_price_cents),
    CHECK (max_price_cents <= 100)
);

CREATE INDEX markets_status_idx ON markets(status);
CREATE INDEX markets_close_time_idx ON markets(close_time);


-- =========================================================
-- 3. Market resolutions
-- One row per resolved/cancelled market outcome
-- =========================================================

CREATE TABLE market_resolutions (
    resolution_id UUID PRIMARY KEY,
    market_id UUID NOT NULL UNIQUE REFERENCES markets(market_id),

    outcome TEXT NOT NULL CHECK (outcome IN ('YES', 'NO', 'CANCELLED')),
    resolved_by UUID REFERENCES accounts(account_id),
    notes TEXT,

    resolved_at TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- =========================================================
-- 4. Positions
-- Snapshot table for fast reads
-- =========================================================

CREATE TABLE positions (
    account_id UUID NOT NULL REFERENCES accounts(account_id),
    market_id UUID NOT NULL REFERENCES markets(market_id),

    yes_shares BIGINT NOT NULL DEFAULT 0 CHECK (yes_shares >= 0),
    no_shares BIGINT NOT NULL DEFAULT 0 CHECK (no_shares >= 0),

    locked_yes_shares BIGINT NOT NULL DEFAULT 0 CHECK (locked_yes_shares >= 0),
    locked_no_shares BIGINT NOT NULL DEFAULT 0 CHECK (locked_no_shares >= 0),

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (account_id, market_id)
);

CREATE INDEX positions_market_idx ON positions(market_id);


-- =========================================================
-- 5. Orders
-- Limit orders only for v1
-- YES / NO side for prediction markets
-- =========================================================

CREATE TABLE orders (
    request_id UUID PRIMARY KEY,

    global_seq BIGINT GENERATED ALWAYS AS IDENTITY (
        START WITH 1
        INCREMENT BY 1
        CACHE 1
    ),

    account_id UUID NOT NULL REFERENCES accounts(account_id),
    market_id UUID NOT NULL REFERENCES markets(market_id),

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

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX orders_global_seq_uq ON orders(global_seq);
CREATE INDEX orders_market_seq_idx ON orders(market_id, global_seq);
CREATE INDEX orders_account_seq_idx ON orders(account_id, global_seq);
CREATE INDEX orders_market_status_price_idx ON orders(market_id, status, price_cents);
CREATE INDEX orders_market_side_status_price_idx ON orders(market_id, side, status, price_cents);


-- =========================================================
-- 6. Order cancels
-- Separate audit/event table for cancellation requests
-- =========================================================

CREATE TABLE order_cancels (
    cancel_id UUID PRIMARY KEY,
    order_id UUID NOT NULL REFERENCES orders(request_id),
    account_id UUID NOT NULL REFERENCES accounts(account_id),

    cancel_seq BIGINT GENERATED ALWAYS AS IDENTITY (
        START WITH 1
        INCREMENT BY 1
        CACHE 1
    ),

    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX order_cancels_seq_uq ON order_cancels(cancel_seq);
CREATE INDEX order_cancels_order_idx ON order_cancels(order_id);


-- =========================================================
-- 7. Trades
-- One row per execution
-- =========================================================

CREATE TABLE trades (
    trade_id UUID PRIMARY KEY,

    match_seq BIGINT GENERATED ALWAYS AS IDENTITY (
        START WITH 1
        INCREMENT BY 1
        CACHE 1
    ),

    market_id UUID NOT NULL REFERENCES markets(market_id),

    resting_order_id UUID NOT NULL REFERENCES orders(request_id),
    aggressing_order_id UUID NOT NULL REFERENCES orders(request_id),

    qty BIGINT NOT NULL CHECK (qty > 0),
    yes_price_cents BIGINT NOT NULL CHECK (yes_price_cents >= 0 AND yes_price_cents <= 100),

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX trades_match_seq_uq ON trades(match_seq);
CREATE INDEX trades_market_seq_idx ON trades(market_id, match_seq);
CREATE INDEX trades_resting_order_idx ON trades(resting_order_id);
CREATE INDEX trades_aggressing_order_idx ON trades(aggressing_order_id);


-- =========================================================
-- 8. Trade parties
-- Useful to explicitly record who got what in each trade
-- =========================================================

CREATE TABLE trade_parties (
    trade_id UUID NOT NULL REFERENCES trades(trade_id),
    account_id UUID NOT NULL REFERENCES accounts(account_id),

    role TEXT NOT NULL CHECK (role IN ('BUYER', 'SELLER')),
    side_acquired TEXT NOT NULL CHECK (side_acquired IN ('YES', 'NO')),

    qty BIGINT NOT NULL CHECK (qty > 0),
    cash_delta_cents BIGINT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (trade_id, account_id)
);

CREATE INDEX trade_parties_account_idx ON trade_parties(account_id);


-- =========================================================
-- 9. Cash transactions
-- Deposits / withdrawals with external payment refs
-- =========================================================

CREATE TABLE cash_transactions (
    cash_txn_id UUID PRIMARY KEY,
    account_id UUID NOT NULL REFERENCES accounts(account_id),

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

CREATE INDEX cash_transactions_account_idx ON cash_transactions(account_id, created_at);
CREATE INDEX cash_transactions_status_idx ON cash_transactions(status);


-- =========================================================
-- 10. Ledger entries
-- Append-only accounting truth
-- =========================================================

CREATE TABLE ledger_entries (
    entry_id UUID PRIMARY KEY,

    account_id UUID NOT NULL REFERENCES accounts(account_id),
    market_id UUID REFERENCES markets(market_id),
    order_id UUID REFERENCES orders(request_id),
    trade_id UUID REFERENCES trades(trade_id),
    cash_txn_id UUID REFERENCES cash_transactions(cash_txn_id),

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

CREATE INDEX ledger_entries_account_idx ON ledger_entries(account_id, created_at);
CREATE INDEX ledger_entries_market_idx ON ledger_entries(market_id, created_at);
CREATE INDEX ledger_entries_trade_idx ON ledger_entries(trade_id);
CREATE INDEX ledger_entries_order_idx ON ledger_entries(order_id);
CREATE INDEX ledger_entries_cash_txn_idx ON ledger_entries(cash_txn_id);


-- =========================================================
-- 11. Market settlements
-- Per-account settlement result after market resolution
-- =========================================================

CREATE TABLE market_settlements (
    settlement_id UUID PRIMARY KEY,

    market_id UUID NOT NULL REFERENCES markets(market_id),
    account_id UUID NOT NULL REFERENCES accounts(account_id),

    winning_side TEXT NOT NULL CHECK (winning_side IN ('YES', 'NO', 'CANCELLED')),
    payout_cents BIGINT NOT NULL CHECK (payout_cents >= 0),

    yes_shares_settled BIGINT NOT NULL DEFAULT 0 CHECK (yes_shares_settled >= 0),
    no_shares_settled BIGINT NOT NULL DEFAULT 0 CHECK (no_shares_settled >= 0),

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (market_id, account_id)
);

CREATE INDEX market_settlements_account_idx ON market_settlements(account_id);


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
