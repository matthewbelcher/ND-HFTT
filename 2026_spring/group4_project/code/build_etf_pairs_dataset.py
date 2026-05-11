import warnings
import datetime
import wrds
import pandas as pd
import os

warnings.filterwarnings('ignore')
pd.set_option('mode.chained_assignment', None)

# Fields
OUT_DIR_PATH = 'data_etf_pairs'
ETF_TICKERS  = ['SPY', 'VOO', 'IVV']   # ETF tickers to query
INTERVAL_MIN = 30
INTERVAL_SEC = 0
MARKET_OPEN  = datetime.time(9, 30, 0)
MARKET_CLOSE = datetime.time(16, 0, 0)
QUERY_START    = '09:30:00'

DAYS = [
    ('2024-07-08', '2024-07-05'),
    ('2024-07-09', '2024-07-08'),
    ('2024-07-10', '2024-07-09'),
    ('2024-07-11', '2024-07-10'),
    ('2024-07-12', '2024-07-11'),
]

def generate_intervals(open_t, close_t, interval_min):
    result, dt = [], datetime.datetime.combine(datetime.date.today(), open_t)
    close_dt   = datetime.datetime.combine(datetime.date.today(), close_t)
    delta      = datetime.timedelta(minutes=interval_min, seconds=INTERVAL_SEC)
    while dt < close_dt:
        result.append((dt.time(), (dt + delta).time()))
        dt += delta
    return result

INTERVALS = generate_intervals(MARKET_OPEN, MARKET_CLOSE, INTERVAL_MIN)

# WRDS Connection
db = wrds.Connection(wrds_username='lukelagunowich')

for target_date, prev_date in DAYS:
    date_str   = target_date.replace('-', '')
    taq_schema = f'taqm_{target_date[:4]}'
    out_dir    = f'{OUT_DIR_PATH}/{date_str}'
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n{target_date} | ETFs: {ETF_TICKERS}')

    prev_quotes = None

    for h_start_t, h_end_t in INTERVALS:
        h_start = h_start_t.strftime('%H:%M:%S')
        h_end   = h_end_t.strftime('%H:%M:%S')

        query_start = QUERY_START if prev_quotes is None else h_start

        tickers_sql = tuple(ETF_TICKERS) if len(ETF_TICKERS) > 1 else f"('{ETF_TICKERS[0]}')"

        query = f"""
            SELECT time_m, sym_root AS ticker, best_bid, best_ask, best_bidsiz, best_asksiz
            FROM {taq_schema}.nbbom_{date_str}
            WHERE sym_root IN {tuple(ETF_TICKERS)}
            AND sym_suffix IS NULL
            AND time_m BETWEEN '{query_start}' AND '{h_end}'
        """
        df = db.raw_sql(query)
        df['time_m'] = pd.to_datetime(df['time_m'].astype(str)).dt.time

        # Inject seed row from previous interval
        if prev_quotes is not None:
            seed_time = (
                datetime.datetime.combine(datetime.date.today(), h_start_t)
                - datetime.timedelta(microseconds=1)
            ).time()
            seed_rows = [
                {'time_m': seed_time, 'ticker': t,
                 'best_bid':    row['best_bid'],
                 'best_ask':    row['best_ask'],
                 'best_bidsiz': row['best_bidsiz'],
                 'best_asksiz': row['best_asksiz']}
                for t, row in prev_quotes.items()
            ]
            df = pd.concat([pd.DataFrame(seed_rows), df], ignore_index=True)

        # Pivot and ffill
        bids    = df.pivot_table(index='time_m', columns='ticker', values='best_bid').ffill()
        asks    = df.pivot_table(index='time_m', columns='ticker', values='best_ask').ffill()
        bid_szs = df.pivot_table(index='time_m', columns='ticker', values='best_bidsiz').ffill()
        ask_szs = df.pivot_table(index='time_m', columns='ticker', values='best_asksiz').ffill()

        # Capture last row
        if len(bids) > 0:
            prev_quotes = {t: {'best_bid': bids[t].iloc[-1], 'best_ask': asks[t].iloc[-1],
                               'best_bidsiz': bid_szs[t].iloc[-1], 'best_asksiz': ask_szs[t].iloc[-1]}
                           for t in bids.columns if pd.notna(bids[t].iloc[-1])}

        # Trim to interval
        bids    = bids[bids.index >= h_start_t]
        asks    = asks[asks.index >= h_start_t]
        bid_szs = bid_szs[bid_szs.index >= h_start_t]
        ask_szs = ask_szs[ask_szs.index >= h_start_t]

        if len(bids) == 0:
            print(f'  {h_start}: no data, skipping')
            continue

        # Compute mid for each ETF
        I    = bid_szs / (bid_szs + ask_szs)
        mids = bids * (1 - I) + asks * I

        # Flatten to one row per timestamp and column per ETF
        out = pd.DataFrame(index=bids.index)
        for t in ETF_TICKERS:
            if t not in bids.columns:
                continue
            out[f'{t}_bid'] = bids[t]
            out[f'{t}_ask'] = asks[t]
            out[f'{t}_mid'] = mids[t]

        out = out.reset_index()
        out['time_m'] = pd.to_datetime(target_date + ' ' + out['time_m'].astype(str), format='mixed')

        h_start_str = h_start.replace(':', '')
        h_end_str   = h_end.replace(':', '')
        out_path    = f'{out_dir}/etf_pairs_{date_str}_{h_start_str}_{h_end_str}.parquet'
        out.to_parquet(out_path)
        print(f'{h_start}–{h_end}: {len(out):>6} rows saved to {out_path}')
