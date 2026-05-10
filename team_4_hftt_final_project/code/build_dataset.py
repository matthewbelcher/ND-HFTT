import json
import warnings
import datetime
import wrds
import pandas as pd
import numpy as np
import pyarrow
import os

warnings.filterwarnings('ignore')
pd.set_option('mode.chained_assignment', None)

# Fields
OUT_DIR_PATH = 'data_V2'
ETF_TICKER   = 'SPY'
INTERVAL_MIN = 0
INTERVAL_SEC = 15
MARKET_OPEN  = datetime.time(9, 35, 00) # set market open to start of data window (9:35 for 9:30 NYSE opening bell)
MARKET_CLOSE = datetime.time(16, 00, 00)
QUERY_START    = '09:34:00' # start query roughly 1 to 5 minutes before market open/day start to prevent missing quotes

# Week of July 8-12, 2024 (calm week)
# (day, prev_day)
# DAYS = [
#     ('2024-07-08', '2024-07-05'),
#     ('2024-07-09', '2024-07-08'),
#     ('2024-07-10', '2024-07-09'),
#     ('2024-07-11', '2024-07-10'),
#     ('2024-07-12', '2024-07-11'),
# ]

# September 18, 2024 (FOMC release at 2pm)
# (day, prev_day)
# DAYS = [
#     ('2024-09-18','2024-09-17') 
# ]

# # Week of July 15-19, 2024 (earnings week)
# DAYS = [
#     ('2024-07-15', '2024-07-12'),
#     ('2024-07-16', '2024-07-15'),
#     ('2024-07-17', '2024-07-16'),
#     ('2024-07-18', '2024-07-17'),
#     ('2024-07-19', '2024-07-18'),
# ]

# April 3, 2025 (post-Liberation Day tariff announcement)
# (day, prev_day)
DAYS = [
    ('2025-04-03','2025-04-02') 
]


if DAYS is None:
    print('No dates selected')
    exit()

def generate_intervals(open_t, close_t, interval_min):
    result, dt = [], datetime.datetime.combine(datetime.date.today(), open_t)
    close_dt   = datetime.datetime.combine(datetime.date.today(), close_t)
    delta      = datetime.timedelta(minutes=interval_min, seconds=INTERVAL_SEC)
    while dt < close_dt:
        result.append((dt.time(), (dt + delta).time()))
        dt += delta
    return result

INTERVALS = generate_intervals(MARKET_OPEN, MARKET_CLOSE, INTERVAL_MIN)

# Map of constituent tickers from massive.com to TAQ (sym_root, sym_suffix) pairs.
TICKER_MAP_MASTER = {
    'BF/B':   [('BF',   'B')],
    'BF B':   [('BF',   'B')],
    'BG.UN':  [('BG',   None)],
    'BRK/B':  [('BRK',  'B')],
    'BRK B':  [('BRK',  'B')],
    'CMCSA':  [('CMCS', 'A')],
    'FRT.UN': [('FRT',  None)],
    'GOOGL':  [('GOOG', 'L')],
    'PEAK':   [('DOC',  None)],
    'GEV-W':  [('GEV',  None)],
    'SKG.LN': [],                 
    'LEN':    [('LEN',  None)],
    'BIO':    [('BIO',  None)],
    'TAP':    [('TAP',  None)],
    'PARA':   [('PARA', 'A'), ('PARA', None)],
    'WRK':    [],
    'ABST': [],
    'ALKEY': [],
    'SGF': [],
}

def _suffix_cond(root, suffix):
    if suffix is None:
        return f"(sym_root = '{root}' AND sym_suffix IS NULL)"
    return f"(sym_root = '{root}' AND sym_suffix = '{suffix}')"

# WRDS connection
db = wrds.Connection(wrds_username='lukelagunowich')

for target_date, prev_date in DAYS:
    date_str      = target_date.replace('-', '')
    prev_date_str = prev_date.replace('-', '')
    taq_schema    = f'taqm_{target_date[:4]}'
    out_dir       = f'{OUT_DIR_PATH}/{date_str}'

    try:
        with open(f'{OUT_DIR_PATH}/{ETF_TICKER}_constituents_{date_str}.json') as f:
            const_data = json.load(f)
        with open(f'{OUT_DIR_PATH}/{ETF_TICKER}_constituents_{prev_date_str}.json') as f:
            prev_const_data = json.load(f)
        with open(f'{OUT_DIR_PATH}/{ETF_TICKER}_ff_{prev_date_str}.json') as f:
            ff_data = json.load(f)
    except FileNotFoundError as e:
        print(f'{target_date}: Missing fund flows or constituent data')
        exit()

    os.makedirs(out_dir, exist_ok=True)

    shrout       = ff_data['results'][0]['shares_outstanding']
    const_shares = {r['constituent_ticker']: r['shares_held'] for r in const_data['results'] if 'constituent_ticker' in r and 'shares_held' in r}
    usd_today    = sum(r.get('shares_held', r.get('market_value', 0)) for r in const_data['results']      if 'constituent_ticker' not in r)
    usd_prev     = sum(r.get('shares_held', r.get('market_value', 0)) for r in prev_const_data['results'] if 'constituent_ticker' not in r)
    usd          = (usd_today + usd_prev) / 2
    print(f'usd_today={usd_today:,.0f} | usd_prev={usd_prev:,.0f} | usd_avg={usd:,.0f}')
    ticker_list  = tuple(const_shares.keys())

    ticker_map    = {t: pairs for t, pairs in TICKER_MAP_MASTER.items() if t in const_shares}
    reverse_map   = {(root + (suffix or '')): orig for orig, pairs in ticker_map.items() for root, suffix in pairs if pairs}

    standard_tickers = tuple(t for t in ticker_list if t not in ticker_map)
    special_where    = ' OR '.join(
        _suffix_cond(r, s)
        for pairs in ticker_map.values()
        for r, s in pairs 
    )

    print(f'\n{target_date} | shrout={shrout:,.0f} | {len(ticker_list)} tickers')

    scale       = None # Now permanently set to 1.0 (handled in dynamic rescale)
    prev_bid    = None   
    prev_ask    = None
    prev_bidsiz = None
    prev_asksiz = None

    for h_start_t, h_end_t in INTERVALS:
        h_start = h_start_t.strftime('%H:%M:%S')
        h_end   = h_end_t.strftime('%H:%M:%S')

        if prev_bid is None:
            query_start = QUERY_START
        else:
            query_start = h_start

        # Basket query
        print('Basket query...')
        basket_query = f"""
            SELECT time_m, sym_root AS ticker, best_bid, best_ask, best_bidsiz, best_asksiz
            FROM {taq_schema}.nbbom_{date_str}
            WHERE sym_root IN {standard_tickers}
            AND sym_suffix IS NULL
            AND time_m BETWEEN '{query_start}' AND '{h_end}'
        """
        special_query = f"""
            SELECT time_m,
                   sym_root || COALESCE(sym_suffix, '') AS ticker,
                   best_bid, best_ask, best_bidsiz, best_asksiz
            FROM {taq_schema}.nbbom_{date_str}
            WHERE ({special_where})
            AND time_m BETWEEN '{query_start}' AND '{h_end}'
        """
        basket_nbbo            = db.raw_sql(basket_query)
        special_nbbo           = db.raw_sql(special_query)
        special_nbbo['ticker'] = special_nbbo['ticker'].map(reverse_map)
        basket_nbbo            = pd.concat([basket_nbbo, special_nbbo], ignore_index=True)

        # Inject seed row from previous interval (all intervals after the first)
        if prev_bid is not None:
            seed_time = (
                datetime.datetime.combine(datetime.date.today(), h_start_t)
                - datetime.timedelta(microseconds=1)
            ).time()
            seed_rows = [
                {'time_m': seed_time, 'ticker': t,
                 'best_bid':    prev_bid.get(t),
                 'best_ask':    prev_ask.get(t),
                 'best_bidsiz': prev_bidsiz.get(t),
                 'best_asksiz': prev_asksiz.get(t)}
                for t in prev_bid.index
                if pd.notna(prev_bid.get(t))
            ]
            if seed_rows:
                basket_nbbo = pd.concat(
                    [pd.DataFrame(seed_rows), basket_nbbo], ignore_index=True
                )

        # Pivot and ffill
        print('Pivot and ffill...')
        basket_nbbo['time_m'] = pd.to_datetime(basket_nbbo['time_m'].astype(str)).dt.time
        bids    = basket_nbbo.pivot_table(index='time_m', columns='ticker', values='best_bid').ffill()
        asks    = basket_nbbo.pivot_table(index='time_m', columns='ticker', values='best_ask').ffill()
        bid_szs = basket_nbbo.pivot_table(index='time_m', columns='ticker', values='best_bidsiz').ffill()
        ask_szs = basket_nbbo.pivot_table(index='time_m', columns='ticker', values='best_asksiz').ffill()

        basket_tickers = set(bids.columns)
        const_tickers  = set(const_shares.keys())
        missing_from_basket = const_tickers - basket_tickers
        extra_in_basket     = basket_tickers - const_tickers
        if missing_from_basket:
            print(f'  MISSING from basket (no TAQ quote): {sorted(missing_from_basket)}')
        if extra_in_basket:
            print(f'  EXTRA in basket (not in constituents): {sorted(extra_in_basket)}')

        if len(bids) > 0:
            prev_bid    = bids.iloc[-1]
            prev_ask    = asks.iloc[-1]
            prev_bidsiz = bid_szs.iloc[-1]
            prev_asksiz = ask_szs.iloc[-1]

        bids    = bids[bids.index >= h_start_t]
        asks    = asks[asks.index >= h_start_t]
        bid_szs = bid_szs[bid_szs.index >= h_start_t]
        ask_szs = ask_szs[ask_szs.index >= h_start_t]

        # Coverage filter
        min_coverage = int(1.0 * bids.shape[1])
        mask = bids.notna().sum(axis=1) >= min_coverage
        bids    = bids[mask]
        asks    = asks[mask]
        bid_szs = bid_szs[mask]
        ask_szs = ask_szs[mask]

        if len(bids) == 0:
            print(f'{h_start}: no data remaining after coverage filter, skipping')
            continue

        # Basket price
        print('Generate basket price...')
        I    = bid_szs / (bid_szs + ask_szs)
        mids = bids * (1 - I) + asks * I

        synth_bid_raw = (bids.mul(pd.Series(const_shares), axis=1).sum(axis=1) + usd) / shrout
        synth_ask_raw = (asks.mul(pd.Series(const_shares), axis=1).sum(axis=1) + usd) / shrout
        synth_mid_raw = (mids.mul(pd.Series(const_shares), axis=1).sum(axis=1) + usd) / shrout

        arb_df = pd.DataFrame({
            'synth_bid_raw': synth_bid_raw,
            'synth_ask_raw': synth_ask_raw,
            'synth_mid_raw': synth_mid_raw,
        }).reset_index().sort_values('time_m')
        arb_df['time_m'] = pd.to_datetime(target_date + ' ' + arb_df['time_m'].astype(str), format='mixed')

        # ETF query
        print('ETF query...')
        etf_query = f"""
            SELECT time_m, best_bid AS etf_bid, best_ask AS etf_ask, best_bidsiz, best_asksiz
            FROM {taq_schema}.nbbom_{date_str}
            WHERE sym_root = '{ETF_TICKER}'
            AND time_m BETWEEN '{h_start}' AND '{h_end}'
        """
        etf_nbbo            = db.raw_sql(etf_query)
        etf_nbbo['time_m']  = pd.to_datetime(target_date + ' ' + etf_nbbo['time_m'].astype(str), format='mixed')
        I_etf               = etf_nbbo['best_bidsiz'] / (etf_nbbo['best_bidsiz'] + etf_nbbo['best_asksiz'])
        etf_nbbo['etf_mid'] = etf_nbbo['etf_bid'] * (1 - I_etf) + etf_nbbo['etf_ask'] * I_etf

        print('Merging basket and ETF...')
        arb_df = pd.merge_asof(arb_df, etf_nbbo.sort_values('time_m'), on='time_m', direction='backward')
        arb_df[['etf_bid', 'etf_ask', 'etf_mid']] = arb_df[['etf_bid', 'etf_ask', 'etf_mid']].ffill()

        # Scale (degredated)
        if scale is None:
            # idx = arb_df['etf_mid'].first_valid_index()
            # if idx is not None:
            #     scale = arb_df.loc[idx, 'etf_mid'] / arb_df.loc[idx, 'synth_mid_raw']
            #     print(f'  Scale: {scale:.6f} (set at {h_start}, held constant for {target_date})')
            scale = 1.0
            print(f'  Scale: {scale:.6f} (set at {h_start}, held constant for {target_date})')

        if scale is None:
            continue

        arb_df['synth_bid'] = arb_df['synth_bid_raw'] * scale
        arb_df['synth_ask'] = arb_df['synth_ask_raw'] * scale
        arb_df['synth_mid'] = arb_df['synth_mid_raw'] * scale

        # Save
        final_dataset = arb_df[['time_m', 'synth_bid', 'synth_ask', 'etf_bid', 'etf_ask', 'etf_mid', 'synth_mid']]
        h_start_str   = h_start.replace(':', '')
        h_end_str     = h_end.replace(':', '')
        out_path      = f'{out_dir}/{ETF_TICKER}_{date_str}_{h_start_str}_{h_end_str}.parquet'
        final_dataset.to_parquet(out_path)
        print(f'{h_start}–{h_end}: {len(final_dataset):>6} rows saved to {out_path}')
