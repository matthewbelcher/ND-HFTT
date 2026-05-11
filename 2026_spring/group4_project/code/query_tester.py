import json
import warnings
import wrds
import pandas as pd
import numpy as np
import pyarrow
import os

warnings.filterwarnings('ignore')
pd.set_option('mode.chained_assignment', None)

# Connection & Parameters
db = wrds.Connection(wrds_username='lukelagunowich')

etf_ticker = 'VOO'
target_date = '2024-07-09'
h_start, h_end = '09:35:00', '09:35:15'
day_start = '09:30:00' # Set to be ~5 minutes from h_start
date_str = target_date.replace('-', '')
datetime_str = date_str + h_start.replace(':', '') + h_end.replace(':', '')
taq_schema = f'taqm_{target_date[:4]}'

# tickers = ('BF/B', 'BG.UN', 'BRK/B', 'CMCSA', 'FRT.UN', 'GOOGL', 'PEAK')

# tickers = ('BF', 'BG', 'BRK', 'CMCS', 'FRT', 'GOOG', 'DOC')
tickers = ('ABST', 'ALKEY', 'EQIX', 'GEC', 'REGN', 'SGF', 'SKG.LN', 'TDG')

diag_query = f"""
    SELECT DISTINCT sym_root, sym_suffix, COUNT(*) as n
    FROM {taq_schema}.nbbom_{date_str}
    WHERE sym_root IN {tickers}
    GROUP BY sym_root, sym_suffix
"""
print(db.raw_sql(diag_query))

# db.raw_sql(diag_query).to_csv('temp.csv')

# # Tickers whose TAQ sym_root/sym_suffix differs from the constituent ticker
# ticker_map = {
#     'BF/B':   ('BF',   'B'),
#     'BG.UN':  ('BG',   None),
#     'BRK/B':  ('BRK',  'B'),
#     'CMCSA':  ('CMCS', 'A'),
#     'FRT.UN': ('FRT',  None),
#     'GOOGL':  ('GOOG', 'L'),
#     'PEAK':   ('DOC',  None),
# }
# reverse_map = {(root + (suffix or '')): orig for orig, (root, suffix) in ticker_map.items()}
# standard_tickers = tuple(t for t in ticker_list if t not in ticker_map)

# def _suffix_cond(root, suffix):
#     if suffix is None:
#         return f"(sym_root = '{root}' AND sym_suffix IS NULL)"
#     return f"(sym_root = '{root}' AND sym_suffix = '{suffix}')"

# special_where = ' OR '.join(_suffix_cond(r, s) for r, s in ticker_map.values())

# print('Basket query...')
# basket_query = f"""
#     SELECT time_m, sym_root AS ticker, best_bid, best_ask, best_bidsiz, best_asksiz
#     FROM {taq_schema}.nbbom_{date_str}
#     WHERE sym_root IN {standard_tickers}
#     AND sym_suffix IS NULL
#     AND time_m BETWEEN '{day_start}' AND '{h_end}'
# """
# special_query = f"""
#     SELECT time_m,
#            sym_root || COALESCE(sym_suffix, '') AS ticker,
#            best_bid, best_ask, best_bidsiz, best_asksiz
#     FROM {taq_schema}.nbbom_{date_str}
#     WHERE ({special_where})
#     AND time_m BETWEEN '{day_start}' AND '{h_end}'
# """
# basket_nbbo = db.raw_sql(basket_query)
# special_nbbo = db.raw_sql(special_query)
# special_nbbo['ticker'] = special_nbbo['ticker'].map(reverse_map)
# basket_nbbo = pd.concat([basket_nbbo, special_nbbo], ignore_index=True)
# bids = basket_nbbo.pivot_table(index='time_m', columns='ticker', values='best_bid').ffill()

