import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY    = os.environ['MASSIVE_API_KEY']
BASE_URL   = 'https://api.massive.com/etf-global/v1'
ETF        = 'SPY'
OUT_DIR    = 'data_v2'

# # Dates needed for the week of July 8-12, 2024
# CONSTITUENT_DATES = [
#     '2024-07-08',
#     '2024-07-09',
#     '2024-07-10',
#     '2024-07-11',
#     '2024-07-12',
# ]

# FUND_FLOW_DATES = [
#     '2024-07-05',  # prev day for 2024-07-08
#     '2024-07-08',  # prev day for 2024-07-09
#     '2024-07-09',  # prev day for 2024-07-10
#     '2024-07-10',  # prev day for 2024-07-11
#     '2024-07-11',  # prev day for 2024-07-12
# ]

# Dates needed for the week of July 15-19, 2024
# CONSTITUENT_DATES = [
#     '2024-07-12',  # prev day for 2024-07-15
#     '2024-07-15',
#     '2024-07-16',
#     '2024-07-17',
#     '2024-07-18',
#     '2024-07-19',
# ]

# FUND_FLOW_DATES = [
#     '2024-07-12',  # prev day for 2024-07-15
#     '2024-07-15',  # prev day for 2024-07-16
#     '2024-07-16',  # prev day for 2024-07-17
#     '2024-07-17',  # prev day for 2024-07-18
#     '2024-07-18',  # prev day for 2024-07-19
# ]

# September 18, 2024 (FOMC release at 2pm)
# CONSTITUENT_DATES = [
#     '2024-09-17',
#     '2024-09-18'
# ]

# FUND_FLOW_DATES = [
#     '2024-09-17',
#     '2024-09-18'
# ]

# April 3, 2025 (post-Liberation Day tariff announcement)
CONSTITUENT_DATES = [
    '2025-04-02',
    '2025-04-03'
]

FUND_FLOW_DATES = [
    '2025-04-02',
    '2025-04-03'
]

os.makedirs(OUT_DIR, exist_ok=True)

def download(url, out_path):
    if os.path.exists(out_path):
        print(f'Already exists, skipping: {out_path}')
        return
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(out_path, 'w') as f:
        f.write(resp.text)
    print(f'Saved: {out_path}')

print('Downloading constituents...')
for date in CONSTITUENT_DATES:
    date_str = date.replace('-', '')
    url = (
        f'{BASE_URL}/constituents'
        f'?composite_ticker={ETF}'
        f'&processed_date={date}'
        f'&limit=550'
        f'&sort=composite_ticker.asc'
        f'&apiKey={API_KEY}'
    )
    out_path = f'{OUT_DIR}/{ETF}_constituents_{date_str}.json'
    download(url, out_path)

print('\nDownloading fund flows...')
for date in FUND_FLOW_DATES:
    date_str = date.replace('-', '')
    url = (
        f'{BASE_URL}/fund-flows'
        f'?processed_date={date}'
        f'&composite_ticker={ETF}'
        f'&limit=100'
        f'&sort=composite_ticker.asc'
        f'&apiKey={API_KEY}'
    )
    out_path = f'{OUT_DIR}/{ETF}_ff_{date_str}.json'
    download(url, out_path)
