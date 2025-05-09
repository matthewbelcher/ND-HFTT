#API Attempt
import csv
import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time 

def get_btc_price_history_1min_coinbase(date_str):
    start_date = datetime.strptime(date_str, '%Y-%m-%d')
    current_date = datetime.now()
    if start_date > current_date:
        print(f"Error: Cannot fetch future data. Please use a date before {current_date.strftime('%Y-%m-%d')}")
        return None

    end_date = start_date + timedelta(days=1)
    start_iso = start_date.isoformat()
    end_iso = end_date.isoformat()

    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    all_data = []
    current_start = start_date

    try:
        while current_start < end_date:
            end_batch = current_start + timedelta(hours=1)
            params = {
                'start': current_start.isoformat(),
                'end': min(end_batch, end_date).isoformat(),
                'granularity': 60
            }

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_data.extend(data)
            current_start = end_batch
            time.sleep(0.1)

        if not all_data:
            print(f"No data available for {date_str}.")
            return None

        df = pd.DataFrame(all_data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df[['timestamp', 'open']].rename(columns={'open': 'price'})
        df['price'] = df['price'].astype(float)
        df = df.sort_values(by='timestamp')

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def get_btc_price_history_1min_binance(date_str):
    start_date = datetime.strptime(date_str, '%Y-%m-%d')
    current_date = datetime.now()
    if start_date > current_date:
        print(f"Error: Cannot fetch future data. Please use a date before {current_date.strftime('%Y-%m-%d')}")
        return None

    end_date = start_date + timedelta(days=1)
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

    url = "https://api.binance.us/api/v3/klines"
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1m',
        'startTime': start_ts,
        'endTime': end_ts,
        'limit': 1000
    }

    all_data = []
    current_start = start_ts

    try:
        while current_start < end_ts:
            params['startTime'] = current_start
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_data.extend(data)
            current_start = data[-1][0] + 60000
            time.sleep(0.1)

        if not all_data:
            print(f"No data available for {date_str}.")
            return None

        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 
            'volume', 'close_time', 'quote_volume', 'trades',
            'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['timestamp', 'open']].rename(columns={'open': 'price'})
        df['price'] = df['price'].astype(float)
        df = df.sort_values(by='timestamp').reset_index(drop=True)

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def find_arbitrage_opportunities(coinbase_data, binance_data):
    differences = []
    for timestamp in coinbase_data:
        if timestamp in binance_data:
            binance_price = binance_data[timestamp]
            coinbase_price = coinbase_data[timestamp]
            if binance_price != coinbase_price:
                price_difference = binance_price - coinbase_price
                differences.append({
                    'timestamp': timestamp,
                    'binance_price': round(binance_price, 2),
                    'coinbase_price': round(coinbase_price, 2),
                    'price_difference': round(price_difference, 2)
                })

    differences.sort(key=lambda x: abs(x['price_difference']), reverse=True)
    return differences

def save_arbitrage_opportunities(differences, date_str):
    filename = f'arbitrage_{date_str}.csv'
    with open(filename, 'w', newline='') as diff_file:
        fieldnames = ['timestamp', 'binance_price', 'coinbase_price', 'price_difference']
        diff_writer = csv.DictWriter(diff_file, fieldnames=fieldnames)
        diff_writer.writeheader()

        for diff in differences:
            diff_writer.writerow(diff)

    print(f"Arbitrage opportunities have been written to '{filename}'")

def plot_arbitrage_opportunities(date_str):
    df = pd.read_csv(f'arbitrage_{date_str}.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
    df['price_difference'] = df['price_difference'].abs()

    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['price_difference'], label='Price Difference (Absolute)', color='blue', linestyle='-', marker='o', markersize=4)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    threshold = 100
    plt.fill_between(df['timestamp'], df['price_difference'], threshold, where=(df['price_difference'] > threshold), color='red', alpha=0.3, label=f'Above Threshold (${threshold})')

    plt.ylim(bottom=0)
    plt.xlabel('Time')
    plt.ylabel('Price Difference (Absolute)')
    plt.title(f'Absolute Price Differences Between Binance and Coinbase on {date_str}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'arbitrage_{date_str}.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    date_str = input("Enter the date (YYYY-MM-DD) for which you want to fetch BTC price data: ")
    print(f"Fetching 1-minute BTC price data from Coinbase and Binance for {date_str}...")

    coinbase_df = get_btc_price_history_1min_coinbase(date_str)
    binance_df = get_btc_price_history_1min_binance(date_str)

    if coinbase_df is not None and binance_df is not None:
        coinbase_data = dict(zip(coinbase_df['timestamp'], coinbase_df['price']))
        binance_data = dict(zip(binance_df['timestamp'], binance_df['price']))

        differences = find_arbitrage_opportunities(coinbase_data, binance_data)
        save_arbitrage_opportunities(differences, date_str)
        plot_arbitrage_opportunities(date_str)

if __name__ == "__main__":
    main()