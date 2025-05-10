# OANDA Forex Data Fetcher

A Python utility for fetching foreign exchange (forex) data from the OANDA API. This tool allows you to download historical and real-time price data for various currency pairs.

## Features

- Fetch the latest real-time candle data
- Download historical data for specific date ranges
- Support for multiple timeframes (M1, M5, M15, M30, H1, H4, D, W, M)
- Built-in rate limiting and error handling
- Data validation and processing
- Save data to CSV files

## Prerequisites

- Python 3.8 or higher
- OANDA API access (trading account and API key)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/oanda-forex-data-fetcher.git
   cd oanda-forex-data-fetcher
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your API credentials by copying the example `.env` file:
   ```
   cp .env.example .env
   ```

5. Edit the `.env` file with your OANDA API key:
   ```
   OANDA_API_KEY=your_api_key_here
   ```

## Usage

### Basic Usage

Run the script with default settings:

```
python fetch_oanda_data.py
```

This will:
- Fetch the latest 100 H1 candles for EUR/USD
- Download historical data for EUR/USD for the last 30 days
- Save all data to CSV files in the `data` directory

### Customizing in Main Function

You can customize the data fetching by modifying the `main()` function in `fetch_oanda_data.py`. Examples:

```python
# Fetch the latest 50 candles for GBP/USD with 15-minute timeframe
latest_df = fetcher.get_latest_candles("GBP_USD", count=50, timeframe="M15")

# Fetch historical data for USD/JPY for a specific date range
from datetime import datetime
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 3, 31)
historical_df = fetcher.get_historical_candles(
    "USD_JPY", 
    start_time=start_date,
    end_time=end_date,
    timeframe="H4"
)
```

### Available Timeframes

- M1: 1 minute
- M5: 5 minutes
- M15: 15 minutes
- M30: 30 minutes
- H1: 1 hour
- H4: 4 hours
- D: 1 day
- W: 1 week
- M: 1 month

## API Rate Limits

OANDA has rate limits for API requests. This tool implements automatic rate limiting to avoid exceeding these limits. If you're fetching large amounts of data, the process may take some time due to these limits.

## Data Output

Data is saved to CSV files in the `data` directory with filenames following this pattern:
- `{instrument}_{timeframe}_{date}.csv`

Example: `EUR_USD_H1_20250331.csv`

Each CSV contains the following columns:
- timestamp: The candle timestamp
- open: Opening price
- high: Highest price during the period
- low: Lowest price during the period
- close: Closing price
- volume: Trading volume

## Error Handling

The tool includes error handling and logging. Logs are written to:
- Console (standard output)
- `oanda_fetcher.log` file

## License

This project is licensed under the MIT License - see the LICENSE file for details.