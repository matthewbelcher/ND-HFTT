"""
Module for fetching forex data from OANDA API.
Enhanced to support statistical arbitrage with multi-pair fetching capabilities.
"""

import os
import time
import logging
from typing import Optional, Dict, List, Any, Union
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import concurrent.futures

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('OANDAFetcher')

# Constants
OANDA_API_URL = os.getenv('OANDA_API_URL', 'https://api-fxtrade.oanda.com/v3')
INSTRUMENTS = os.getenv('INSTRUMENTS', 'EUR_USD,GBP_USD,USD_JPY,USD_CAD').split(',')

class OandaDataFetcher:
    """
    Enhanced class for fetching forex data from OANDA API with support for
    fetching multiple currency pairs simultaneously.
    """
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OandaDataFetcher with API credentials."""
        self.api_key = api_key or os.getenv('OANDA_API_KEY')

        if not self.api_key:
            raise ValueError("Please provide OANDA_API_KEY either as parameter or in .env file")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept-Datetime-Format": "RFC3339"
        }

        self.max_retries = 3
        self.retry_delay = 1
        self.request_timeout = 30
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = datetime.now()
        self.rate_limit_pause = 1.0  # seconds between requests
        
        # Set maximum workers for parallel requests
        self.max_workers = min(10, os.cpu_count() * 2)
        
        logger.info(f"Initialized OandaDataFetcher with max_workers={self.max_workers}")

    def _handle_rate_limit(self):
        """Implement rate limiting"""
        current_time = datetime.now()
        time_since_last_request = (current_time - self.last_request_time).total_seconds()

        if time_since_last_request < self.rate_limit_pause:
            sleep_time = self.rate_limit_pause - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = datetime.now()
        self.request_count += 1

    def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        retry_count: int = 0
    ) -> Optional[Dict]:
        """Make an API request with retry logic."""
        self._handle_rate_limit()
        
        try:
            response = requests.get(
                f"{OANDA_API_URL}/{endpoint}",
                headers=self.headers,
                params=params,
                timeout=self.request_timeout
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Too many requests
                logger.warning("Rate limit hit, increasing pause time")
                self.rate_limit_pause *= 2
                time.sleep(self.rate_limit_pause)
                return self._make_request(endpoint, params)  # Retry
            else:
                if retry_count < self.max_retries:
                    logger.warning(f"Request failed, retrying ({retry_count + 1}/{self.max_retries}): {response.status_code} - {response.text}")
                    time.sleep(self.retry_delay * (2 ** retry_count))
                    return self._make_request(endpoint, params, retry_count + 1)
                else:
                    logger.error(f"Request failed after {self.max_retries} retries: {response.status_code} - {response.text}")
                    return None

        except requests.exceptions.RequestException as e:
            if retry_count < self.max_retries:
                logger.warning(f"Request error, retrying ({retry_count + 1}/{self.max_retries}): {str(e)}")
                time.sleep(self.retry_delay * (2 ** retry_count))
                return self._make_request(endpoint, params, retry_count + 1)
            else:
                logger.error(f"Request failed after {self.max_retries} retries: {str(e)}")
                return None

    def validate_instrument(self, instrument: str) -> None:
        """Validate that the instrument is supported."""
        if instrument not in INSTRUMENTS:
            raise ValueError(f"Invalid instrument: {instrument}. Must be one of {INSTRUMENTS}")

    def validate_timeframe(self, timeframe: str) -> str:
        """Validate that the timeframe is supported."""
        valid_timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D", "W", "M"]
        if timeframe not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}")
        return timeframe

    def get_latest_candles(
        self,
        instrument: str,
        count: int = 100,
        timeframe: str = "H1"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch the most recent candles for a given instrument.
        
        Args:
            instrument: Currency pair to fetch (e.g., "EUR_USD")
            count: Number of candles to fetch
            timeframe: Candle timeframe (e.g., "H1" for hourly)
            
        Returns:
            DataFrame with candle data or None if request failed
        """
        self.validate_instrument(instrument)
        timeframe = self.validate_timeframe(timeframe)
        
        endpoint = f"instruments/{instrument}/candles"
        params = {
            "price": "M",  # Midpoint prices
            "granularity": timeframe,
            "count": count
        }
        
        logger.info(f"Fetching {count} latest {timeframe} candles for {instrument}")
        response_data = self._make_request(endpoint, params)
        
        if response_data and 'candles' in response_data:
            return self._process_candles(response_data['candles'])
        return None

    def get_historical_candles(
        self,
        instrument: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        timeframe: str = "H1"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical candles for a specific time period.
        
        Args:
            instrument: Currency pair to fetch (e.g., "EUR_USD")
            start_time: Start time for data fetch
            end_time: End time for data fetch (defaults to now)
            timeframe: Candle timeframe (e.g., "H1" for hourly)
            
        Returns:
            DataFrame with candle data or None if request failed
        """
        self.validate_instrument(instrument)
        timeframe = self.validate_timeframe(timeframe)
        
        if end_time is None:
            end_time = datetime.now()
            
        # Split into chunks to avoid hitting API limits
        date_ranges = self._calculate_date_ranges(start_time, end_time)
        all_candles = []
        
        for start, end in date_ranges:
            endpoint = f"instruments/{instrument}/candles"
            params = {
                "price": "M",  # Midpoint prices
                "granularity": timeframe,
                "from": start.isoformat() + "Z",
                "to": end.isoformat() + "Z"
            }
            
            logger.info(f"Fetching {timeframe} candles for {instrument} from {start} to {end}")
            response_data = self._make_request(endpoint, params)
            
            if response_data and 'candles' in response_data:
                candles = response_data['candles']
                all_candles.extend(candles)
                logger.info(f"Fetched {len(candles)} candles")
                time.sleep(0.5)  # Small pause for rate limiting
            else:
                logger.error(f"Failed to fetch data for {instrument} from {start} to {end}")
                
        if not all_candles:
            return None
            
        return self._process_candles(all_candles)
    
    def fetch_multiple_instruments(
        self, 
        instruments: List[str], 
        timeframe: str = "H1",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        days_back: Optional[int] = None,
        parallel: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple instruments, optionally in parallel.
        
        Args:
            instruments: List of currency pairs to fetch
            timeframe: Candle timeframe (e.g., "H1" for hourly)
            start_time: Start time for data fetch
            end_time: End time for data fetch (defaults to now)
            days_back: Alternative to start_time, days to look back from end_time
            parallel: Whether to fetch in parallel (faster but higher API load)
            
        Returns:
            Dictionary of instrument -> DataFrame with price data
        """
        logger.info(f"Fetching data for {len(instruments)} instruments")
        
        # Calculate start and end times
        if end_time is None:
            end_time = datetime.now()
            
        if start_time is None and days_back is not None:
            start_time = end_time - timedelta(days=days_back)
        elif start_time is None:
            start_time = end_time - timedelta(days=180)  # Default to 180 days
        
        result_data = {}
        
        if parallel:
            # Fetch in parallel using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Create a dictionary of futures to instrument names
                futures = {
                    executor.submit(
                        self.get_historical_candles, 
                        instrument, 
                        start_time, 
                        end_time, 
                        timeframe
                    ): instrument 
                    for instrument in instruments
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    instrument = futures[future]
                    try:
                        df = future.result()
                        if df is not None and not df.empty:
                            result_data[instrument] = df
                            logger.info(f"Successfully fetched {len(df)} candles for {instrument}")
                        else:
                            logger.error(f"Failed to fetch data for {instrument}")
                    except Exception as e:
                        logger.error(f"Error fetching {instrument}: {str(e)}")
        else:
            # Fetch sequentially
            for instrument in instruments:
                try:
                    df = self.get_historical_candles(instrument, start_time, end_time, timeframe)
                    if df is not None and not df.empty:
                        result_data[instrument] = df
                        logger.info(f"Successfully fetched {len(df)} candles for {instrument}")
                    else:
                        logger.error(f"Failed to fetch data for {instrument}")
                except Exception as e:
                    logger.error(f"Error fetching {instrument}: {str(e)}")
        
        logger.info(f"Successfully fetched data for {len(result_data)} out of {len(instruments)} instruments")
        return result_data
    
    def fetch_for_cointegration(
        self,
        lookback_days: int = 180,
        timeframe: str = "H1",
        instruments: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data specifically for cointegration analysis.
        
        Args:
            lookback_days: Number of days to look back (default: 180)
            timeframe: Candle timeframe (default: "H1")
            instruments: Optional list of instruments to fetch (default: all configured instruments)
            
        Returns:
            Dictionary of instrument -> DataFrame with price data
        """
        if instruments is None:
            instruments = INSTRUMENTS
            
        logger.info(f"Fetching data for cointegration analysis: {len(instruments)} instruments, {lookback_days} days lookback")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        return self.fetch_multiple_instruments(
            instruments=instruments,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            parallel=True
        )
        
    def _calculate_date_ranges(
        self, 
        start_time: datetime,
        end_time: datetime
    ) -> List[tuple]:
        """Split date range into monthly chunks to avoid hitting API limits."""
        ranges = []
        current_date = start_time
        
        while current_date < end_time:
            # Move to next month or end date, whichever is sooner
            if current_date.month == 12:
                next_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                next_date = current_date.replace(month=current_date.month + 1)
                
            next_date = min(next_date, end_time)
            ranges.append((current_date, next_date))
            current_date = next_date
            
        return ranges
        
    def _process_candles(self, candles: List[Dict]) -> pd.DataFrame:
        """Convert candle data to DataFrame and process it."""
        processed_data = []
        
        for candle in candles:
            if candle.get('complete', True):  # Only use complete candles
                processed_data.append({
                    'timestamp': pd.to_datetime(candle['time']),
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                })
                
        df = pd.DataFrame(processed_data)
        
        if not df.empty:
            # Clean the dataframe
            df = df.sort_values('timestamp')
            df = df.drop_duplicates()
            
        return df
        
    def save_data_to_csv(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        instrument: Optional[str] = None,
        timeframe: str = "H1",
        output_dir: str = "data"
    ) -> Dict[str, str]:
        """
        Save data to CSV files.
        
        Args:
            data: DataFrame or dictionary of DataFrames to save
            instrument: Instrument name (required if data is a DataFrame)
            timeframe: Timeframe of the data
            output_dir: Directory to save the data
            
        Returns:
            Dictionary mapping instruments to saved file paths
        """
        Path(output_dir).mkdir(exist_ok=True)
        saved_files = {}
        
        if isinstance(data, pd.DataFrame):
            if instrument is None:
                raise ValueError("Instrument name is required when saving a single DataFrame")
                
            filename = f"{instrument}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(output_dir, filename)
            
            data.to_csv(filepath, index=False)
            logger.info(f"Saved {len(data)} candles to {filepath}")
            saved_files[instrument] = filepath
        else:
            # Handle dictionary of DataFrames
            for instr, df in data.items():
                filename = f"{instr}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
                filepath = os.path.join(output_dir, filename)
                
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {len(df)} candles to {filepath}")
                saved_files[instr] = filepath
                
        return saved_files


def main():
    """
    Example usage of the enhanced OandaDataFetcher for statistical arbitrage.
    Demonstrates getting data for multiple instruments and preparing for cointegration analysis.
    """
    fetcher = OandaDataFetcher()
    
    # Example 1: Fetch data for cointegration analysis
    instruments = [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CAD", 
        "AUD_USD", "NZD_USD", "EUR_GBP", "EUR_JPY"
    ]
    
    timeframe = "H1"
    lookback_days = 180
    
    print(f"Fetching data for {len(instruments)} instruments over {lookback_days} days...")
    pair_data = fetcher.fetch_for_cointegration(
        lookback_days=lookback_days,
        timeframe=timeframe,
        instruments=instruments
    )
    
    print(f"Successfully fetched data for {len(pair_data)} instruments")
    
    # Example 2: Get latest data for a specific instrument
    instrument = "EUR_USD"
    latest_df = fetcher.get_latest_candles(instrument, count=100)
    
    if latest_df is not None:
        print(f"\nLatest {instrument} data:")
        print(latest_df.head())
    
    # Example 3: Save all data to CSV files
    saved_files = fetcher.save_data_to_csv(pair_data, timeframe=timeframe)
    print(f"\nSaved data to {len(saved_files)} CSV files")
    
    return pair_data  # Return data for further analysis


if __name__ == "__main__":
    main()