import os
import dask.dataframe as dd
import logging

def load_and_preprocess(parquet_path, start_date='2023-01-01', end_date='2023-12-31', filepath=None):
    """Load and preprocess data from Parquet files if available, otherwise from CSV."""
    if os.path.exists(parquet_path):
        ddf = dd.read_parquet(parquet_path)
        logging.info(f"Loaded existing Parquet data from {parquet_path}")
        # Check and add 'mid' if missing
        if 'mid' not in ddf.columns and 'BID' in ddf.columns and 'ASK' in ddf.columns:
            ddf['mid'] = ((ddf.BID + ddf.ASK) / 2).astype('float32')
            logging.info("Computed missing 'mid' column from BID and ASK")
    else:
        if filepath is None or not os.path.exists(filepath):
            raise FileNotFoundError(f"Parquet file not found at {parquet_path} and no valid CSV filepath provided")
        # Load from CSV
        usecols = ['DATE', 'TIME_M', 'BID', 'BIDSIZ', 'ASK', 'ASKSIZ', 'SYM_ROOT', 'SYM_SUFFIX']
        ddf = dd.read_csv(
            filepath,
            usecols=usecols,
            dtype={
                'DATE': 'object', 'TIME_M': 'object',
                'BID': 'float32', 'BIDSIZ': 'int32',
                'ASK': 'float32', 'ASKSIZ': 'int32',
                'SYM_ROOT': 'object', 'SYM_SUFFIX': 'object'
            },
            blocksize="64MB",
            assume_missing=True
        )
        # Preprocess: create datetime column and compute mid
        dates1 = dd.to_datetime(ddf['DATE'], format='%m/%d/%Y', errors='coerce')
        dates2 = dd.to_datetime(ddf['DATE'], format='%Y-%m-%d', errors='coerce')
        ddf['date'] = dates1.fillna(dates2)
        times = ddf['TIME_M'].str.extract(r'(?P<min>\d+):(?P<sec>\d+\.\d+)')
        ddf['minutes'] = dd.to_numeric(times['min'], errors='coerce').fillna(0).astype('int32')
        ddf['seconds'] = dd.to_numeric(times['sec'], errors='coerce').fillna(0).astype('float32')
        ddf['datetime'] = ddf['date'] + dd.to_timedelta(ddf['minutes'] * 60 + ddf['seconds'], unit='s')
        ddf['mid'] = ((ddf.BID + ddf.ASK) / 2).astype('float32')
        ddf = (
            ddf
            .drop(columns=['DATE', 'TIME_M', 'date', 'minutes', 'seconds'])
            .dropna(subset=['BID', 'ASK'])
            .loc[lambda df: (df.BID > 0) & (df.ASK > 0)]
        )
        # Save preprocessed data to Parquet
        ddf.to_parquet(parquet_path, write_index=False)
        logging.info(f"Preprocessed CSV and saved to {parquet_path}")

    # Filter by date range
    ddf = ddf[(ddf.datetime >= start_date) & (ddf.datetime <= end_date)]
    return ddf.persist()