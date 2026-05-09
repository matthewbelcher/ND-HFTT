#!/usr/bin/env python3
import numpy as np
import pandas as pd
import time
import argparse
from datetime import datetime
import os
import sqlite3
import psutil
import concurrent.futures
import multiprocessing as mp
from collections import defaultdict
import json
import pyarrow as pa
import pyarrow.parquet as pq
import bisect
from tqdm import tqdm
import numba
import gc
import signal
import sys
import random
from multiprocessing import Pool
from functools import partial
import warnings
import mmap

# Silence warnings that we can't do anything about
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Global memory settings
memory_limit = int(psutil.virtual_memory().total * 0.8)  # Use up to 80% of system RAM
print(f"Memory limit set to {memory_limit / (1024**3):.1f} GB")

# Configure Numba for performance
numba.config.THREADING_LAYER = 'threadsafe'
numba.set_num_threads(min(mp.cpu_count(), 16))

# Numba-accelerated calculations
@numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_metrics_fast(best_bid, best_ask, bid_sizes, ask_sizes):
    """Numba-optimized metrics calculation with parallel processing"""
    spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0
    mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0
    
    bid_depth = np.sum(bid_sizes)
    ask_depth = np.sum(ask_sizes)
    
    imbalance = bid_depth / ask_depth if ask_depth > 0 else float('inf')
    
    total_depth = bid_depth + ask_depth
    if total_depth > 0 and best_bid > 0 and best_ask > 0:
        weighted_mid = (best_bid * ask_depth + best_ask * bid_depth) / total_depth
    else:
        weighted_mid = mid_price
    
    return spread, mid_price, bid_depth, ask_depth, imbalance, weighted_mid

@numba.jit(nopython=True, fastmath=True, cache=True)
def find_relevant_fills(fills_timestamps, timestamps, window_size):
    """
    Optimized binary search to find relevant fill events for each timestamp
    
    Args:
        fills_timestamps: Sorted array of fill timestamps
        timestamps: Array of target timestamps to process
        window_size: Window size for searching around each timestamp
        
    Returns:
        List of (start_idx, end_idx) for each timestamp
    """
    results = np.zeros((len(timestamps), 2), dtype=np.int32)
    
    for i, timestamp in enumerate(timestamps):
        idx = np.searchsorted(fills_timestamps, timestamp)
        
        # Handle edge cases
        if idx == len(fills_timestamps):
            idx = len(fills_timestamps) - 1
        
        # Find the window around the index
        start_idx = max(0, idx - window_size)
        end_idx = min(len(fills_timestamps), idx + window_size)
        
        results[i, 0] = start_idx
        results[i, 1] = end_idx
        
    return results

def connect_db(db_file):
    """Connect to SQLite database with extreme performance optimization"""
    conn = sqlite3.connect(db_file, isolation_level=None)  # Autocommit mode
    
    # Extreme performance settings - trade safety for speed
    conn.execute("PRAGMA journal_mode=OFF")  # Disable journaling completely
    conn.execute("PRAGMA synchronous=OFF")   # No syncing to disk
    conn.execute("PRAGMA cache_size=-500000") # 500MB cache
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=30000000000") # Use memory mapping for file access
    conn.execute("PRAGMA page_size=16384")   # Larger pages for sequential reads
    conn.execute("PRAGMA locking_mode=EXCLUSIVE") # Exclusive access
    conn.execute("PRAGMA threads=8")         # Use multiple threads
    
    conn.row_factory = sqlite3.Row
    return conn

def batch_process_timestamps(args):
    """
    Process a batch of timestamps with vectorized operations
    
    Args:
        args: Tuple of (fills_array, timestamps_array)
        
    Returns:
        Array of metrics for all timestamps
    """
    fills_array, timestamps_batch = args
    
    if len(fills_array) == 0 or len(timestamps_batch) == 0:
        return []
    
    # Extract timestamps from fills for faster searching
    fills_timestamps = np.array([f['timestamp'] for f in fills_array])
    
    # Find relevant fill indices for all timestamps in batch using optimized search
    window_size = min(100, max(10, int(len(fills_array) * 0.01)))
    idx_ranges = find_relevant_fills(fills_timestamps, timestamps_batch, window_size)
    
    results = []
    
    # Process each timestamp
    for i, timestamp in enumerate(timestamps_batch):
        start_idx, end_idx = idx_ranges[i]
        relevant_fills = fills_array[start_idx:end_idx]
        
        # Filter fills before the timestamp
        prev_fills = [f for f in relevant_fills if f['timestamp'] <= timestamp]
        
        if not prev_fills:
            # Use default values if no previous fills
            result = {
                'timestamp': timestamp,
                'mid_price': 0.5,
                'best_bid': 0.495,
                'best_ask': 0.505,
                'spread': 0.01,
                'bid_depth': 50,
                'ask_depth': 50,
                'imbalance': 1.0,
                'weighted_mid': 0.5
            }
        else:
            # Process with existing logic but more optimized
            if len(prev_fills) >= 3:
                # Extract recent prices and sizes efficiently
                recent_idx = max(0, len(prev_fills) - 20)
                recent_fills = prev_fills[recent_idx:]
                
                # Use NumPy for faster calculations
                recent_prices = np.array([f['price'] for f in recent_fills])
                recent_sizes = np.array([f['size'] for f in recent_fills])
                
                price_volatility = np.std(recent_prices) if len(recent_prices) > 1 else 0
                avg_size = np.mean(recent_sizes)
                
                # Determine price trend
                price_trend = recent_prices[-1] - recent_prices[0] if len(recent_prices) > 1 else 0
                
                # Use the most recent price as mid price
                mid_price = recent_prices[-1]
                
                # Scale book depth based on market activity
                activity_factor = min(5, max(1, np.log10(len(prev_fills) + 1)))
                base_size = max(avg_size * activity_factor * 10, 20)
                
                # Scale spread by volatility
                spread_factor = max(0.001, 0.003 * (1 + price_volatility / max(0.001, mid_price)))
                
                # Add trend bias to the prices
                trend_bias = 0.0005 * np.sign(price_trend) * min(10, abs(price_trend) / max(0.01, mid_price))
                
                # Calculate bid/ask prices with trend bias
                best_bid = round(mid_price * (1 - spread_factor + trend_bias), 6)
                best_ask = round(mid_price * (1 + spread_factor - trend_bias), 6)
                
                # Actual spread
                spread = best_ask - best_bid
                
                # Size with trend bias
                bid_size = base_size
                ask_size = base_size
                
                if price_trend > 0:
                    bid_size *= 1.3  # More buying pressure
                elif price_trend < 0:
                    ask_size *= 1.3  # More selling pressure
                
                # Calculate actual depths
                decay_factor = 0.9
                level_multipliers = np.array([1, decay_factor, decay_factor**2, decay_factor**3, decay_factor**4])
                bid_depth = bid_size * np.sum(level_multipliers)
                ask_depth = ask_size * np.sum(level_multipliers)
                
                # Calculate imbalance
                imbalance = bid_depth / ask_depth if ask_depth > 0 else float('inf')
                
                # Calculate weighted mid
                weighted_mid = (best_bid * ask_depth + best_ask * bid_depth) / (bid_depth + ask_depth)
            else:
                # Simple metrics based on limited data
                last_price = prev_fills[-1]['price']
                last_size = prev_fills[-1]['size']
                
                # Use simple spread
                best_bid = round(last_price * 0.995, 6)
                best_ask = round(last_price * 1.005, 6)
                
                spread = best_ask - best_bid
                mid_price = last_price
                bid_depth = last_size * 5
                ask_depth = last_size * 5
                imbalance = 1.0
                weighted_mid = last_price
                
            result = {
                'timestamp': timestamp,
                'mid_price': mid_price,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'imbalance': imbalance,
                'weighted_mid': weighted_mid
            }
        
        results.append(result)
    
    return results

def process_timestamp_chunk(args):
    """
    Process a chunk of timestamps in one process
    
    Args:
        args: Tuple of (fills_array, timestamps_chunk)
        
    Returns:
        List of orderbook metrics
    """
    # Use the batch processor for better performance
    return batch_process_timestamps(args)

class UltraFastOrderbookReconstructor:
    """
    Ultra-optimized implementation for reconstructing orderbooks from millions of events
    in less than an hour.
    """
    
    def __init__(self, db_file="polymarket_orderbook.db", output_dir="orderbook_data"):
        """Initialize with the path to the SQLite database and output directory"""
        self.db_file = db_file
        self.output_dir = output_dir
        self.candidates = ["Trump", "Harris"]
        # Optimize worker count based on CPU count
        self.max_workers = min(mp.cpu_count(), 8)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        
        # Use smaller chunks to process data
        self.chunk_size = 5000  # Reduce from 10k to 5k records at a time
        
        # Orderbook reconstruction settings
        self.next_order_id = 0
        self.depth_levels = 10  # Number of price levels to maintain in orderbook
        
        # Activity thresholds (events per second)
        self.high_activity_threshold = 50  # More than 50 events/sec = high activity
        
        # Performance monitoring
        self.start_time = None
        self.processed_events = 0
        
        # Maximum events to keep in memory at once
        self.max_events_in_memory = 200000  # Increased for better batching
        
        # Initialize database - create indexes for faster queries
        self._initialize_database()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)
    
    def _initialize_database(self):
        """Create indexes and optimize the database for faster queries"""
        try:
            conn = self.connect_db()
            cursor = conn.cursor()
            
            # Create indexes for common queries
            print("Creating database indexes for faster queries...")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_candidate_timestamp ON order_filled_events (candidate, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_full ON order_filled_events (candidate, timestamp, maker_asset_id, taker_asset_id)")
            
            # Analyze tables for query optimization
            cursor.execute("ANALYZE")
            conn.close()
            print("Database indexes created and optimized")
        except Exception as e:
            print(f"Warning: Could not create database indexes: {e}")
    
    def handle_interrupt(self, signum, frame):
        """Handle keyboard interrupts gracefully"""
        # Only respond to SIGINT (Ctrl+C) and SIGTERM
        if signum in (signal.SIGINT, signal.SIGTERM):
            print("\nReceived interrupt signal. Cleaning up and saving progress...")
            sys.exit(0)
    
    def connect_db(self):
        """Connect to SQLite database with optimized settings for faster queries"""
        return connect_db(self.db_file)
    
    def generate_order_id(self):
        """Generate a unique order ID"""
        self.next_order_id += 1
        return f"o{self.next_order_id}"
    
    def get_time_range_for_candidate(self, candidate):
        """
        Get the full time range for a candidate's events
        
        Args:
            candidate: Candidate name
            
        Returns:
            Tuple of (start_time, end_time)
        """
        conn = self.connect_db()
        cursor = conn.cursor()
        
        # Check if reconstructed_orderbook table exists
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='reconstructed_orderbook'")
        has_reconstructed = cursor.fetchone()[0] > 0
        
        if has_reconstructed:
            cursor.execute("""
                SELECT MIN(timestamp), MAX(timestamp)
                FROM reconstructed_orderbook
                WHERE candidate = ?
            """, (candidate,))
        else:
            cursor.execute("""
                SELECT MIN(timestamp), MAX(timestamp)
                FROM order_filled_events
                WHERE candidate = ?
            """, (candidate,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] is not None:
            return (result[0], result[1])
        else:
            return (0, 0)  # No data found
    
    def count_events_for_candidate(self, candidate):
        """Count the total number of events for a candidate"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        # First check if reconstructed_orderbook table exists
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='reconstructed_orderbook'")
        has_reconstructed = cursor.fetchone()[0] > 0
        
        if has_reconstructed:
            cursor.execute("""
                SELECT COUNT(*) 
                FROM reconstructed_orderbook 
                WHERE candidate = ?
            """, (candidate,))
        else:
            cursor.execute("""
                SELECT COUNT(*) 
                FROM order_filled_events 
                WHERE candidate = ?
            """, (candidate,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def stream_fill_events(self, candidate, start_time=None, end_time=None):
        """
        Optimized streaming of fill events with better SQL query performance
        
        Args:
            candidate: Candidate to load data for
            start_time: Optional start timestamp
            end_time: Optional end timestamp
            
        Yields:
            Lists of fill events in chunks
        """
        conn = self.connect_db()
        
        try:
            cursor = conn.cursor()
            
            # Optimize SQL query with direct filtering
            params = [candidate]
            conditions = ["candidate = ?"]
            
            if start_time is not None:
                conditions.append("timestamp >= ?")
                params.append(int(start_time))
            
            if end_time is not None:
                conditions.append("timestamp <= ?")
                params.append(int(end_time))
            
            where_clause = " AND ".join(conditions)
            
            # Count total rows to process
            count_query = f"SELECT COUNT(*) FROM order_filled_events WHERE {where_clause}"
            cursor.execute(count_query, params)
            total_rows = cursor.fetchone()[0]
            
            # If small enough, load all at once for better performance
            if total_rows < self.max_events_in_memory:
                query = f"""
                    SELECT 
                        timestamp, maker, maker_asset_id, taker_asset_id, 
                        maker_amount_filled, taker_amount_filled, id
                    FROM order_filled_events
                    WHERE {where_clause}
                    ORDER BY timestamp ASC
                """
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Process all rows at once
                chunk_events = []
                for row in rows:
                    timestamp, maker, maker_asset_id, taker_asset_id, maker_amount, taker_amount, event_id = row
                    
                    # Calculate price with error handling
                    try:
                        price = float(taker_amount) / float(maker_amount) if float(maker_amount) > 0 else 0
                    except (ValueError, TypeError):
                        price = 0
                    
                    # Default to BUY for simplicity
                    side = "BUY"
                    
                    # Convert amounts from raw units
                    size = float(maker_amount) / 1e18 if maker_amount else 0
                    
                    event = {
                        'timestamp': int(timestamp),
                        'price': price,
                        'side': side,
                        'size': size,
                        'maker': maker,
                        'asset_id': maker_asset_id,
                        'fill_id': f"{timestamp}_{maker_asset_id}_{side}",
                        'candidate': candidate
                    }
                    
                    chunk_events.append(event)
                
                yield chunk_events
            else:
                # For larger datasets, stream in optimized chunks
                offset = 0
                optimized_chunk_size = min(self.max_events_in_memory, max(self.chunk_size, total_rows // 10))
                
                while True:
                    query = f"""
                        SELECT 
                            timestamp, maker, maker_asset_id, taker_asset_id, 
                            maker_amount_filled, taker_amount_filled, id
                        FROM order_filled_events
                        WHERE {where_clause}
                        ORDER BY timestamp ASC
                        LIMIT {optimized_chunk_size} OFFSET {offset}
                    """
                    
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    if not rows:
                        break
                    
                    chunk_events = []
                    for row in rows:
                        timestamp, maker, maker_asset_id, taker_asset_id, maker_amount, taker_amount, event_id = row
                        
                        # Calculate price with error handling
                        try:
                            price = float(taker_amount) / float(maker_amount) if float(maker_amount) > 0 else 0
                        except (ValueError, TypeError):
                            price = 0
                        
                        # Default to BUY for simplicity
                        side = "BUY"
                        
                        # Convert amounts from raw units
                        size = float(maker_amount) / 1e18 if maker_amount else 0
                        
                        event = {
                            'timestamp': int(timestamp),
                            'price': price,
                            'side': side,
                            'size': size,
                            'maker': maker,
                            'asset_id': maker_asset_id,
                            'fill_id': f"{timestamp}_{maker_asset_id}_{side}",
                            'candidate': candidate
                        }
                        
                        chunk_events.append(event)
                    
                    yield chunk_events
                    offset += optimized_chunk_size
        
        except Exception as e:
            print(f"Error in stream_fill_events: {str(e)}")
            yield []
        
        finally:
            conn.close()
    
    def get_all_event_timestamps(self, candidate, start_time=None, end_time=None, count_only=False):
        """
        Ultra-optimized version of get_all_event_timestamps with SQL query optimization
        and memory mapping for large result sets
        """
        conn = self.connect_db()
        
        try:
            # Create temporary index for faster querying if it doesn't exist
            cursor = conn.cursor()
            
            # Prepare query with parameters and filter in SQL where possible
            params = [candidate]
            conditions = ["candidate = ?"]
            
            if start_time is not None:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            
            if end_time is not None:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            where_clause = " AND ".join(conditions)
            
            if count_only:
                # Just get count for faster operation
                query = f"SELECT COUNT(*) FROM order_filled_events WHERE {where_clause}"
                cursor.execute(query, params)
                result = cursor.fetchone()[0]
            else:
                # For better performance, use direct query with LIMIT/OFFSET in batches
                # to avoid loading everything into memory at once
                total_query = f"SELECT COUNT(*) FROM order_filled_events WHERE {where_clause}"
                cursor.execute(total_query, params)
                total_count = cursor.fetchone()[0]
                
                # If result set is very large, use memory-mapped array
                if total_count > 1000000:  # 1M+ timestamps
                    print(f"Large result set detected ({total_count:,} timestamps), using memory mapping")
                    # Create memory mapped temp file for results
                    mm_filename = f"temp_{candidate}_{int(time.time())}.dat"
                    mm_array = np.memmap(mm_filename, dtype=np.int64, mode='w+', shape=(total_count,))
                    
                    # Load in chunks to avoid memory issues
                    batch_size = 500000
                    loaded = 0
                    
                    while loaded < total_count:
                        batch_query = f"""
                            SELECT timestamp FROM order_filled_events 
                            WHERE {where_clause} 
                            ORDER BY timestamp
                            LIMIT {batch_size} OFFSET {loaded}
                        """
                        cursor.execute(batch_query, params)
                        batch_data = np.array([row[0] for row in cursor.fetchall()], dtype=np.int64)
                        mm_array[loaded:loaded+len(batch_data)] = batch_data
                        loaded += len(batch_data)
                        print(f"Loaded {loaded:,}/{total_count:,} timestamps ({(loaded/total_count)*100:.1f}%)")
                    
                    # Flush to disk and return the memory-mapped array
                    mm_array.flush()
                    result = mm_array
                    
                    # Register for cleanup
                    self._temp_files = getattr(self, '_temp_files', [])
                    self._temp_files.append(mm_filename)
                else:
                    # For smaller result sets, use standard approach
                    query = f"SELECT timestamp FROM order_filled_events WHERE {where_clause} ORDER BY timestamp"
                    cursor.execute(query, params)
                    result = np.array([row[0] for row in cursor.fetchall()], dtype=np.int64)
            
            return result
        
        except Exception as e:
            print(f"Error in get_all_event_timestamps: {str(e)}")
            import traceback
            traceback.print_exc()
            return [] if not count_only else 0
        
        finally:
            conn.close()
    
    def analyze_activity_by_time_windows(self, candidate, time_range=None):
        """
        Analyze activity levels by time windows to determine sampling strategy
        
        Args:
            candidate: Candidate to analyze
            time_range: Optional tuple of (start_time, end_time)
            
        Returns:
            List of activity segments
        """
        print(f"Analyzing activity patterns for {candidate}...")
        
        # Get time range for candidate
        if time_range:
            start_time, end_time = time_range
        else:
            start_time, end_time = self.get_time_range_for_candidate(candidate)
        
        if start_time == 0 and end_time == 0:
            print(f"No data found for {candidate}")
            return []
        
        print(f"Time range: {datetime.fromtimestamp(start_time)} to {datetime.fromtimestamp(end_time)}")
        
        # Analyze in windows of 1 minute (60 seconds)
        window_size = 60
        segments = []
        window_counts = {}
        
        # Stream data and count events per window
        for chunk in self.stream_fill_events(candidate, start_time, end_time):
            for event in chunk:
                ts = event['timestamp']
                window_start = (ts // window_size) * window_size
                
                if window_start not in window_counts:
                    window_counts[window_start] = 0
                window_counts[window_start] += 1
        
        # Convert window counts to segments
        for window_start, count in window_counts.items():
            events_per_second = count / window_size
            is_high_activity = events_per_second >= self.high_activity_threshold
            
            segments.append({
                'start': window_start,
                'end': window_start + window_size,
                'events': count,
                'events_per_second': events_per_second,
                'is_high_activity': is_high_activity
            })
        
        # Sort segments by start time
        segments.sort(key=lambda x: x['start'])
        
        # Statistics on segments
        high_activity_segments = [s for s in segments if s['is_high_activity']]
        print(f"Found {len(segments)} segments, {len(high_activity_segments)} high activity")
        
        if high_activity_segments:
            avg_high_activity = sum(s['events_per_second'] for s in high_activity_segments) / len(high_activity_segments)
            print(f"Average events/sec in high activity periods: {avg_high_activity:.2f}")
        
        return segments
    
    def process_candidate_in_parallel(self, candidate, timestamps, fills, batch_size=2000):
        """
        Process a candidate's timestamps in parallel using multiprocessing
        
        Args:
            candidate: Candidate name
            timestamps: List of timestamps to process
            fills: List of fill events
            batch_size: Batch size for parallel processing
            
        Returns:
            List of orderbook metrics
        """
        # Sort fills by timestamp for binary search
        fills.sort(key=lambda x: x['timestamp'])
        fills_array = np.array(fills)
        
        # Split timestamps into batches
        timestamp_batches = [timestamps[i:i+batch_size] for i in range(0, len(timestamps), batch_size)]
        
        all_metrics = []
        
        # Process each batch in parallel
        with mp.Pool(processes=self.max_workers) as pool:
            args = [(fills_array, batch) for batch in timestamp_batches]
            
            # Map function across the pool
            results = list(tqdm(
                pool.imap(process_timestamp_chunk, args),
                total=len(timestamp_batches),
                desc=f"Processing {len(timestamps)} timestamps for {candidate}"
            ))
            
            # Flatten results
            for batch_result in results:
                all_metrics.extend(batch_result)
        
        return all_metrics
    
    def process_candidate_direct(self, candidate, num_processes=None, max_sample=None):
        """
        Hyper-optimized processing with smarter chunking and vectorized operations
        
        Args:
            candidate: Candidate name
            num_processes: Number of processes to use (defaults to system setting)
            max_sample: Maximum number of samples to process
            
        Returns:
            Dict with metrics results
        """
        # Use provided process count or default to system setting
        if num_processes is None:
            num_processes = min(mp.cpu_count(), 12)  # Allow more processes for high-end systems
        
        print(f"\nProcessing {candidate} with hyper-optimized processing (workers: {num_processes})")
        
        # Optimize process pool
        mp.set_start_method('spawn', force=True)
        
        # Get time range for the candidate (cached for performance)
        start_time, end_time = self.get_time_range_for_candidate(candidate)
        if start_time == 0 and end_time == 0:
            print(f"No data found for candidate {candidate}")
            return {}
        
        # Convert to datetime for easier manipulation
        start_date = datetime.fromtimestamp(start_time)
        end_date = datetime.fromtimestamp(end_time)
        
        print(f"Full time range: {start_date} to {end_date}")
        
        # Get total number of timestamps to process
        total_timestamps = self.get_all_event_timestamps(candidate, start_time, end_time, count_only=True)
        print(f"Total timestamps to process: {total_timestamps:,}")
        
        # If max_sample is specified, we'll sample timestamps
        if max_sample and max_sample < total_timestamps:
            print(f"Sampling {max_sample:,} timestamps out of {total_timestamps:,}")
            
            # Get all timestamps for sampling using optimized method
            all_timestamps = self.get_all_event_timestamps(candidate, start_time, end_time)
            
            # Random sample with a fixed seed for reproducibility
            np.random.seed(42)
            if isinstance(all_timestamps, np.ndarray):
                # Handle NumPy array case with optimized sampling
                if len(all_timestamps) > 10000000:  # For extremely large arrays
                    # Stratified sampling - take evenly spaced samples
                    indices = np.linspace(0, len(all_timestamps)-1, max_sample, dtype=int)
                    sampled_timestamps = all_timestamps[indices]
                else:
                    # Random sampling for normal sized arrays
                    sampled_indices = np.random.choice(len(all_timestamps), max_sample, replace=False)
                    sampled_timestamps = all_timestamps[sampled_indices]
                    sampled_timestamps.sort()
            else:
                # Handle list case
                sampled_timestamps = sorted(random.sample(all_timestamps, max_sample))
            
            # Process the sampled timestamps with optimized batch size and chunking
            batch_size = max(100, min(5000, len(sampled_timestamps) // (num_processes * 2)))
            
            # Load all fills for context - more efficient for small samples
            all_fills = []
            buffer = 86400 * 2  # 2 day buffer
            for fills_chunk in self.stream_fill_events(candidate, start_time - buffer, end_time + buffer):
                all_fills.extend(fills_chunk)
            
            print(f"Loaded {len(all_fills):,} fill events for context")
            
            # Sort fills for faster lookup
            all_fills.sort(key=lambda x: x['timestamp'])
            fills_array = np.array(all_fills)
            
            # Process in optimized batches
            timestamp_batches = np.array_split(sampled_timestamps, 
                                               max(1, min(num_processes * 4, len(sampled_timestamps) // batch_size)))
            
            # Process each batch in parallel with optimized workload
            with mp.Pool(processes=num_processes) as pool:
                args = [(fills_array, batch) for batch in timestamp_batches]
                results_list = list(tqdm(
                    pool.imap(batch_process_timestamps, args),
                    total=len(timestamp_batches),
                    desc=f"Processing {len(sampled_timestamps):,} sampled timestamps"
                ))
                
                # Flatten results efficiently
                metrics_list = [item for sublist in results_list for item in sublist]
            
            # Save and return results
            return self._save_and_summarize_metrics(metrics_list, candidate, f"{candidate}_sampled_{max_sample}")
        
        # Optimize chunking strategy based on data size and memory constraints
        
        # 1. For very large datasets, use adaptive chunking
        if total_timestamps > 5000000:  # 5M+ timestamps
            # Use adaptive chunk sizes - weekly for most, daily for recent
            
            # Define election month - use October 2024
            election_month_start = int(datetime(2024, 10, 1).timestamp())
            
            # Create chunks with adaptive sizing
            chunks = []
            
            # First chunk the data by quarters for historical data (3-month chunks)
            current_date = start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            while current_date < end_date:
                # If we're in the intense period (last 3 months before election), break out
                if current_date.timestamp() >= election_month_start - 7776000:  # 90 days before election
                    break
                    
                # Calculate next quarter (3 months)
                month_offset = 3
                if current_date.month <= 9:
                    next_date = datetime(current_date.year, current_date.month + month_offset, 1)
                else:
                    next_year = current_date.year + 1
                    next_month = (current_date.month + month_offset) % 12
                    if next_month == 0:
                        next_month = 12
                    next_date = datetime(next_year, next_month, 1)
                    
                # Don't go past the 3-month election cutoff
                if next_date.timestamp() > election_month_start - 7776000:
                    next_date = datetime.fromtimestamp(election_month_start - 7776000)
                    
                # Add the quarterly chunk
                chunks.append((int(current_date.timestamp()), int(next_date.timestamp())))
                current_date = next_date
            
            # Next, process the pre-election period (3 months before) with monthly chunks
            while current_date.timestamp() < election_month_start:
                # Calculate next month
                if current_date.month == 12:
                    next_date = datetime(current_date.year + 1, 1, 1)
                else:
                    next_date = datetime(current_date.year, current_date.month + 1, 1)
                    
                # Add the monthly chunk
                chunks.append((int(current_date.timestamp()), int(next_date.timestamp())))
                current_date = next_date
            
            # Finally, process election month and after with daily chunks
            while current_date.timestamp() < end_date.timestamp():
                next_date = current_date + pd.Timedelta(days=1)
                next_date = min(next_date, end_date)
                
                chunks.append((int(current_date.timestamp()), int(next_date.timestamp())))
                current_date = next_date
        else:
            # For smaller datasets, use simpler monthly/daily chunking
            
            # Define election month - use October 2024
            election_month_start = int(datetime(2024, 10, 1).timestamp())
            
            # Create our chunks - monthly for historical, daily for election month
            chunks = []
            
            # Process historical data in monthly chunks
            current_date = start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            while current_date < end_date:
                # If we're in the election month or later, break out to do daily processing
                if current_date.timestamp() >= election_month_start:
                    break
                    
                # Calculate next month
                if current_date.month == 12:
                    next_date = datetime(current_date.year + 1, 1, 1)
                else:
                    next_date = datetime(current_date.year, current_date.month + 1, 1)
                    
                # Don't go past the end date
                next_date = min(next_date, end_date)
                
                # Add the monthly chunk
                chunks.append((int(current_date.timestamp()), int(next_date.timestamp())))
                current_date = next_date
            
            # Process election month and beyond in daily chunks
            if current_date.timestamp() < end_date.timestamp():
                # Start from where monthly processing left off
                daily_start = int(current_date.timestamp())
                
                # Process daily chunks
                current_time = daily_start
                while current_time < end_time:
                    next_time = min(current_time + 86400, end_time)  # 86400 = 1 day in seconds
                    chunks.append((current_time, next_time))
                    current_time = next_time
        
        # Log chunking strategy
        quarterly_chunks = len([c for c in chunks if c[1]-c[0] > 86400*90])
        monthly_chunks = len([c for c in chunks if 86400*25 < c[1]-c[0] <= 86400*90])
        daily_chunks = len([c for c in chunks if c[1]-c[0] <= 86400])
        
        print(f"Optimized chunking strategy: {len(chunks)} chunks "
              f"({quarterly_chunks} quarterly, {monthly_chunks} monthly, {daily_chunks} daily)")
        
        # Initialize metrics storage with optimized structure
        all_metrics_combined = {}
        total_processed = 0
        
        # Process each chunk with optimized methods
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            # Get readable dates
            start_date_str = datetime.fromtimestamp(chunk_start).strftime('%Y-%m-%d')
            end_date_str = datetime.fromtimestamp(chunk_end).strftime('%Y-%m-%d')
            
            # Determine chunk type for reporting
            chunk_span = chunk_end - chunk_start
            if chunk_span > 86400*90:
                chunk_type = "quarterly"
            elif chunk_span > 86400:
                chunk_type = "monthly"
            else:
                chunk_type = "daily"
            
            print(f"\nProcessing {chunk_type} chunk {chunk_idx+1}/{len(chunks)}: {start_date_str} to {end_date_str}")
            
            # Get timestamps for this chunk with optimized query
            chunk_timestamps = self.get_all_event_timestamps(candidate, chunk_start, chunk_end)
            
            # Check if array is empty - handle NumPy array case
            if isinstance(chunk_timestamps, np.ndarray):
                has_timestamps = len(chunk_timestamps) > 0
            else:
                has_timestamps = bool(chunk_timestamps)
                
            if not has_timestamps:
                print(f"No timestamps in this chunk, skipping...")
                continue
            
            # Progress indicator
            print(f"Processing {len(chunk_timestamps):,} timestamps in this chunk")
            
            # Use optimal batch size based on data size and system memory
            batch_size = max(50, min(5000, len(chunk_timestamps) // (num_processes * 4)))
            
            # Create a buffer window around the chunk for context - larger for longer spans
            buffer = min(86400 * 7, max(3600 * 3, chunk_span // 10))  # Adaptive buffer
            load_start = chunk_start - buffer
            load_end = chunk_end + buffer
            
            # Load all fills for this chunk with buffer to avoid repeated database access
            all_fills = []
            for fills_chunk in self.stream_fill_events(candidate, load_start, load_end):
                all_fills.extend(fills_chunk)
            
            if not all_fills:
                print(f"No fill events found for this time range, skipping...")
                continue
                
            print(f"Loaded {len(all_fills):,} fill events for context")
            
            # Sort fills for faster lookup
            all_fills.sort(key=lambda x: x['timestamp'])
            fills_array = np.array(all_fills)
            
            # Process timestamps in optimized batches - use array split for more even distribution
            timestamp_batches = np.array_split(chunk_timestamps, 
                                              max(1, min(num_processes * 4, len(chunk_timestamps) // batch_size)))
            
            # Process each batch in parallel with optimized workload
            chunk_metrics = []
            with mp.Pool(processes=num_processes) as pool:
                args = [(fills_array, batch) for batch in timestamp_batches]
                results = list(tqdm(
                    pool.imap(batch_process_timestamps, args),
                    total=len(timestamp_batches),
                    desc=f"Processing batches"
                ))
                
                # Combine results efficiently
                for batch_result in results:
                    chunk_metrics.extend(batch_result)
            
            # Convert to combined format with optimized method
            chunk_metrics_dict = self.combine_metrics(chunk_metrics)
            
            # Combine with overall metrics efficiently
            if not all_metrics_combined:
                all_metrics_combined = chunk_metrics_dict
            else:
                # Merge efficiently using vectorized operations where possible
                for key, values in chunk_metrics_dict.items():
                    if key in all_metrics_combined:
                        # If values are numeric arrays, use NumPy concatenate
                        if isinstance(values, np.ndarray) and isinstance(all_metrics_combined[key], np.ndarray):
                            all_metrics_combined[key] = np.concatenate([all_metrics_combined[key], values])
                        else:
                            all_metrics_combined[key].extend(values)
                    else:
                        all_metrics_combined[key] = values
            
            # Update progress counter
            total_processed += len(chunk_timestamps)
            print(f"Total processed so far: {total_processed:,}/{total_timestamps:,} ({(total_processed/total_timestamps)*100:.2f}%)")
            
            # Free memory explicitly
            del chunk_timestamps
            del all_fills
            del fills_array
            del chunk_metrics
            del chunk_metrics_dict
            gc.collect()
            
            # Save intermediate results with adaptive frequency
            save_frequency = max(1, min(10, len(chunks) // 10))  # Save ~10 checkpoints total
            if chunk_idx % save_frequency == 0 or chunk_type != "daily":
                self._save_intermediate_metrics(all_metrics_combined, candidate, f"checkpoint_{chunk_idx}")
        
        # Save the final combined metrics
        return self._save_and_summarize_metrics(all_metrics_combined, candidate)

    def _save_intermediate_metrics(self, metrics, candidate, suffix=""):
        """Optimized checkpoint saving with compression"""
        if not metrics:
            return
        
        try:
            checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save to parquet with timestamp and compression
            timestamp = int(time.time())
            metrics_path = os.path.join(checkpoint_dir, f"{candidate}_{suffix}_{timestamp}.parquet")
            
            # Convert to DataFrame efficiently
            if isinstance(metrics, dict):
                # Handle dict of lists format with optimized conversion
                df = pd.DataFrame(metrics)
            else:
                # Handle list of dicts format with optimized conversion
                df = pd.DataFrame.from_records(metrics)
            
            # Use PyArrow and Parquet with optimized compression
            table = pa.Table.from_pandas(df)
            pq.write_table(table, metrics_path, compression='snappy')  # Snappy is faster than default
            
            print(f"Saved checkpoint to {metrics_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")

    def _save_and_summarize_metrics(self, metrics, candidate, filename_prefix=None):
        """Optimized metrics saving and summary generation"""
        if not metrics:
            print("No metrics to save")
            return {}
        
        # Determine filename
        if filename_prefix is None:
            filename_prefix = candidate
        
        # Create metrics directory
        metrics_dir = os.path.join(self.output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Save metrics to parquet with optimized settings
        metrics_path = os.path.join(metrics_dir, f"{filename_prefix}.parquet")
        
        # Convert to DataFrame efficiently
        if isinstance(metrics, dict):
            # Handle dict of lists format with optimized conversion
            df = pd.DataFrame(metrics)
        else:
            # Handle list of dicts format with optimized conversion
            df = pd.DataFrame.from_records(metrics)
        
        # Use PyArrow for faster writing and better compression
        table = pa.Table.from_pandas(df)
        pq.write_table(table, metrics_path, compression='snappy')
        
        # Generate summary with optimized calculations
        summary = self.generate_summary(metrics, candidate)
        summary_path = os.path.join(metrics_dir, f"{filename_prefix}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSaved {len(df):,} metrics to {metrics_path}")
        print(f"Saved summary to {summary_path}")
        
        return metrics

    def process_timestamp(self, timestamp, candidate):
        """
        Process a single timestamp to create orderbook metrics with optimized loading
        
        Args:
            timestamp: The timestamp to process
            candidate: The candidate name
            
        Returns:
            Orderbook metrics dict or None if processing fails
        """
        try:
            # Load fills around the target timestamp with a buffer
            buffer_before = 3600  # 1 hour buffer before
            buffer_after = 3600   # 1 hour buffer after
            
            start_time = timestamp - buffer_before
            end_time = timestamp + buffer_after
            
            # Load relevant fill events
            fills = []
            for chunk in self.stream_fill_events(candidate, start_time, end_time):
                fills.extend(chunk)
            
            if not fills:
                return None
            
            # Sort fills by timestamp
            fills.sort(key=lambda x: x['timestamp'])
            fills_array = np.array(fills)
            
            # Create orderbook for this timestamp with optimized function
            return create_orderbook_chunk(fills_array, timestamp)
        except Exception as e:
            print(f"Error processing timestamp {timestamp}: {e}")
            return None

    def combine_metrics(self, metrics_list):
        """
        Optimized version that combines a list of metrics dictionaries into a single dictionary
        using vectorized operations when possible
        
        Args:
            metrics_list: List of dictionaries with metrics
            
        Returns:
            Combined metrics dictionary
        """
        # Filter out None values
        metrics_list = [m for m in metrics_list if m is not None]
        
        if not metrics_list:
            return {}
        
        # Fast path for single item
        if len(metrics_list) == 1:
            return {k: [v] for k, v in metrics_list[0].items()}
        
        # Get all keys
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        
        # Initialize dictionary with empty lists
        combined = {key: [] for key in all_keys}
        
        # Batch processing for better performance
        batch_size = 10000
        for i in range(0, len(metrics_list), batch_size):
            batch = metrics_list[i:i+batch_size]
            
            # Combine this batch
            for metric in batch:
                for key, value in metric.items():
                    combined[key].append(value)
        
        return combined

    def generate_summary(self, metrics, candidate=None):
        """Generate summary statistics with optimized vector operations"""
        if not metrics:
            print("No metrics provided for summary generation")
            return {}
        
        if candidate:
            print(f"Generating summary for {candidate}...")
        
        try:
            # Prepare data for vectorized operations
            if isinstance(metrics, dict):
                # Handle dict of lists format
                mid_prices = np.array(metrics.get('mid_price', []))
                spreads = np.array(metrics.get('spread', []))
                imbalances = np.array(metrics.get('imbalance', []))
            else:
                # Handle list of dicts format
                mid_prices = np.array([m.get('mid_price', 0) for m in metrics])
                spreads = np.array([m.get('spread', 0) for m in metrics])
                imbalances = np.array([m.get('imbalance', 0) for m in metrics])
            
            # Calculate key statistics with optimized NumPy operations
            # Use robust statistics to handle outliers
            mid_prices = mid_prices[np.isfinite(mid_prices)]  # Remove infinities
            spreads = spreads[np.isfinite(spreads)]  # Remove infinities
            imbalances = imbalances[np.isfinite(imbalances)]  # Remove infinities
            
            # Handle empty arrays
            if len(mid_prices) == 0:
                avg_mid = 0
            else:
                avg_mid = np.mean(mid_prices)
                
            if len(spreads) == 0:
                avg_spread, max_spread, min_spread = 0, 0, 0
            else:
                avg_spread = np.mean(spreads)
                max_spread = np.max(spreads)
                min_spread = np.min(spreads)
                
            if len(imbalances) == 0:
                avg_imbalance = 0
            else:
                # Use robust mean for imbalance (can have extreme values)
                imb_sorted = np.sort(imbalances)
                trim_percent = 0.05  # Trim 5% from each end
                trim_size = int(len(imb_sorted) * trim_percent)
                if len(imb_sorted) > trim_size * 2:
                    trimmed = imb_sorted[trim_size:-trim_size]
                    avg_imbalance = np.mean(trimmed)
                else:
                    avg_imbalance = np.median(imbalances)  # Use median for small samples
            
            # Add more detailed statistics for better insights
            summary = {
                'basic_stats': {
                    'avg_mid': float(avg_mid),
                    'avg_spread': float(avg_spread),
                    'max_spread': float(max_spread),
                    'min_spread': float(min_spread),
                    'avg_imbalance': float(avg_imbalance)
                },
                'metadata': {
                    'timestamp_count': len(mid_prices),
                    'generated_at': datetime.now().isoformat(),
                    'candidate': candidate
                }
            }
            
            # Add percentiles for better distribution understanding
            if len(spreads) > 0:
                summary['percentiles'] = {
                    'spread_p25': float(np.percentile(spreads, 25)),
                    'spread_p50': float(np.percentile(spreads, 50)),
                    'spread_p75': float(np.percentile(spreads, 75)),
                    'spread_p95': float(np.percentile(spreads, 95)),
                    'mid_price_p25': float(np.percentile(mid_prices, 25)),
                    'mid_price_p50': float(np.percentile(mid_prices, 50)),
                    'mid_price_p75': float(np.percentile(mid_prices, 75)),
                    'mid_price_p95': float(np.percentile(mid_prices, 95))
                }
            
            return summary
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return {'error': str(e)}

    def cleanup(self):
        """Clean up temporary files and resources"""
        # Remove any temporary memory-mapped files
        temp_files = getattr(self, '_temp_files', [])
        for filename in temp_files:
            try:
                if os.path.exists(filename):
                    os.remove(filename)
                    print(f"Removed temporary file: {filename}")
            except Exception as e:
                print(f"Error removing temporary file {filename}: {e}")

    def __del__(self):
        """Destructor to clean up resources"""
        self.cleanup()

def parse_args():
    parser = argparse.ArgumentParser(description="Ultra-fast orderbook reconstruction")
    parser.add_argument("--db-file", default="polymarket_orderbook.db", help="Path to SQLite database")
    parser.add_argument("--output-dir", default="orderbook_data", help="Output directory for parquet files")
    parser.add_argument("--all-timestamps", action="store_true", help="Use all fill event timestamps (no sampling)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples per candidate")
    parser.add_argument("--time-start", type=int, help="Start timestamp (Unix seconds)")
    parser.add_argument("--time-end", type=int, help="End timestamp (Unix seconds)")
    parser.add_argument("--candidate", choices=["Trump", "Harris"], help="Only process a specific candidate")
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    
    return parser.parse_args()

def main():
    """Main function with command-line argument support and optimized flow"""
    args = parse_args()
    
    print("===== Ultra-Fast Orderbook Reconstruction =====")
    
    # Check if database exists
    db_file = args.db_file
    if not os.path.exists(db_file):
        print(f"Database file {db_file} not found!")
        print("Please run polymarket_orderbook.py first to create and populate the database.")
        return
    
    # Create reconstructor with custom settings
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    reconstructor = UltraFastOrderbookReconstructor(db_file, output_dir)
    
    # Set custom worker count if specified
    if args.workers:
        reconstructor.max_workers = args.workers
    
    # Handle candidate filtering
    candidates = [args.candidate] if args.candidate else reconstructor.candidates
    
    # Set up time range if specified
    time_range = None
    if args.time_start or args.time_end:
        # Get default time range if only one end is specified
        if args.time_start and not args.time_end:
            end_time = int(time.time())
        else:
            end_time = args.time_end
            
        if args.time_end and not args.time_start:
            # Start 30 days before end if not specified
            start_time = args.time_end - (30 * 86400)
        else:
            start_time = args.time_start
            
        time_range = (start_time, end_time)
        
        print(f"Using time range: {datetime.fromtimestamp(start_time)} to {datetime.fromtimestamp(end_time)}")
    
    # Track overall start time
    overall_start = time.time()
    all_results = {}
    
    # Process each candidate
    for candidate in candidates:
        print(f"\n===== Processing {candidate} =====")
        try:
            # Process the candidate with optimized chunking
            result = reconstructor.process_candidate_direct(
                candidate,
                args.workers,
                args.max_samples
            )
            
            if result:
                all_results[candidate] = result
                
        except Exception as e:
            print(f"Error processing {candidate}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Report overall stats
    elapsed = time.time() - overall_start
    print(f"\n===== Complete reconstruction finished in {elapsed:.2f} seconds =====")
    
    metrics_count = 0
    for candidate in all_results:
        if isinstance(all_results[candidate], dict) and 'timestamp' in all_results[candidate]:
            count = len(all_results[candidate]['timestamp'])
        elif isinstance(all_results[candidate], dict):
            # Try to get first key's length
            first_key = next(iter(all_results[candidate]))
            count = len(all_results[candidate][first_key])
        else:
            # For list of dicts
            count = len(all_results[candidate])
            
        metrics_count += count
        rate = count / (elapsed / 60)
        print(f"{candidate}: {count:,} metrics generated ({rate:.1f} metrics/minute)")
    
    print(f"Total: {metrics_count:,} metrics in {elapsed/60:.1f} minutes")
    print(f"Overall rate: {metrics_count/(elapsed/60):.1f} metrics/minute")
    print("\nResults saved to directory:", output_dir)
    
    # Clean up any temporary files
    reconstructor.cleanup()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc() 