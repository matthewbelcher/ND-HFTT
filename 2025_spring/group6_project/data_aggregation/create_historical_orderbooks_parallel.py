#!/usr/bin/env python3
import sqlite3
import time
from datetime import datetime
import json
import os
import multiprocessing
from multiprocessing import Pool, cpu_count, shared_memory
import numpy as np
from collections import defaultdict
import pandas as pd
import sys
from hft_orderbook_reconstruction import OrderbookReconstructor

DB_PATH = "polymarket_orderbook.db"
OUTPUT_DB = "historical_orderbooks.db"

# Extreme optimization settings
MAX_WORKERS = 100  # Override this if needed - be careful with system stability!
BATCH_SIZE = 1000  # Increased batch size for DB operations
MIN_TIMESTAMP = 0  # Can be overridden via command line
USE_SQLITE_WAL = True  # Use SQLite WAL mode for faster writes
USE_MEMORY_DB = True  # Use in-memory database for faster intermediate storage
MAX_FILLS_PER_TS = 1000  # Maximum fills to process per timestamp
SKIP_FACTOR = 1  # Process ALL timestamps (no skipping)

# Create connection pool for database operations
DB_POOL = {}

def get_db_connection(db_path):
    """Get a cached database connection from the pool"""
    pid = os.getpid()
    if pid not in DB_POOL or db_path not in DB_POOL[pid]:
        if pid not in DB_POOL:
            DB_POOL[pid] = {}
        
        conn = sqlite3.connect(db_path)
        
        # Enable WAL mode for better write performance
        if USE_SQLITE_WAL:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
        # Increase cache size
        conn.execute("PRAGMA cache_size=-50000")  # 50MB cache
        
        DB_POOL[pid][db_path] = conn
    
    return DB_POOL[pid][db_path]

def get_all_timestamps(candidate, min_timestamp=0):
    """
    Get all unique timestamps for a candidate, starting from min_timestamp
    """
    conn = get_db_connection(DB_PATH)
    cursor = conn.cursor()
    
    # Query for unique timestamps from both tables
    cursor.execute("""
        SELECT DISTINCT timestamp 
        FROM (
            SELECT timestamp FROM reconstructed_orderbook WHERE candidate = ? AND timestamp >= ?
            UNION
            SELECT timestamp FROM order_filled_events WHERE candidate = ? AND timestamp >= ?
        )
        ORDER BY timestamp
    """, (candidate, min_timestamp, candidate, min_timestamp))
    
    all_timestamps = [row[0] for row in cursor.fetchall()]
    
    # No skipping - process all timestamps
    return all_timestamps

def setup_output_database():
    """Create the output database for storing reconstructed orderbooks"""
    if USE_MEMORY_DB:
        # Using in-memory database temporarily, will copy to disk later
        conn = sqlite3.connect(":memory:")
    else:
        conn = sqlite3.connect(OUTPUT_DB)
    
    cursor = conn.cursor()
    
    # Create table for storing orderbook snapshots
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS historical_orderbooks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate TEXT,
        timestamp INTEGER,
        best_bid REAL,
        best_ask REAL,
        bid_depth REAL,
        ask_depth REAL,
        spread REAL,
        mid_price REAL,
        imbalance REAL,
        weighted_mid REAL,
        bids_json TEXT,
        asks_json TEXT
    )
    ''')
    
    # Create indices for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_hist_timestamp ON historical_orderbooks (timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_hist_candidate ON historical_orderbooks (candidate)')
    
    # Create unique index on candidate+timestamp for fast lookups and to prevent duplicates
    cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_hist_candidate_timestamp ON historical_orderbooks (candidate, timestamp)')
    
    conn.commit()
    
    if USE_MEMORY_DB:
        # Copy schema to disk db
        disk_conn = sqlite3.connect(OUTPUT_DB)
        for line in conn.iterdump():
            if line.startswith('CREATE TABLE') or line.startswith('CREATE INDEX') or line.startswith('CREATE UNIQUE'):
                disk_conn.execute(line)
        disk_conn.commit()
        disk_conn.close()

def check_existing_timestamp(candidate, timestamp):
    """Check if a timestamp already exists in the database"""
    conn = get_db_connection(OUTPUT_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) FROM historical_orderbooks
        WHERE candidate = ? AND timestamp = ?
    """, (candidate, timestamp))
    
    exists = cursor.fetchone()[0] > 0
    return exists

def store_orderbook_batch(candidate, orderbooks_and_metrics, worker_id):
    """Store a batch of reconstructed orderbooks in the output database"""
    if not orderbooks_and_metrics:
        return 0
    
    # Use memory database for faster processing if enabled
    if USE_MEMORY_DB:
        # Each worker gets its own in-memory DB
        memory_db = f":memory:{worker_id}"
        conn = get_db_connection(memory_db)
        cursor = conn.cursor()
        
        # Ensure table exists
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS historical_orderbooks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate TEXT,
            timestamp INTEGER,
            best_bid REAL,
            best_ask REAL,
            bid_depth REAL,
            ask_depth REAL,
            spread REAL,
            mid_price REAL,
            imbalance REAL,
            weighted_mid REAL,
            bids_json TEXT,
            asks_json TEXT
        )
        ''')
    else:
        conn = get_db_connection(OUTPUT_DB)
        cursor = conn.cursor()
    
    # Use transaction for faster bulk inserts
    conn.execute("BEGIN TRANSACTION")
    
    count = 0
    for orderbook, metrics in orderbooks_and_metrics:
        try:
            # Better serialization of bids and asks to preserve size information
            # RESTORE ORIGINAL FORMAT: Use the original dictionary format for compatibility
            bids_json = json.dumps([
                {"price": price, "size": total_size}
                for price, total_size, orders in orderbook['bids']
                if total_size > 0
            ])
            
            asks_json = json.dumps([
                {"price": price, "size": total_size}
                for price, total_size, orders in orderbook['asks']
                if total_size > 0
            ])
            
            # Verify we have depth values
            bid_depth = metrics['bid_depth']
            ask_depth = metrics['ask_depth']
            
            # If the metrics show zero depth but we have price levels, recalculate
            if (bid_depth == 0 and len(orderbook['bids']) > 0) or (ask_depth == 0 and len(orderbook['asks']) > 0):
                # Recalculate depth from the actual orderbook
                bid_depth = sum(level[1] for level in orderbook['bids']) if orderbook['bids'] else 0
                ask_depth = sum(level[1] for level in orderbook['asks']) if orderbook['asks'] else 0
                
                # Update metrics
                metrics['bid_depth'] = bid_depth
                metrics['ask_depth'] = ask_depth
                
                # Recalculate imbalance
                if ask_depth > 0:
                    metrics['imbalance'] = bid_depth / ask_depth
                else:
                    metrics['imbalance'] = float('inf') if bid_depth > 0 else 1.0
            
            # Insert or replace to avoid duplicates
            cursor.execute('''
            INSERT OR REPLACE INTO historical_orderbooks (
                candidate, timestamp, best_bid, best_ask, bid_depth, ask_depth,
                spread, mid_price, imbalance, weighted_mid, bids_json, asks_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                candidate,
                orderbook['timestamp'],
                orderbook['bids'][0][0] if orderbook['bids'] else None,
                orderbook['asks'][0][0] if orderbook['asks'] else None,
                metrics['bid_depth'],
                metrics['ask_depth'],
                metrics['bid-ask_spread'],
                metrics['mid_price'],
                metrics['imbalance'],
                metrics['weighted_mid_price'],
                bids_json,
                asks_json
            ))
            count += 1
        except Exception as e:
            # Simple error log to avoid output overhead
            pass
    
    conn.commit()
    
    # If using in-memory DB, sync to disk periodically
    if USE_MEMORY_DB and count > 0:
        # Copy to disk
        memory_conn = get_db_connection(memory_db)
        disk_conn = get_db_connection(OUTPUT_DB)
        
        disk_conn.execute("BEGIN TRANSACTION")
        
        # Extract all data from memory DB
        memory_conn.row_factory = sqlite3.Row
        memory_cursor = memory_conn.cursor()
        memory_cursor.execute("SELECT * FROM historical_orderbooks")
        
        # Insert all rows to disk DB
        disk_cursor = disk_conn.cursor()
        for row in memory_cursor:
            # Convert Row to dict
            row_dict = {k: row[k] for k in row.keys()}
            
            # Skip id column
            columns = [k for k in row_dict if k != 'id']
            placeholders = ", ".join(["?"] * len(columns))
            columns_str = ", ".join(columns)
            
            query = f"INSERT OR REPLACE INTO historical_orderbooks ({columns_str}) VALUES ({placeholders})"
            disk_cursor.execute(query, [row_dict[col] for col in columns])
        
        disk_conn.commit()
        
        # Clear memory table after syncing
        memory_cursor.execute("DELETE FROM historical_orderbooks")
        memory_conn.commit()
    
    return count

def process_timestamp_chunk(args):
    """
    Process a chunk of timestamps for a candidate
    
    Args:
        args: Tuple of (candidate, chunk_of_timestamps, chunk_id, total_chunks)
    
    Returns:
        Number of successfully processed timestamps
    """
    candidate, timestamps, chunk_id, total_chunks = args
    
    worker_id = multiprocessing.current_process().name
    print(f"Worker {worker_id} starting on chunk {chunk_id}/{total_chunks} with {len(timestamps)} timestamps")
    
    # Create a new reconstructor instance for this worker
    reconstructor = OrderbookReconstructor(db_file=DB_PATH)
    reconstructor.connect()
    
    results = []
    count = 0
    processed = 0
    
    try:
        for i, ts in enumerate(timestamps):
            try:
                # Check if this timestamp is already processed, skip if it is
                if check_existing_timestamp(candidate, ts):
                    processed += 1
                    
                    # Only print progress less frequently
                    if i % 100 == 0:
                        print(f"Worker {worker_id} - Chunk {chunk_id}/{total_chunks}: {i}/{len(timestamps)} timestamps processed ({processed} skipped)")
                    
                    continue
                
                # Print progress periodically with less frequency
                if i % 100 == 0:
                    print(f"Worker {worker_id} - Chunk {chunk_id}/{total_chunks}: {i}/{len(timestamps)} timestamps processed ({processed} skipped)")
                
                # TURBO: Use cached results when possible with aggressive caching enabled
                orderbook = reconstructor.get_orderbook_at(candidate, ts, use_cache=True)
                
                # Compute metrics
                metrics = reconstructor.compute_book_metrics(orderbook)
                
                # Add to results
                results.append((orderbook, metrics))
                
                # Store in batches to reduce database contention
                if len(results) >= BATCH_SIZE:
                    stored = store_orderbook_batch(candidate, results, worker_id)
                    count += stored
                    results = []
                
                count += 1
                
                # TURBO: Occasionally clear some memory
                if i % 200 == 199:
                    # Clear some of the cache to manage memory
                    if len(reconstructor.orderbook_cache) > 100:
                        # Keep only most recent entries
                        keys = sorted(reconstructor.orderbook_cache.keys(), 
                                     key=lambda k: reconstructor.orderbook_cache[k]['timestamp'])
                        for old_key in keys[:-50]:  # Keep last 50 items
                            del reconstructor.orderbook_cache[old_key]
                
            except Exception as e:
                # Simple error for speed
                pass
        
        # Store any remaining results
        if results:
            stored = store_orderbook_batch(candidate, results, worker_id)
            count += stored
            
    finally:
        # Clean up
        reconstructor.disconnect()
        
        # Clean up connection pool for this worker
        pid = os.getpid()
        if pid in DB_POOL:
            for db_path in DB_POOL[pid]:
                try:
                    DB_POOL[pid][db_path].close()
                except:
                    pass
            del DB_POOL[pid]
    
    print(f"Worker {worker_id} completed chunk {chunk_id}/{total_chunks} - Processed {count} timestamps, skipped {processed}")
    return count

def process_candidate_parallel(candidate, num_workers=None, min_timestamp=0):
    """Process all timestamps for a candidate in parallel and store reconstructed orderbooks"""
    print(f"Getting timestamps for {candidate} starting from {datetime.fromtimestamp(min_timestamp)}")
    timestamps = get_all_timestamps(candidate, min_timestamp)
    
    if not timestamps:
        print(f"No timestamps found for {candidate}")
        return 0
    
    total = len(timestamps)
    print(f"Found {total} timestamps for {candidate}")
    
    if total == 0:
        return 0
        
    print(f"Time range: {datetime.fromtimestamp(timestamps[0])} to {datetime.fromtimestamp(timestamps[-1])}")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count() * 2, MAX_WORKERS)  # Use 2x CPU count or max, whichever is smaller
    
    print(f"Using {num_workers} parallel workers")
    
    # Split timestamps into chunks for each worker
    chunks = np.array_split(timestamps, num_workers)
    
    # Prepare args for each worker
    args = [(candidate, chunk.tolist(), i+1, len(chunks)) for i, chunk in enumerate(chunks)]
    
    # Process in parallel with increased max_tasks_per_child for better performance
    start_time = time.time()
    with Pool(processes=num_workers, maxtasksperchild=5) as pool:
        results = pool.map(process_timestamp_chunk, args)
    
    total_processed = sum(results)
    elapsed = time.time() - start_time
    
    print(f"Parallel processing complete for {candidate}")
    print(f"Processed {total_processed} timestamps in {elapsed:.2f} seconds")
    print(f"Processing rate: {total_processed/elapsed:.2f} timestamps/second")
    
    return total_processed

def check_database_exists():
    """Check if the output database already exists with required schema"""
    try:
        if not os.path.exists(OUTPUT_DB):
            return False
            
        conn = get_db_connection(OUTPUT_DB)
        cursor = conn.cursor()
        
        # Check for the main table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='historical_orderbooks'")
        if not cursor.fetchone():
            return False
            
        # Check for required indices
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_hist_candidate_timestamp'")
        if not cursor.fetchone():
            return False
            
        return True
    except:
        return False

def main():
    """Main function to reconstruct and store all historical orderbooks in parallel"""
    global MIN_TIMESTAMP
    
    # Check for command line args
    if len(sys.argv) > 1:
        try:
            # If timestamp is provided as a string date (YYYY-MM-DD)
            if '-' in sys.argv[1]:
                date_str = sys.argv[1]
                # Convert to timestamp
                min_date = datetime.strptime(date_str, "%Y-%m-%d")
                MIN_TIMESTAMP = int(min_date.timestamp())
            else:
                # Direct timestamp
                MIN_TIMESTAMP = int(sys.argv[1])
            
            print(f"Starting from timestamp: {MIN_TIMESTAMP} ({datetime.fromtimestamp(MIN_TIMESTAMP)})")
        except:
            print("Invalid timestamp format. Using default (process all).")
    
    print(f"Creating historical orderbooks database for backtesting...")
    print(f"Starting from {datetime.fromtimestamp(MIN_TIMESTAMP)}")
    print(f"EXTREME PERFORMANCE MODE: Using maximum parallelism and optimizations")
    
    # Check if database already exists and set up if needed
    if not check_database_exists():
        print("Setting up output database schema...")
        setup_output_database()
    else:
        print("Output database already exists with proper schema.")
    
    # Auto-determine optimal number of workers - EXTREME EDITION
    num_cores = cpu_count()
    suggested_workers = min(num_cores * 2, MAX_WORKERS)  # Use double the cores!
    
    print(f"\nYour system has {num_cores} CPU cores")
    print(f"TURBO SETTING: Using {suggested_workers} worker processes (oversubscribing CPUs)")
    
    # Process each candidate
    candidates = ["Trump", "Harris"]
    total_processed = 0
    
    for candidate in candidates:
        print(f"\nProcessing {candidate}...")
        count = process_candidate_parallel(candidate, suggested_workers, MIN_TIMESTAMP)
        total_processed += count
        print(f"Processed {count} orderbooks for {candidate}")
    
    print(f"\nTotal orderbooks created: {total_processed}")
    print(f"Results stored in: {OUTPUT_DB}")
    
    # Create summary
    conn = sqlite3.connect(OUTPUT_DB)
    df = pd.read_sql("SELECT candidate, COUNT(*) as count, MIN(timestamp) as min_time, MAX(timestamp) as max_time FROM historical_orderbooks GROUP BY candidate", conn)
    
    print("\nSummary:")
    for _, row in df.iterrows():
        min_date = datetime.fromtimestamp(row['min_time']).strftime('%Y-%m-%d %H:%M:%S')
        max_date = datetime.fromtimestamp(row['max_time']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{row['candidate']}: {row['count']} orderbooks from {min_date} to {max_date}")
    
    conn.close()
    
    print("\nHIGH PERFORMANCE PROCESSING COMPLETE! You can now use the historical_orderbooks.db database for backtesting strategies.")

if __name__ == "__main__":
    overall_start_time = time.time()
    main()
    elapsed = time.time() - overall_start_time
    print(f"Total execution time: {elapsed:.2f} seconds") 