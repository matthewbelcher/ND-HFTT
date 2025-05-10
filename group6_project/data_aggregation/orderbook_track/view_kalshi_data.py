import sqlite3
import json
import argparse
from datetime import datetime
from tabulate import tabulate
import os

# Database file
DB_PATH = "kalshi_orderbooks.db"

def check_db_exists():
    """Check if the database file exists"""
    if not os.path.exists(DB_PATH):
        print(f"Database file {DB_PATH} not found.")
        return False
    return True

def list_markets():
    """List all markets in the database"""
    if not check_db_exists():
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT market_ticker, asset, event_ticker, description, last_updated FROM markets ORDER BY asset, market_ticker")
    markets = cursor.fetchall()
    
    if not markets:
        print("No markets found in the database.")
        conn.close()
        return
    
    headers = ["Market Ticker", "Asset", "Event", "Description", "Last Updated"]
    print("\n=== MARKETS ===")
    print(tabulate(markets, headers=headers, tablefmt="grid"))
    
    conn.close()

def view_market_summary(market_ticker=None):
    """View summary of data for a specific market or all markets"""
    if not check_db_exists():
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if market_ticker:
        # Check if market exists
        cursor.execute("SELECT COUNT(*) FROM markets WHERE market_ticker = ?", (market_ticker,))
        if cursor.fetchone()[0] == 0:
            print(f"Market {market_ticker} not found in the database.")
            conn.close()
            return
        
        where_clause = f"WHERE market_ticker = '{market_ticker}'"
    else:
        where_clause = ""
    
    # Get snapshot counts
    cursor.execute(f"""
    SELECT market_ticker, COUNT(*), MIN(timestamp), MAX(timestamp)
    FROM orderbook_snapshots
    {where_clause}
    GROUP BY market_ticker
    """)
    snapshot_stats = cursor.fetchall()
    
    # Get delta counts
    cursor.execute(f"""
    SELECT market_ticker, COUNT(*), MIN(timestamp), MAX(timestamp)
    FROM orderbook_deltas
    {where_clause}
    GROUP BY market_ticker
    """)
    delta_stats = cursor.fetchall()
    
    print("\n=== DATA SUMMARY ===")
    
    if snapshot_stats:
        headers = ["Market Ticker", "Snapshot Count", "First Snapshot", "Latest Snapshot"]
        print("\nOrderbook Snapshots:")
        print(tabulate(snapshot_stats, headers=headers, tablefmt="grid"))
    else:
        print("\nNo orderbook snapshots found.")
    
    if delta_stats:
        headers = ["Market Ticker", "Delta Count", "First Delta", "Latest Delta"]
        print("\nOrderbook Deltas:")
        print(tabulate(delta_stats, headers=headers, tablefmt="grid"))
    else:
        print("\nNo orderbook deltas found.")
    
    conn.close()

def view_latest_snapshot(market_ticker, full_data=False):
    """View the latest orderbook snapshot for a specific market"""
    if not check_db_exists():
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if market exists
    cursor.execute("SELECT COUNT(*) FROM markets WHERE market_ticker = ?", (market_ticker,))
    if cursor.fetchone()[0] == 0:
        print(f"Market {market_ticker} not found in the database.")
        conn.close()
        return
    
    # Get the latest snapshot
    cursor.execute("""
    SELECT id, timestamp, snapshot_data
    FROM orderbook_snapshots
    WHERE market_ticker = ?
    ORDER BY timestamp DESC
    LIMIT 1
    """, (market_ticker,))
    
    snapshot = cursor.fetchone()
    
    if not snapshot:
        print(f"No snapshots found for market {market_ticker}.")
        conn.close()
        return
    
    snapshot_id, timestamp, snapshot_data = snapshot
    snapshot_data = json.loads(snapshot_data)
    
    print(f"\n=== LATEST SNAPSHOT FOR {market_ticker} ===")
    print(f"Snapshot ID: {snapshot_id}")
    print(f"Timestamp: {timestamp}")
    
    yes_orders = snapshot_data.get("yes", [])
    no_orders = snapshot_data.get("no", [])
    
    if full_data:
        # Show full orderbook
        if yes_orders:
            print("\nYES Orders:")
            yes_formatted = [[order[0]/100, order[1]] for order in yes_orders]
            print(tabulate(yes_formatted, headers=["Price", "Size"], tablefmt="grid"))
        else:
            print("\nNo YES orders in this snapshot.")
        
        if no_orders:
            print("\nNO Orders:")
            no_formatted = [[order[0]/100, order[1]] for order in no_orders]
            print(tabulate(no_formatted, headers=["Price", "Size"], tablefmt="grid"))
        else:
            print("\nNo NO orders in this snapshot.")
    else:
        # Show summary of orderbook
        yes_total_size = sum(order[1] for order in yes_orders) if yes_orders else 0
        no_total_size = sum(order[1] for order in no_orders) if no_orders else 0
        
        print(f"\nYES Orders: {len(yes_orders)} orders, total size: {yes_total_size}")
        if yes_orders:
            best_yes_bid = max(order[0]/100 for order in yes_orders)
            print(f"Best YES bid: {best_yes_bid}")
        
        print(f"\nNO Orders: {len(no_orders)} orders, total size: {no_total_size}")
        if no_orders:
            best_no_bid = max(order[0]/100 for order in no_orders)
            print(f"Best NO bid: {best_no_bid}")
    
    conn.close()

def view_recent_deltas(market_ticker, limit=10):
    """View recent orderbook deltas for a specific market"""
    if not check_db_exists():
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if market exists
    cursor.execute("SELECT COUNT(*) FROM markets WHERE market_ticker = ?", (market_ticker,))
    if cursor.fetchone()[0] == 0:
        print(f"Market {market_ticker} not found in the database.")
        conn.close()
        return
    
    # Get recent deltas
    cursor.execute("""
    SELECT id, timestamp, price, delta, side, seq
    FROM orderbook_deltas
    WHERE market_ticker = ?
    ORDER BY timestamp DESC
    LIMIT ?
    """, (market_ticker, limit))
    
    deltas = cursor.fetchall()
    
    if not deltas:
        print(f"No deltas found for market {market_ticker}.")
        conn.close()
        return
    
    print(f"\n=== RECENT DELTAS FOR {market_ticker} ===")
    headers = ["ID", "Timestamp", "Price (cents)", "Delta", "Side", "Sequence"]
    print(tabulate(deltas, headers=headers, tablefmt="grid"))
    
    conn.close()

def show_market_stats():
    """Show comprehensive statistics about all markets including change metrics"""
    if not check_db_exists():
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get total market count
    cursor.execute("SELECT COUNT(*) FROM markets")
    market_count = cursor.fetchone()[0]
    
    print(f"\n=== KALSHI MARKET STATISTICS ===")
    print(f"Total Markets: {market_count}")
    
    # Get stats for each market
    cursor.execute("""
    SELECT m.market_ticker, m.asset, m.event_ticker, 
           COUNT(DISTINCT s.id) as snapshot_count,
           COUNT(DISTINCT d.id) as delta_count,
           MIN(s.timestamp) as first_snapshot,
           MAX(s.timestamp) as last_snapshot
    FROM markets m
    LEFT JOIN orderbook_snapshots s ON m.market_ticker = s.market_ticker
    LEFT JOIN orderbook_deltas d ON m.market_ticker = d.market_ticker
    GROUP BY m.market_ticker
    ORDER BY m.asset, m.market_ticker
    """)
    
    market_stats = cursor.fetchall()
    
    if not market_stats:
        print("No market data found in the database.")
        conn.close()
        return
    
    # Calculate price changes for markets with snapshots
    markets_with_changes = []
    
    for stat in market_stats:
        market_ticker = stat[0]
        snapshot_count = stat[3]
        
        if snapshot_count > 0:
            # Get first snapshot
            cursor.execute("""
            SELECT timestamp, snapshot_data
            FROM orderbook_snapshots
            WHERE market_ticker = ?
            ORDER BY timestamp ASC
            LIMIT 1
            """, (market_ticker,))
            
            first = cursor.fetchone()
            first_data = json.loads(first[1]) if first else None
            
            # Get last snapshot
            cursor.execute("""
            SELECT timestamp, snapshot_data
            FROM orderbook_snapshots
            WHERE market_ticker = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """, (market_ticker,))
            
            last = cursor.fetchone()
            last_data = json.loads(last[1]) if last else None
            
            # Calculate price and volume changes
            yes_price_change = "N/A"
            no_price_change = "N/A"
            yes_volume_change = "N/A"
            no_volume_change = "N/A"
            
            if first_data and last_data:
                # Calculate best YES bid change
                first_yes = first_data.get("yes", [])
                last_yes = last_data.get("yes", [])
                
                first_yes_bid = max([order[0]/100 for order in first_yes]) if first_yes else None
                last_yes_bid = max([order[0]/100 for order in last_yes]) if last_yes else None
                
                if first_yes_bid is not None and last_yes_bid is not None:
                    yes_price_change = f"{last_yes_bid - first_yes_bid:.2f}"
                
                # Calculate best NO bid change
                first_no = first_data.get("no", [])
                last_no = last_data.get("no", [])
                
                first_no_bid = max([order[0]/100 for order in first_no]) if first_no else None
                last_no_bid = max([order[0]/100 for order in last_no]) if last_no else None
                
                if first_no_bid is not None and last_no_bid is not None:
                    no_price_change = f"{last_no_bid - first_no_bid:.2f}"
                
                # Calculate volume changes
                first_yes_volume = sum(order[1] for order in first_yes) if first_yes else 0
                last_yes_volume = sum(order[1] for order in last_yes) if last_yes else 0
                yes_volume_change = f"{last_yes_volume - first_yes_volume}"
                
                first_no_volume = sum(order[1] for order in first_no) if first_no else 0
                last_no_volume = sum(order[1] for order in last_no) if last_no else 0
                no_volume_change = f"{last_no_volume - first_no_volume}"
            
            # Add to result list
            markets_with_changes.append(list(stat) + [yes_price_change, no_price_change, yes_volume_change, no_volume_change])
        else:
            # No snapshots
            markets_with_changes.append(list(stat) + ["N/A", "N/A", "N/A", "N/A"])
    
    headers = ["Market Ticker", "Asset", "Event", "Snapshots", "Deltas", "First Snapshot", "Last Snapshot", 
               "YES Price Δ", "NO Price Δ", "YES Volume Δ", "NO Volume Δ"]
    
    print("\n=== MARKET DETAILS ===")
    print(tabulate(markets_with_changes, headers=headers, tablefmt="grid"))
    
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="View Kalshi orderbook data from the database.")
    parser.add_argument("--markets", action="store_true", help="List all markets in the database")
    parser.add_argument("--summary", action="store_true", help="View summary of all data in the database")
    parser.add_argument("--stats", action="store_true", help="Show comprehensive statistics for all markets")
    parser.add_argument("--market", type=str, help="Market ticker to view (e.g., KXBTC-25APR1600-T75750)")
    parser.add_argument("--snapshot", action="store_true", help="View latest orderbook snapshot for the specified market")
    parser.add_argument("--full", action="store_true", help="Show full orderbook data when viewing snapshots")
    parser.add_argument("--deltas", action="store_true", help="View recent orderbook deltas for the specified market")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of deltas to show (default: 10)")
    
    args = parser.parse_args()
    
    try:
        if args.markets:
            list_markets()
            return
        
        if args.summary:
            view_market_summary(args.market)
            return
        
        if args.stats:
            show_market_stats()
            return
        
        if args.market:
            if args.snapshot:
                view_latest_snapshot(args.market, args.full)
            elif args.deltas:
                view_recent_deltas(args.market, args.limit)
            else:
                # If just market is specified, show summary for that market
                view_market_summary(args.market)
        else:
            # Default action if no specific argument is provided
            list_markets()
            view_market_summary()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 