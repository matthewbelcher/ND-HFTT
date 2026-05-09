#!/usr/bin/env python3
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import os
import json
from datetime import datetime
from tqdm import tqdm
import imageio
from matplotlib.colors import LinearSegmentedColormap
import argparse

# Constants for visualization
OUTPUT_DIR = "kalshi_animations"
FPS = 30
DURATION_SECONDS = 20  # Total animation duration in seconds
MAX_FRAMES = FPS * DURATION_SECONDS  # Total frames in the animation

# Database connection
DB_PATH = "/root/home/ACE/orderbook_track/kalshi_orderbooks.db"

def check_db_exists():
    """Check if the database file exists"""
    if not os.path.exists(DB_PATH):
        print(f"Database file {DB_PATH} not found.")
        return False
    return True

def get_markets_with_most_data(limit=5):
    """Get markets with the most orderbook deltas"""
    if not check_db_exists():
        return []
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if required tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='orderbook_deltas'")
    deltas_table_exists = cursor.fetchone() is not None
    
    if not deltas_table_exists:
        print("Error: orderbook_deltas table does not exist in the database.")
        conn.close()
        return []
    
    # Check if the markets table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='markets'")
    markets_table_exists = cursor.fetchone() is not None
    
    if markets_table_exists:
        try:
            # Try to get markets with the most deltas via markets table join
            cursor.execute("""
            SELECT m.market_ticker, m.asset, COUNT(d.id) as delta_count
            FROM markets m
            JOIN orderbook_deltas d ON m.market_ticker = d.market_ticker
            GROUP BY m.market_ticker, m.asset
            ORDER BY delta_count DESC
            LIMIT ?
            """, (limit,))
            
            markets = cursor.fetchall()
            
            if markets:
                conn.close()
                return markets
        except sqlite3.Error as e:
            print(f"Error querying with markets join: {e}")
    
    # Fallback: get markets directly from orderbook_deltas table
    print("Using fallback query without markets table...")
    cursor.execute("""
    SELECT market_ticker, NULL as asset, COUNT(id) as delta_count
    FROM orderbook_deltas
    GROUP BY market_ticker
    ORDER BY delta_count DESC
    LIMIT ?
    """, (limit,))
    
    markets = cursor.fetchall()
    conn.close()
    
    return markets

def get_market_data(market_ticker):
    """Get all orderbook snapshots and deltas for a market"""
    if not check_db_exists():
        return None, None
    
    conn = sqlite3.connect(DB_PATH)
    
    # Get snapshots
    snapshots_df = pd.read_sql_query("""
    SELECT id, market_ticker, timestamp, snapshot_data
    FROM orderbook_snapshots
    WHERE market_ticker = ?
    ORDER BY timestamp
    """, conn, params=(market_ticker,))
    
    # Get deltas
    deltas_df = pd.read_sql_query("""
    SELECT id, market_ticker, timestamp, price, delta, side, seq
    FROM orderbook_deltas
    WHERE market_ticker = ?
    ORDER BY timestamp
    """, conn, params=(market_ticker,))
    
    conn.close()
    
    # Convert timestamp to datetime
    if not snapshots_df.empty:
        # Convert timestamp to proper format
        snapshots_df['datetime'] = pd.to_datetime(snapshots_df['timestamp'])
        # Ensure timestamp is numeric (Unix timestamp)
        snapshots_df['timestamp_num'] = snapshots_df['datetime'].astype('int64') // 10**9
    
    if not deltas_df.empty:
        deltas_df['datetime'] = pd.to_datetime(deltas_df['timestamp'])
        deltas_df['timestamp_num'] = deltas_df['datetime'].astype('int64') // 10**9
    
    return snapshots_df, deltas_df

def reconstruct_orderbook(snapshots_df, deltas_df, timestamps):
    """Reconstruct orderbook at multiple timestamps for animation"""
    orderbooks = []
    
    # Sort all data by timestamp
    if snapshots_df.empty:
        print("No snapshots available, cannot reconstruct orderbook")
        return orderbooks
    
    # Parse the snapshot data
    snapshots_df['parsed_data'] = snapshots_df['snapshot_data'].apply(json.loads)
    
    # For each requested timestamp, find the closest prior snapshot and apply deltas
    for ts in tqdm(timestamps, desc="Reconstructing orderbooks"):
        # Find the latest snapshot before this timestamp
        prior_snapshots = snapshots_df[snapshots_df['timestamp_num'] <= ts]
        
        if prior_snapshots.empty:
            # No prior snapshot, use the earliest one
            base_snapshot = snapshots_df.iloc[0]
        else:
            # Use the latest prior snapshot
            base_snapshot = prior_snapshots.iloc[-1]
        
        # Get the base orderbook
        base_orderbook = base_snapshot['parsed_data']
        
        # Apply all deltas between the snapshot and the target timestamp
        deltas_to_apply = deltas_df[
            (deltas_df['timestamp_num'] > base_snapshot['timestamp_num']) & 
            (deltas_df['timestamp_num'] <= ts)
        ]
        
        # Create a copy of the base orderbook
        current_orderbook = {
            'yes': base_orderbook.get('yes', []).copy(),
            'no': base_orderbook.get('no', []).copy()
        }
        
        # Apply deltas in sequence
        for _, delta in deltas_to_apply.iterrows():
            price = delta['price']
            delta_amount = delta['delta']
            side = delta['side']
            
            # Find the price level in the orderbook
            if side in current_orderbook:
                # Find the price level
                price_level_idx = None
                for i, order in enumerate(current_orderbook[side]):
                    if order[0] == price:
                        price_level_idx = i
                        break
                
                if price_level_idx is not None:
                    # Update existing price level
                    current_size = current_orderbook[side][price_level_idx][1]
                    new_size = current_size + delta_amount
                    
                    if new_size > 0:
                        # Update size
                        current_orderbook[side][price_level_idx][1] = new_size
                    else:
                        # Remove price level
                        current_orderbook[side].pop(price_level_idx)
                elif delta_amount > 0:
                    # Add new price level
                    current_orderbook[side].append([price, delta_amount])
        
        # Save the reconstructed orderbook
        orderbooks.append({
            'timestamp': ts,
            'datetime': datetime.fromtimestamp(ts),
            'orderbook': current_orderbook
        })
    
    return orderbooks

def calculate_mid_price(orderbook):
    """Calculate the mid price from an orderbook with proper Kalshi interpretation"""
    yes_orders = orderbook.get('yes', [])
    no_orders = orderbook.get('no', [])
    
    if not yes_orders and not no_orders:
        return None
    
    # Find the best bid (highest) for yes and no
    # In Kalshi markets, prices are in cents (0-100)
    best_yes_price = max([order[0]/100 for order in yes_orders]) if yes_orders else None
    best_no_price = max([order[0]/100 for order in no_orders]) if no_orders else None
    
    # For Kalshi probability markets:
    # YES price directly represents probability of event happening
    # NO price represents probability of event NOT happening, so we use (1 - NO price)
    
    if best_yes_price is not None and best_no_price is not None:
        # We have both YES and NO orders - take the average of their implied probabilities
        prob_from_yes = best_yes_price
        prob_from_no = 1.0 - best_no_price
        return (prob_from_yes + prob_from_no) / 2
    elif best_yes_price is not None:
        # Only YES orders
        return best_yes_price
    elif best_no_price is not None:
        # Only NO orders
        return 1.0 - best_no_price
    else:
        return 0.5  # Default to 50% if no orders

def create_price_timeseries(orderbooks):
    """Extract price time series from orderbooks"""
    times = []
    prices = []
    
    for ob in orderbooks:
        mid_price = calculate_mid_price(ob['orderbook'])
        if mid_price is not None:
            times.append(ob['datetime'])
            prices.append(mid_price)
    
    return pd.DataFrame({'datetime': times, 'price': prices})

def create_orderbook_animation(market_ticker, orderbooks, filename=None):
    """Create an animation of the orderbook over time"""
    if not orderbooks:
        print(f"No orderbook data for {market_ticker}")
        return
    
    if filename is None:
        filename = f"{market_ticker}_orderbook.mp4"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create temporary directory for frames
    temp_dir = os.path.join(OUTPUT_DIR, f"temp_{market_ticker}")
    os.makedirs(temp_dir, exist_ok=True)
    
    frame_files = []
    print(f"Generating {len(orderbooks)} frames for {market_ticker} animation...")
    
    # Extract price history for the bottom chart
    price_df = create_price_timeseries(orderbooks)
    
    for i, ob_data in enumerate(tqdm(orderbooks)):
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Extract orderbook data
        timestamp = ob_data['datetime']
        orderbook = ob_data['orderbook']
        
        # Extract yes and no orders
        yes_orders = orderbook.get('yes', [])
        no_orders = orderbook.get('no', [])
        
        # Sort orders by price
        yes_orders_sorted = sorted(yes_orders, key=lambda x: x[0])
        no_orders_sorted = sorted(no_orders, key=lambda x: x[0])
        
        # Calculate sizes
        yes_prices = [order[0]/100 for order in yes_orders_sorted]  # Convert cents to dollars
        yes_sizes = [order[1] for order in yes_orders_sorted]
        yes_cumulative = np.cumsum(yes_sizes)
        
        no_prices = [order[0]/100 for order in no_orders_sorted]  # Convert cents to dollars
        no_sizes = [order[1] for order in no_orders_sorted]
        no_cumulative = np.cumsum(no_sizes)
        
        # Plot depth chart
        if yes_prices:
            ax1.fill_between(yes_prices, 0, yes_cumulative, color='green', alpha=0.7, label='YES Orders')
        
        if no_prices:
            ax1.fill_between(no_prices, 0, no_cumulative, color='red', alpha=0.7, label='NO Orders')
        
        # Add mid price line
        mid_price = calculate_mid_price(orderbook)
        if mid_price is not None:
            max_depth = max(
                max(yes_cumulative) if yes_cumulative.size > 0 else 0,
                max(no_cumulative) if no_cumulative.size > 0 else 0
            )
            ax1.axvline(x=mid_price, color='black', linestyle='--', alpha=0.7, 
                       label=f'Implied Probability: {mid_price:.2f}')
        
        # Set title and labels
        ax1.set_title(f"{market_ticker} Orderbook - {timestamp}", fontsize=16)
        ax1.set_xlabel("Price (probability)", fontsize=12)
        ax1.set_ylabel("Cumulative Size", fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Set x-axis limits to 0-1 for probability
        ax1.set_xlim(0, 1)
        
        # Plot price history on the bottom chart
        if not price_df.empty:
            current_time_idx = price_df[price_df['datetime'] <= timestamp].index
            if len(current_time_idx) > 0:
                history_df = price_df.iloc[:current_time_idx[-1] + 1]
                
                ax2.plot(history_df['datetime'], history_df['price'], 'k-', linewidth=2)
                ax2.scatter([timestamp], [mid_price], color='red', s=100, zorder=5)
                
                ax2.set_title("Implied Probability History", fontsize=14)
                ax2.set_xlabel("Date", fontsize=12)
                ax2.set_ylabel("Probability", fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 1)  # Probabilities are between 0 and 1
        
        plt.tight_layout()
        
        # Save frame
        frame_file = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_file, dpi=100)
        plt.close(fig)
        
        frame_files.append(frame_file)
    
    # Create video
    print(f"Creating animation: {filename}")
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    with imageio.get_writer(output_path, fps=FPS) as writer:
        for frame_file in frame_files:
            image = imageio.imread(frame_file)
            writer.append_data(image)
    
    print(f"Animation saved to {output_path}")
    
    # Clean up temporary files
    for frame_file in frame_files:
        os.remove(frame_file)
    os.rmdir(temp_dir)

def create_heatmap_animation(market_ticker, orderbooks, filename=None):
    """Create a heatmap visualization of the orderbook over time"""
    if not orderbooks:
        print(f"No orderbook data for {market_ticker}")
        return
    
    if filename is None:
        filename = f"{market_ticker}_heatmap.mp4"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create temporary directory for frames
    temp_dir = os.path.join(OUTPUT_DIR, f"temp_heatmap_{market_ticker}")
    os.makedirs(temp_dir, exist_ok=True)
    
    frame_files = []
    print(f"Generating {len(orderbooks)} frames for {market_ticker} heatmap animation...")
    
    # Create custom colormaps
    green_cmap = LinearSegmentedColormap.from_list("Green", [(0.8,1,0.8), (0,0.7,0)])
    red_cmap = LinearSegmentedColormap.from_list("Red", [(1,0.8,0.8), (0.7,0,0)])
    
    # Extract price history for the bottom chart
    price_df = create_price_timeseries(orderbooks)
    
    for i, ob_data in enumerate(tqdm(orderbooks)):
        # Setup the figure with gridspec for complex layout
        fig = plt.figure(figsize=(12, 8), dpi=100, facecolor='black')
        gs = GridSpec(3, 1, height_ratios=[1, 2, 1])
        
        # Axis for title and timestamp
        ax_title = plt.subplot(gs[0])
        # Main orderbook heatmap
        ax_heatmap = plt.subplot(gs[1])
        # Price chart at bottom
        ax_price = plt.subplot(gs[2])
        
        # Extract orderbook data
        timestamp = ob_data['datetime']
        orderbook = ob_data['orderbook']
        
        # Title and timestamp
        title_text = f"Kalshi Market: {market_ticker}"
        timestamp_text = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        ax_title.text(0.5, 0.7, title_text, ha='center', va='center', fontsize=18, color='white', 
                    fontweight='bold', transform=ax_title.transAxes)
        ax_title.text(0.5, 0.3, timestamp_text, ha='center', va='center', fontsize=14, color='#bbbbbb',
                    transform=ax_title.transAxes)
        ax_title.axis('off')
        
        # Extract yes and no orders
        yes_orders = orderbook.get('yes', [])
        no_orders = orderbook.get('no', [])
        
        # Get the max size for normalization
        all_sizes = []
        if yes_orders:
            all_sizes.extend([order[1] for order in yes_orders])
        if no_orders:
            all_sizes.extend([order[1] for order in no_orders])
            
        max_size = max(all_sizes) if all_sizes else 1
        
        # Plot YES orders (on the positive side)
        for order in yes_orders:
            price = order[0] / 100  # Convert to probability
            size = min(order[1] / max_size, 1.0)  # Normalize size
            
            rect = plt.Rectangle((price-0.01, 0), 0.02, size,
                               facecolor=green_cmap(size),
                               alpha=0.8, edgecolor=None)
            ax_heatmap.add_patch(rect)
        
        # Plot NO orders (on the negative side)
        for order in no_orders:
            price = order[0] / 100  # Convert to probability
            size = min(order[1] / max_size, 1.0)  # Normalize size
            
            rect = plt.Rectangle((price-0.01, 0), 0.02, -size,
                               facecolor=red_cmap(size),
                               alpha=0.8, edgecolor=None)
            ax_heatmap.add_patch(rect)
        
        # Add mid price line
        mid_price = calculate_mid_price(orderbook)
        if mid_price is not None:
            ax_heatmap.axvline(x=mid_price, color='white', linestyle='--', alpha=0.7)
            
            # Add text for probability
            ax_heatmap.text(mid_price, 0.5, f"{mid_price:.2f}", 
                          color='white', fontsize=12, ha='center', va='center',
                          bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))
        
        # Set axis limits
        ax_heatmap.set_xlim(0, 1)  # Probability range
        ax_heatmap.set_ylim(-1.1, 1.1)
        
        # Set grid and background
        ax_heatmap.set_facecolor('black')
        ax_heatmap.grid(True, color='#333333', linestyle='-', linewidth=0.5)
        for spine in ax_heatmap.spines.values():
            spine.set_color('#444444')
        ax_heatmap.tick_params(colors='white')
        
        # Add YES/NO labels
        ax_heatmap.text(0.02, 0.8, "YES", color='green', fontsize=14, ha='left', va='center',
                      transform=ax_heatmap.transAxes)
        ax_heatmap.text(0.02, 0.2, "NO", color='red', fontsize=14, ha='left', va='center',
                      transform=ax_heatmap.transAxes)
        
        # Plot price history
        if not price_df.empty:
            current_time_idx = price_df[price_df['datetime'] <= timestamp].index
            if len(current_time_idx) > 0:
                history_df = price_df.iloc[:current_time_idx[-1] + 1]
                
                ax_price.plot(history_df['datetime'], history_df['price'], 
                            color='white', linewidth=2, alpha=0.8)
                ax_price.scatter([timestamp], [mid_price], 
                               color='yellow', s=80, zorder=5, edgecolor='black')
                
                ax_price.set_title("Implied Probability History", color='white', fontsize=14)
                ax_price.set_xlabel("Date", color='white')
                ax_price.set_ylabel("Probability", color='white')
                ax_price.set_ylim(0, 1)  # Probabilities range from 0 to 1
                
                # Set grid and background
                ax_price.set_facecolor('black')
                ax_price.grid(True, color='#333333', linestyle='-', linewidth=0.5)
                for spine in ax_price.spines.values():
                    spine.set_color('#444444')
                ax_price.tick_params(colors='white')
        
        plt.tight_layout()
        
        # Save frame
        frame_file = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_file, facecolor='black')
        plt.close()
        
        frame_files.append(frame_file)
    
    # Create video
    print(f"Creating animation: {filename}")
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    with imageio.get_writer(output_path, fps=FPS) as writer:
        for frame_file in frame_files:
            image = imageio.imread(frame_file)
            writer.append_data(image)
    
    print(f"Animation saved to {output_path}")
    
    # Clean up temporary files
    for frame_file in frame_files:
        os.remove(frame_file)
    os.rmdir(temp_dir)

def create_enhanced_heatmap_animation(market_ticker, orderbooks, filename=None):
    """Create an enhanced, visually stunning heatmap visualization for a specific market"""
    if market_ticker != "KXBTC-25APR2117-B87250":
        # Use regular heatmap for other markets
        return create_heatmap_animation(market_ticker, orderbooks, filename)
        
    if not orderbooks:
        print(f"No orderbook data for {market_ticker}")
        return
    
    if filename is None:
        filename = f"{market_ticker}_enhanced_heatmap.mp4"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create temporary directory for frames
    temp_dir = os.path.join(OUTPUT_DIR, f"temp_enhanced_{market_ticker}")
    os.makedirs(temp_dir, exist_ok=True)
    
    frame_files = []
    print(f"Generating {len(orderbooks)} frames for {market_ticker} enhanced animation...")
    
    # Create custom colormaps with more vibrant colors
    yes_cmap = LinearSegmentedColormap.from_list("YesGradient", [(0.8,1,0.8), (0,0.8,0.2), (0,0.5,0)])
    no_cmap = LinearSegmentedColormap.from_list("NoGradient", [(1,0.8,0.8), (1,0.2,0.2), (0.6,0,0)])
    
    # Extract price history for the chart
    price_df = create_price_timeseries(orderbooks)
    
    # Calculate additional metrics for visualization
    activity_levels = []
    for i in range(1, len(orderbooks)):
        curr_ob = orderbooks[i]['orderbook']
        prev_ob = orderbooks[i-1]['orderbook']
        
        # Count changes in orders
        curr_yes = set(tuple(order) for order in curr_ob.get('yes', []))
        prev_yes = set(tuple(order) for order in prev_ob.get('yes', []))
        curr_no = set(tuple(order) for order in curr_ob.get('no', []))
        prev_no = set(tuple(order) for order in prev_ob.get('no', []))
        
        changes = len((curr_yes - prev_yes) | (prev_yes - curr_yes) | (curr_no - prev_no) | (prev_no - curr_no))
        activity_levels.append(min(1.0, changes / 20))  # Normalize to 0-1
    
    # Add a leading zero for the first frame
    activity_levels = [0] + activity_levels
    
    # Calculate overall statistics for the visualization
    all_prices = []
    all_sizes = []
    for ob_data in orderbooks:
        orderbook = ob_data['orderbook']
        for side in ['yes', 'no']:
            for order in orderbook.get(side, []):
                all_prices.append(order[0]/100)
                all_sizes.append(order[1])
    
    # Set global min/max for consistent visuals
    global_min_price = min(all_prices) if all_prices else 0
    global_max_price = max(all_prices) if all_prices else 1
    global_max_size = max(all_sizes) if all_sizes else 1
    
    # Parse market ticker to extract Bitcoin price information
    try:
        # Format is KXBTC-25APR2117-B87250 where:
        # - 25APR2117 is the date and time (April 25, 21:17)
        # - B87250 is the price level (87,250)
        ticker_parts = market_ticker.split('-')
        date_part = ticker_parts[1]
        price_part = ticker_parts[2]
        
        # Extract the price
        price_str = price_part.replace('B', '')
        price_level = int(price_str) / 1000  # Convert to thousands
        
        # Extract the date
        date_str = date_part[:5]  # e.g., "25APR"
        time_str = date_part[5:] if len(date_part) > 5 else ""  # e.g., "2117"
        
        # Format the title
        market_title = f"Bitcoin Price {date_str}"
        if time_str:
            # Format time as HH:MM if available
            if len(time_str) >= 4:
                time_formatted = f"{time_str[:2]}:{time_str[2:4]}"
                market_title += f" at {time_formatted}"
                
        # Format the description
        market_description = f"Will BTC exceed ${price_level:,.0f} threshold?"
    except:
        # Fallback if parsing fails
        market_title = market_ticker.split('-')[0]
        market_description = "Bitcoin Price Prediction"
    
    # Create enhanced frames
    for i, ob_data in enumerate(tqdm(orderbooks)):
        # Create figure with dark theme
        fig = plt.figure(figsize=(14, 10), dpi=100, facecolor='black')
        gs = GridSpec(4, 4, height_ratios=[1, 3, 2, 1])
        
        # Create different axes
        ax_title = plt.subplot(gs[0, :])
        ax_book = plt.subplot(gs[1, :])
        ax_depth = plt.subplot(gs[2, :2])
        ax_price = plt.subplot(gs[2, 2:])
        ax_stats = plt.subplot(gs[3, :])
        
        # Extract orderbook data
        timestamp = ob_data['datetime']
        orderbook = ob_data['orderbook']
        
        # Add animated title with clock and activity indicator
        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%H:%M:%S")
        
        # Display market information
        ax_title.text(0.5, 0.7, market_title, ha='center', va='center', fontsize=20, 
                     color='white', fontweight='bold', transform=ax_title.transAxes)
        ax_title.text(0.5, 0.3, market_description, ha='center', va='center', fontsize=16, 
                     color='#cccccc', transform=ax_title.transAxes)
        
        # Add dynamic date/time display with activity indicator
        activity = activity_levels[i]
        date_color = '#1E90FF'  # Dodger Blue
        time_color = f'#{int(255 * (0.5 + activity/2)):02x}{int(255 * (1-activity)):02x}{int(255 * (1-activity)):02x}'
        
        ax_title.text(0.05, 0.5, date_str, ha='left', va='center', fontsize=16, 
                     color=date_color, transform=ax_title.transAxes)
        ax_title.text(0.95, 0.5, time_str, ha='right', va='center', fontsize=16, 
                     color=time_color, transform=ax_title.transAxes)
        
        # Activity indicator
        ax_title.add_patch(plt.Circle((0.97, 0.5), 0.015, color=time_color, alpha=0.7, transform=ax_title.transAxes))
        ax_title.axis('off')
        
        # Extract yes and no orders
        yes_orders = orderbook.get('yes', [])
        no_orders = orderbook.get('no', [])
        
        # Get the max size for normalization
        all_sizes = []
        if yes_orders:
            all_sizes.extend([order[1] for order in yes_orders])
        if no_orders:
            all_sizes.extend([order[1] for order in no_orders])
            
        max_size = max(all_sizes) if all_sizes else 1
        
        # Calculate mid price
        mid_price = calculate_mid_price(orderbook)
        
        # ENHANCED ORDERBOOK VISUALIZATION
        # Sort orders by price
        yes_orders_sorted = sorted(yes_orders, key=lambda x: x[0])
        no_orders_sorted = sorted(no_orders, key=lambda x: x[0])
        
        # Plot YES orders (on the positive side)
        for order in yes_orders:
            # YES price directly represents probability
            price = order[0] / 100  # Convert cents to probability
            size = min(order[1] / max_size, 1.0)  # Normalize size
            
            # Make rectangles more pronounced with glow effect for larger orders
            alpha = 0.7 + size * 0.3
            edgecolor = 'white' if size > 0.8 else None
            linewidth = 1 if size > 0.8 else 0
            
            rect = plt.Rectangle((price-0.015, 0), 0.03, size,
                               facecolor=yes_cmap(size),
                               alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)
            ax_book.add_patch(rect)
            
            # Add glow effect for large orders
            if size > 0.7:
                glow = plt.Rectangle((price-0.02, -0.05), 0.04, size + 0.1,
                                   facecolor=yes_cmap(size), alpha=0.3, edgecolor=None)
                ax_book.add_patch(glow)
        
        # Plot NO orders (on the negative side)
        for order in no_orders:
            # NO price needs to be transformed to show probability of event happening
            # If NO is priced at 0.70, it means 30% chance of event happening
            price = order[0] / 100  # Convert cents to probability
            inverted_price = 1.0 - price  # Transform to show actual probability
            
            size = min(order[1] / max_size, 1.0)  # Normalize size
            
            # Make rectangles more pronounced with glow effect for larger orders
            alpha = 0.7 + size * 0.3
            edgecolor = 'white' if size > 0.8 else None
            linewidth = 1 if size > 0.8 else 0
            
            # Display NO orders at their inverted price position
            rect = plt.Rectangle((inverted_price-0.015, 0), 0.03, -size,
                               facecolor=no_cmap(size),
                               alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)
            ax_book.add_patch(rect)
            
            # Add glow effect for large orders
            if size > 0.7:
                glow = plt.Rectangle((inverted_price-0.02, -0.05), 0.04, -size - 0.1,
                                   facecolor=no_cmap(size), alpha=0.3, edgecolor=None)
                ax_book.add_patch(glow)
        
        # Add mid price line with animation effect
        if mid_price is not None:
            # Create pulsing effect for the mid price line
            pulse = 0.5 + 0.5 * np.sin(i * np.pi / 10)  # Value between 0.5 and 1
            
            # Main price line
            ax_book.axvline(x=mid_price, color='white', linestyle='-', 
                          linewidth=1.5, alpha=0.9)
            
            # Pulsing overlay
            ax_book.axvline(x=mid_price, color='cyan', linestyle='-', 
                          linewidth=3, alpha=0.2 + 0.3 * pulse)
            
            # Add probability text
            prob_percent = int(round(mid_price * 100))
            prob_text = f"{prob_percent}%"
            ax_book.text(mid_price, 0, prob_text, 
                       color='white', fontsize=14, fontweight='bold', ha='center', va='center',
                       bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5',
                               edgecolor='cyan', linewidth=2))
        
        # Set axis limits
        ax_book.set_xlim(0, 1)  # Probability range
        ax_book.set_ylim(-1.1, 1.1)
        
        # Add market probability labels
        for tick_val in [0.1, 0.25, 0.5, 0.75, 0.9]:
            ax_book.text(tick_val, -1.05, f"{int(tick_val*100)}%", 
                       color='#AAAAAA', fontsize=9, ha='center', va='bottom')
        
        # Customize grid and background
        ax_book.set_facecolor('#0A0A25')  # Dark blue background
        ax_book.grid(True, color='#333355', linestyle='-', linewidth=0.5, alpha=0.5)
        for spine in ax_book.spines.values():
            spine.set_color('#444466')
        ax_book.tick_params(colors='#BBBBBB')
        
        # Add YES/NO labels with better styling
        ax_book.text(0.02, 0.85, "YES", color='#00FF88', fontsize=16, fontweight='bold',
                   ha='left', va='center', transform=ax_book.transAxes,
                   bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
        ax_book.text(0.02, 0.15, "NO", color='#FF3366', fontsize=16, fontweight='bold',
                   ha='left', va='center', transform=ax_book.transAxes,
                   bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
        
        # Add a more descriptive explanation
        if mid_price is not None:
            prob_text = f"There's a {int(round(mid_price * 100))}% chance BTC will exceed ${price_level:,.0f}"
            ax_book.text(0.5, -0.9, prob_text, ha='center', va='center', 
                       fontsize=12, color='white', transform=ax_book.transAxes,
                       bbox=dict(facecolor='#00000080', alpha=0.6, boxstyle='round,pad=0.5'))
        
        # DEPTH CHART
        # Calculate sizes for depth chart
        yes_prices = [order[0]/100 for order in yes_orders_sorted]
        yes_sizes = [order[1] for order in yes_orders_sorted]
        yes_cumulative = np.cumsum(yes_sizes) if yes_sizes else np.array([])
        
        # Transform NO prices to show actual probability
        no_prices_transformed = [1.0 - (order[0]/100) for order in no_orders_sorted]
        no_sizes = [order[1] for order in no_orders_sorted]
        
        # Sort the transformed NO prices and corresponding sizes together
        if no_prices_transformed:
            no_data = sorted(zip(no_prices_transformed, no_sizes))
            no_prices_transformed = [p for p, _ in no_data]
            no_sizes = [s for _, s in no_data]
            no_cumulative = np.cumsum(no_sizes)
        else:
            no_cumulative = np.array([])
        
        # Plot enhanced depth chart
        if yes_prices:
            ax_depth.fill_between(yes_prices, 0, yes_cumulative, color='#00CC66', alpha=0.8, label='YES')
            # Add contour line
            ax_depth.plot(yes_prices, yes_cumulative, color='#00FF88', linewidth=2)
        
        if no_prices_transformed:
            ax_depth.fill_between(no_prices_transformed, 0, no_cumulative, color='#CC3355', alpha=0.8, label='NO')
            # Add contour line
            ax_depth.plot(no_prices_transformed, no_cumulative, color='#FF5577', linewidth=2)
        
        # Add mid price line on depth chart
        if mid_price is not None:
            ax_depth.axvline(x=mid_price, color='white', linestyle='--', alpha=0.7)
        
        # Set title and labels
        ax_depth.set_title("Order Book Depth", color='white', fontsize=14)
        ax_depth.set_xlabel("Probability of BTC > $" + f"{price_level:,.0f}", color='#BBBBBB', fontsize=12)
        ax_depth.set_ylabel("Cumulative Size", color='#BBBBBB', fontsize=12)
        ax_depth.legend(loc='upper right')
        
        # X-axis as percentage
        ax_depth.set_xlim(0, 1)
        ax_depth.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax_depth.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        
        # Style depth chart
        ax_depth.set_facecolor('#0A0A25')
        ax_depth.grid(True, color='#333355', linestyle='-', linewidth=0.5, alpha=0.5)
        for spine in ax_depth.spines.values():
            spine.set_color('#444466')
        ax_depth.tick_params(colors='#BBBBBB')
        
        # PRICE HISTORY
        if not price_df.empty:
            current_time_idx = price_df[price_df['datetime'] <= timestamp].index
            if len(current_time_idx) > 0:
                history_df = price_df.iloc[:current_time_idx[-1] + 1]
                
                # Calculate volatility for color intensity
                if len(history_df) > 5:
                    rolling_std = history_df['price'].rolling(5).std().fillna(0)
                    volatility = rolling_std.iloc[-1] if not rolling_std.empty else 0
                    vol_color = f'#{int(255 * min(1, volatility * 20 + 0.3)):02x}FF{int(255 * (1-min(1, volatility * 10))):02x}'
                else:
                    vol_color = '#AAFFAA'
                
                # Plot price history with gradient line
                ax_price.plot(history_df['datetime'], history_df['price'], 
                            color='cyan', linewidth=2, alpha=0.8)
                
                # Add gradient shading for volatility
                ax_price.fill_between(history_df['datetime'], 
                                     history_df['price'] - 0.02, 
                                     history_df['price'] + 0.02,
                                     color=vol_color, alpha=0.3)
                
                # Highlight current point
                if mid_price is not None:
                    ax_price.scatter([timestamp], [mid_price], 
                                   color='yellow', s=100, zorder=5, edgecolor='black')
                
                ax_price.set_title("Probability History", color='white', fontsize=14)
                ax_price.set_xlabel("Time", color='#BBBBBB', fontsize=12)
                ax_price.set_ylabel("Probability", color='#BBBBBB', fontsize=12)
                ax_price.set_ylim(0, 1)  # Probabilities range from 0 to 1
                
                # Y-axis as percentage
                ax_price.set_yticks([0, 0.25, 0.5, 0.75, 1])
                ax_price.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
                
                # Style price chart
                ax_price.set_facecolor('#0A0A25')
                ax_price.grid(True, color='#333355', linestyle='-', linewidth=0.5, alpha=0.5)
                for spine in ax_price.spines.values():
                    spine.set_color('#444466')
                ax_price.tick_params(colors='#BBBBBB')
        
        # STATISTICS BOX
        # Calculate statistics
        yes_depth = sum(order[1] for order in yes_orders) if yes_orders else 0
        no_depth = sum(order[1] for order in no_orders) if no_orders else 0
        imbalance = yes_depth / no_depth if no_depth > 0 else float('inf')
        
        # Calculate additional useful statistics
        yes_orders_count = len(yes_orders)
        no_orders_count = len(no_orders)
        best_yes_price = max([order[0]/100 for order in yes_orders]) if yes_orders else 0
        best_no_price = max([order[0]/100 for order in no_orders]) if no_orders else 0
        implied_spread = (1.0 - best_no_price) - best_yes_price if yes_orders and no_orders else 0
        
        # Format statistics
        stats_text = [
            f"Market Size: {yes_depth + no_depth:,.0f} contracts",
            f"YES/NO Ratio: {imbalance:.2f}",
            f"Orders: {yes_orders_count} YES, {no_orders_count} NO",
            f"Implied Probability: {int(round(mid_price * 100))}%"
        ]
        
        # Display statistics
        for j, text in enumerate(stats_text):
            ax_stats.text(0.5, 0.8 - j*0.25, text, ha='center', va='center',
                        fontsize=12, color='white', transform=ax_stats.transAxes)
        
        ax_stats.set_facecolor('#0A0A25')
        ax_stats.set_axis_off()
        
        # Apply consistent styling to the entire figure
        plt.tight_layout()
        fig.patch.set_facecolor('#050510')
        
        # Add a subtle animated gradient overlay on the entire figure
        gradient = np.linspace(0, 1, 100)
        gradient = np.vstack((gradient, gradient))
        gradient_rgb = plt.cm.viridis(gradient)
        gradient_rgb[..., 3] = 0.05 * (0.5 + 0.5 * np.sin(i * np.pi / 15))  # Animate alpha
        
        # Save frame
        frame_file = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_file, facecolor=fig.get_facecolor(), dpi=100)
        plt.close(fig)
        
        frame_files.append(frame_file)
    
    # Create video with higher quality
    print(f"Creating enhanced animation: {filename}")
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    with imageio.get_writer(output_path, fps=FPS) as writer:
        for frame_file in frame_files:
            image = imageio.imread(frame_file)
            writer.append_data(image)
    
    print(f"Enhanced animation saved to {output_path}")
    
    # Clean up temporary files
    for frame_file in frame_files:
        os.remove(frame_file)
    os.rmdir(temp_dir)

def create_simplified_animation(market_ticker, orderbooks, filename=None):
    """Create a simplified, visually appealing animation focusing only on the order book and depth"""
    if not orderbooks:
        print(f"No orderbook data for {market_ticker}")
        return
    
    if filename is None:
        filename = f"{market_ticker}_simplified.mp4"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create temporary directory for frames
    temp_dir = os.path.join(OUTPUT_DIR, f"temp_simple_{market_ticker}")
    os.makedirs(temp_dir, exist_ok=True)
    
    frame_files = []
    print(f"Generating {len(orderbooks)} frames for {market_ticker} simplified animation...")
    
    # Create custom colormaps with vibrant colors
    yes_cmap = LinearSegmentedColormap.from_list("YesGradient", [(0.9,1,0.9), (0,0.8,0.4), (0,0.6,0)])
    no_cmap = LinearSegmentedColormap.from_list("NoGradient", [(1,0.9,0.9), (1,0.4,0.4), (0.7,0,0)])
    
    # Parse market ticker to extract price information (if available)
    try:
        # Format is KXBTC-25APR2117-B87250
        ticker_parts = market_ticker.split('-')
        date_part = ticker_parts[1]
        price_part = ticker_parts[2]
        
        # Extract the price
        price_str = price_part.replace('B', '')
        price_level = int(price_str) / 1000
        
        # Format the description
        market_description = f"BTC ${price_level:,.0f} Threshold"
    except:
        market_description = market_ticker
    
    # Extract time series for animation
    price_df = create_price_timeseries(orderbooks)
    
    # Calculate overall statistics for consistent scaling
    all_sizes = []
    for ob_data in orderbooks:
        orderbook = ob_data['orderbook']
        for side in ['yes', 'no']:
            for order in orderbook.get(side, []):
                all_sizes.append(order[1])
    
    max_size = max(all_sizes) if all_sizes else 1
    
    # Create frames
    for i, ob_data in enumerate(tqdm(orderbooks)):
        # Create figure
        fig = plt.figure(figsize=(16, 9), dpi=100, facecolor='#000810')
        gs = GridSpec(3, 1, height_ratios=[1, 6, 3])
        
        # Create axes for different elements
        ax_title = plt.subplot(gs[0])
        ax_book = plt.subplot(gs[1])
        ax_depth = plt.subplot(gs[2])
        
        # Extract orderbook data
        timestamp = ob_data['datetime']
        orderbook = ob_data['orderbook']
        
        # Title with clock
        date_str = timestamp.strftime("%b %d, %Y")
        time_str = timestamp.strftime("%H:%M:%S")
        
        # Create pulsing effect for time
        pulse = 0.5 + 0.5 * np.sin(i * np.pi / 10)
        time_color = f'#{int(150 + 105*pulse):02x}{int(200 + 55*pulse):02x}FF'
        
        # Display title and time
        ax_title.text(0.5, 0.5, market_description, ha='center', va='center', 
                     fontsize=24, fontweight='bold', color='white', transform=ax_title.transAxes)
        ax_title.text(0.98, 0.5, time_str, ha='right', va='center', 
                     fontsize=18, color=time_color, transform=ax_title.transAxes)
        ax_title.text(0.02, 0.5, date_str, ha='left', va='center', 
                     fontsize=18, color='#8899FF', transform=ax_title.transAxes)
        ax_title.axis('off')
        
        # Calculate mid price
        mid_price = calculate_mid_price(orderbook)
        
        # Extract YES and NO orders
        yes_orders = orderbook.get('yes', [])
        no_orders = orderbook.get('no', [])
        
        # Sort orders by price
        yes_orders_sorted = sorted(yes_orders, key=lambda x: x[0])
        no_orders_sorted = sorted(no_orders, key=lambda x: x[0])
        
        # MAIN ORDER BOOK VISUALIZATION
        
        # Create baseline for orders
        ax_book.axhline(y=0, color='#445566', linewidth=2, alpha=0.7)
        
        # Plot YES orders (on top)
        for order in yes_orders:
            # YES price directly represents probability
            price = order[0] / 100  # Convert cents to probability
            size = min(order[1] / max_size, 1.0)  # Normalize size
            
            # Dynamic bar width based on order size
            bar_width = 0.01 + 0.03 * size
            
            # Create a glow effect for large orders
            if size > 0.5:
                glow_size = size * 1.2
                for glow in range(3):
                    alpha_glow = 0.1 - glow * 0.03
                    width_glow = bar_width + glow * 0.01
                    rect = plt.Rectangle((price-width_glow/2, 0), width_glow, glow_size,
                                       facecolor=yes_cmap(size), alpha=alpha_glow, edgecolor=None)
                    ax_book.add_patch(rect)
            
            # The main order bar
            rect = plt.Rectangle((price-bar_width/2, 0), bar_width, size,
                               facecolor=yes_cmap(size), alpha=0.8, 
                               edgecolor='white', linewidth=0.5 if size > 0.7 else 0)
            ax_book.add_patch(rect)
        
        # Plot NO orders (on bottom)
        for order in no_orders:
            # NO price needs to be transformed for visualization
            price = order[0] / 100
            inverted_price = 1.0 - price  # Transform to show actual probability
            size = min(order[1] / max_size, 1.0)  # Normalize size
            
            # Dynamic bar width based on order size
            bar_width = 0.01 + 0.03 * size
            
            # Create a glow effect for large orders
            if size > 0.5:
                glow_size = size * 1.2
                for glow in range(3):
                    alpha_glow = 0.1 - glow * 0.03
                    width_glow = bar_width + glow * 0.01
                    rect = plt.Rectangle((inverted_price-width_glow/2, 0), width_glow, -glow_size,
                                       facecolor=no_cmap(size), alpha=alpha_glow, edgecolor=None)
                    ax_book.add_patch(rect)
            
            # The main order bar
            rect = plt.Rectangle((inverted_price-bar_width/2, 0), bar_width, -size,
                               facecolor=no_cmap(size), alpha=0.8, 
                               edgecolor='white', linewidth=0.5 if size > 0.7 else 0)
            ax_book.add_patch(rect)
        
        # Add mid price line with animation effect
        if mid_price is not None:
            # Create pulsing effect for the mid price line
            pulse = 0.5 + 0.5 * np.sin(i * np.pi / 8)
            
            # Add a glow behind the line
            ax_book.axvline(x=mid_price, color='white', linewidth=2+3*pulse, alpha=0.15)
            ax_book.axvline(x=mid_price, color='#00DDFF', linewidth=2, alpha=0.8)
            
            # Add probability text
            prob_percent = int(round(mid_price * 100))
            prob_text = f"{prob_percent}%"
            
            # Create highlight box
            highlight = plt.Rectangle((mid_price-0.05, -1.5), 0.1, 3, 
                                    facecolor='#001122', alpha=0.6, zorder=0)
            ax_book.add_patch(highlight)
            
            # Add the text
            ax_book.text(mid_price, 0, prob_text, color='white', fontsize=18, 
                        fontweight='bold', ha='center', va='center', zorder=10,
                        bbox=dict(facecolor='#00000080', edgecolor='#00DDFF', 
                                boxstyle='round,pad=0.4', alpha=0.7))
            
            # Add "Implied Probability" label
            ax_book.text(mid_price, 0.5, "Implied\nProbability", color='#AADDFF', 
                        fontsize=9, ha='center', va='bottom', zorder=10)
        
        # Add YES/NO labels
        yes_label_color = '#00FF88'
        no_label_color = '#FF3366'
        
        ax_book.text(0.02, 0.92, "YES", transform=ax_book.transAxes, color=yes_label_color, 
                    fontsize=20, fontweight='bold', ha='left', va='center', 
                    bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))
        
        ax_book.text(0.02, 0.08, "NO", transform=ax_book.transAxes, color=no_label_color, 
                    fontsize=20, fontweight='bold', ha='left', va='center',
                    bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # Customize the look
        ax_book.set_facecolor('#001122')
        ax_book.set_xlim(0, 1)
        ax_book.set_ylim(-1.2, 1.2)
        
        # Add subtle grid lines
        ax_book.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='#4477AA')
        
        # Add probability markers on X-axis
        for tick_val in [0, 0.25, 0.5, 0.75, 1]:
            ax_book.text(tick_val, -1.15, f"{int(tick_val*100)}%", 
                       color='#AAAAAA', fontsize=10, ha='center', va='bottom')
        
        # Remove axis ticks and labels
        ax_book.set_xticks([])
        ax_book.set_yticks([])
        
        # Style the spines
        for spine in ax_book.spines.values():
            spine.set_color('#445566')
            spine.set_linewidth(1.5)
        
        # DEPTH CHART
        
        # Calculate cumulative sizes
        yes_prices = [order[0]/100 for order in yes_orders_sorted]
        yes_sizes = [order[1] for order in yes_orders_sorted]
        yes_cumulative = np.cumsum(yes_sizes) if yes_sizes else np.array([])
        
        # Transform NO prices for visualization
        no_prices_transformed = [1.0 - (order[0]/100) for order in no_orders_sorted]
        no_sizes = [order[1] for order in no_orders_sorted]
        
        # Sort the transformed prices
        if no_prices_transformed:
            no_data = sorted(zip(no_prices_transformed, no_sizes))
            no_prices_transformed = [p for p, _ in no_data]
            no_sizes = [s for _, s in no_data]
            no_cumulative = np.cumsum(no_sizes)
        else:
            no_cumulative = np.array([])
        
        # Plot YES depth
        if len(yes_prices) > 0:
            # Fill area
            ax_depth.fill_between(yes_prices, 0, yes_cumulative, color='#00CC66', alpha=0.5)
            
            # Animate flowing edge line with gradient
            if len(yes_prices) > 1:
                points = np.array([yes_prices, yes_cumulative]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Create a gradient along the line
                z = np.linspace(0, 1, len(segments))
                
                # Add pulse effect
                pulse_phase = (i % 20) / 20
                z = (z + pulse_phase) % 1.0
                
                # Create line collection with gradient
                from matplotlib.collections import LineCollection
                lc = LineCollection(segments, array=z, cmap=yes_cmap, linewidth=3, alpha=0.8)
                ax_depth.add_collection(lc)
        
        # Plot NO depth
        if len(no_prices_transformed) > 0:
            # Fill area
            ax_depth.fill_between(no_prices_transformed, 0, no_cumulative, color='#CC3355', alpha=0.5)
            
            # Animate flowing edge line with gradient
            if len(no_prices_transformed) > 1:
                points = np.array([no_prices_transformed, no_cumulative]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Create a gradient along the line
                z = np.linspace(0, 1, len(segments))
                
                # Add pulse effect (offset from YES to create contrast)
                pulse_phase = ((i + 10) % 20) / 20
                z = (z + pulse_phase) % 1.0
                
                # Create line collection with gradient
                from matplotlib.collections import LineCollection
                lc = LineCollection(segments, array=z, cmap=no_cmap, linewidth=3, alpha=0.8)
                ax_depth.add_collection(lc)
        
        # Add mid price line
        if mid_price is not None:
            ax_depth.axvline(x=mid_price, color='#00DDFF', linestyle='-', linewidth=1.5, alpha=0.8)
        
        # Set depth chart limits
        max_depth = max(np.max(yes_cumulative) if len(yes_cumulative) > 0 else 0,
                      np.max(no_cumulative) if len(no_cumulative) > 0 else 0)
        
        # Add some headroom
        max_depth = max_depth * 1.1 if max_depth > 0 else 100
        
        # Style the depth chart
        ax_depth.set_facecolor('#001122')
        ax_depth.set_xlim(0, 1)
        ax_depth.set_ylim(0, max_depth)
        
        # Add depth chart title
        ax_depth.set_title("Order Book Depth", color='white', fontsize=14)
        
        # Add grid for depth chart
        ax_depth.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='#4477AA')
        
        # Add X-axis as percentage
        ax_depth.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax_depth.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax_depth.tick_params(axis='x', colors='#BBBBBB')
        
        # Y-axis labels
        ax_depth.tick_params(axis='y', colors='#BBBBBB')
        
        # Style the spines
        for spine in ax_depth.spines.values():
            spine.set_color('#445566')
            spine.set_linewidth(1.5)
        
        # Add subtle glow effect to whole figure
        plt.tight_layout()
        
        # Add a subtle animated gradient overlay
        if i % 4 == 0:  # Only add overlay every 4 frames to reduce file size
            gradient = np.linspace(0, 1, 100)
            gradient = np.vstack((gradient, gradient))
            gradient_rgb = plt.cm.viridis(gradient)
            gradient_rgb[..., 3] = 0.03  # Very subtle alpha
            
            # Add diagonal pulse wave
            x = np.linspace(0, 1, 100)
            y = np.linspace(0, 1, 100)
            X, Y = np.meshgrid(x, y)
            Z = 0.5 * (1 + np.sin(10 * (X + Y - i/20)))
            
            # Create overlay effect
            plt.figure(figsize=(16, 9))
            plt.imshow(Z, cmap='Blues', alpha=0.1, aspect='auto', extent=[0, 1, 0, 1])
            plt.axis('off')
        
        # Save the frame
        frame_file = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_file, facecolor='#000810', dpi=100)
        plt.close(fig)
        plt.close('all')  # Close any additional figures
        
        frame_files.append(frame_file)
    
    # Create video
    print(f"Creating simplified animation: {filename}")
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    with imageio.get_writer(output_path, fps=FPS) as writer:
        for frame_file in frame_files:
            image = imageio.imread(frame_file)
            writer.append_data(image)
    
    print(f"Simplified animation saved to {output_path}")
    
    # Clean up temporary files
    for frame_file in frame_files:
        os.remove(frame_file)
    os.rmdir(temp_dir)

def check_database_structure():
    """Check if database exists and has required tables, create if missing"""
    if not os.path.exists(DB_PATH):
        print(f"Database file {DB_PATH} not found.")
        return False
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if required tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='orderbook_deltas'")
    deltas_table_exists = cursor.fetchone() is not None
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='orderbook_snapshots'")
    snapshots_table_exists = cursor.fetchone() is not None
    
    if not deltas_table_exists or not snapshots_table_exists:
        print("Warning: Some required tables don't exist. Checking database structure...")
        
        # Check database structure by listing all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"Available tables: {[t[0] for t in tables]}")
    
    conn.close()
    return True

def main():
    parser = argparse.ArgumentParser(description="Create visualizations of Kalshi orderbook data")
    parser.add_argument("--market", type=str, help="Market ticker to visualize (specific market only)")
    parser.add_argument("--top", type=int, default=5, help="Number of top markets to visualize (by delta count)")
    parser.add_argument("--frames", type=int, default=MAX_FRAMES, help="Number of frames to generate")
    parser.add_argument("--type", choices=["orderbook", "heatmap", "both", "enhanced", "simple"], default="both", 
                       help="Type of visualization to create")
    
    args = parser.parse_args()
    
    # Check database structure first
    if not check_database_structure():
        print("Database structure issue detected. Please ensure the database is properly set up.")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    markets_to_process = []
    
    # If specific market is requested
    if args.market:
        # Handle the specific KXBTC market case
        if args.market == "KXBTC-25APR2117-B87250":
            markets_to_process = [(args.market, None, 0)]
            # Override type to simplified for this specific market
            if args.type not in ["simple", "enhanced"]:
                args.type = "simple"
                print(f"Using simplified visualization for special market: {args.market}")
        else:
            # Process a specific market normally
            markets_to_process = [(args.market, None, 0)]
    else:
        # Process top markets by delta count
        markets_to_process = get_markets_with_most_data(args.top)
    
    if not markets_to_process:
        print("No markets found to process.")
        return
    
    print(f"Processing {len(markets_to_process)} markets:")
    for market, asset, delta_count in markets_to_process:
        print(f" - {market} ({asset if asset else 'Unknown'}) - {delta_count} deltas")
    
    for market_ticker, _, _ in markets_to_process:
        print(f"\nProcessing {market_ticker}...")
        
        # Load market data
        snapshots_df, deltas_df = get_market_data(market_ticker)
        
        if snapshots_df.empty or deltas_df.empty:
            print(f"Insufficient data for {market_ticker}, skipping.")
            continue
        
        print(f"Loaded {len(snapshots_df)} snapshots and {len(deltas_df)} deltas.")
        
        # Generate evenly spaced timestamps
        try:
            min_time = snapshots_df['timestamp_num'].min()
            max_time = max(snapshots_df['timestamp_num'].max(), deltas_df['timestamp_num'].max())
            timestamps = np.linspace(min_time, max_time, min(args.frames, 600), dtype=np.int64)
            
            # Reconstruct orderbooks
            orderbooks = reconstruct_orderbook(snapshots_df, deltas_df, timestamps)
            
            if not orderbooks:
                print(f"Could not reconstruct orderbooks for {market_ticker}, skipping.")
                continue
            
            # Create visualizations based on type
            if args.type == "orderbook" or args.type == "both":
                create_orderbook_animation(market_ticker, orderbooks)
            
            if args.type == "heatmap" or args.type == "both":
                create_heatmap_animation(market_ticker, orderbooks)
            
            if args.type == "enhanced":
                create_enhanced_heatmap_animation(market_ticker, orderbooks)
                
            if args.type == "simple":
                create_simplified_animation(market_ticker, orderbooks)
        except Exception as e:
            print(f"Error processing market {market_ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nAll visualizations saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main() 