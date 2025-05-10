#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import imageio
from matplotlib.colors import LinearSegmentedColormap
import mplfinance as mpf
import plotly.io as pio

# Set page config
st.set_page_config(
    page_title="Polymarket Orderbook Comparison",
    page_icon="📊",
    layout="wide"
)

# Path to the output directory containing parquet files
DATA_DIR = "orderbook_data"
OUTPUT_DIR = "orderbook_animations"
FPS = 30
DURATION_SECONDS = 20  # Total animation duration in seconds
MAX_FRAMES = FPS * DURATION_SECONDS  # Total frames in the animation

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

@st.cache_data(ttl=300)
def get_date_range(candidate):
    """Get min and max dates from the parquet files"""
    # Check both possible locations
    possible_paths = [
        os.path.join(DATA_DIR, f"{candidate}.parquet"),
        os.path.join(DATA_DIR, "metrics", f"{candidate}.parquet"),
        os.path.join(DATA_DIR, f"{candidate.lower()}_metrics.parquet")
    ]
    
    for metrics_file in possible_paths:
        if os.path.exists(metrics_file):
            df = pd.read_parquet(metrics_file)
            
            if not df.empty:
                min_time = df['timestamp'].min()
                max_time = df['timestamp'].max()
                return min_time, max_time
    
    return 0, 0

@st.cache_data(ttl=300)
def get_available_timestamps(candidate):
    """Get all available timestamps for a candidate"""
    # Check both possible locations
    possible_paths = [
        os.path.join(DATA_DIR, f"{candidate}.parquet"),
        os.path.join(DATA_DIR, "metrics", f"{candidate}.parquet"),
        os.path.join(DATA_DIR, f"{candidate.lower()}_metrics.parquet")
    ]
    
    for metrics_file in possible_paths:
        if os.path.exists(metrics_file):
            df = pd.read_parquet(metrics_file)
            return df['timestamp'].tolist()
    
    return []

@st.cache_data(ttl=300)
def get_candidates():
    """Get all candidates from the parquet files"""
    candidates = []
    
    # Look for *.parquet files in all possible locations
    pattern_options = [
        os.path.join(DATA_DIR, "*.parquet"),
        os.path.join(DATA_DIR, "metrics", "*.parquet"),
        os.path.join(DATA_DIR, "*_metrics.parquet")
    ]
    
    for pattern in pattern_options:
        for file in glob.glob(pattern):
            if "_checkpoint_" not in file and "_sampled_" not in file and "_summary" not in file:
                # Extract candidate name from filename
                filename = os.path.basename(file)
                # Handle both naming formats
                if filename.endswith("_metrics.parquet"):
                    candidate = filename.split("_metrics.parquet")[0].capitalize()
                else:
                    candidate = filename.split(".parquet")[0].capitalize()
                
                if candidate not in candidates:
                    candidates.append(candidate)
    
    return candidates

@st.cache_data(ttl=300)
def get_summary_stats():
    """Get summary statistics for all candidates"""
    summary_data = []
    
    candidates = get_candidates()
    
    for candidate in candidates:
        df = get_candidate_data(candidate)
        
        if df.empty:
            continue
        
        # Calculate basic stats
        summary = {
            'candidate': candidate,
            'count': len(df),
            'min_time': df['timestamp'].min(),
            'max_time': df['timestamp'].max(),
            'avg_price': df['mid_price'].mean(),
            'min_price': df['mid_price'].min(),
            'max_price': df['mid_price'].max(),
            'avg_spread': df['spread'].mean(),
            'avg_imbalance': df['imbalance'].mean()
        }
        
        summary_data.append(summary)
    
    return pd.DataFrame(summary_data)

@st.cache_data(ttl=300)
def get_candidate_data(candidate):
    """Get all data for a candidate from any available location"""
    # Check both possible locations
    possible_paths = [
        os.path.join(DATA_DIR, f"{candidate}.parquet"),
        os.path.join(DATA_DIR, "metrics", f"{candidate}.parquet"),
        os.path.join(DATA_DIR, f"{candidate.lower()}_metrics.parquet")
    ]
    
    for metrics_file in possible_paths:
        if os.path.exists(metrics_file):
            st.sidebar.info(f"Using data file: {metrics_file}")
            df = pd.read_parquet(metrics_file)
            # Ensure datetime column exists
            if 'datetime' not in df.columns and 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            return df
    
    # If we get here, no file was found
    st.sidebar.warning(f"No data file found for {candidate}")
    return pd.DataFrame()

@st.cache_data(ttl=300)
def get_price_time_series(candidate, start_time, end_time):
    """Get a time series of prices for a candidate"""
    df = get_candidate_data(candidate)
    
    if df.empty:
        return pd.DataFrame()
    
    # Filter by time range
    df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    return df

def get_orderbook_at_timestamp(candidate, timestamp):
    """Get the orderbook data at a specific timestamp"""
    df = get_candidate_data(candidate)
    
    if df.empty:
        return None
    
    # Get metrics for this timestamp
    metrics_row = df[df['timestamp'] == timestamp]
    
    if metrics_row.empty:
        # Find closest timestamp
        closest_ts = find_nearest_timestamp(df['timestamp'].tolist(), timestamp)
        if closest_ts:
            metrics_row = df[df['timestamp'] == closest_ts]
        
        if metrics_row.empty:
            return None
    
    metrics = metrics_row.iloc[0].to_dict()
    
    # Reconstruct bids and asks using best bid/ask from metrics
    bids = []
    asks = []
    
    # Create synthetic levels based on metrics
    if metrics['best_bid'] and metrics['best_ask']:
        # Create some synthetic levels based on best bid/ask
        best_bid = metrics['best_bid']
        best_ask = metrics['best_ask']
        bid_depth = metrics['bid_depth']
        ask_depth = metrics['ask_depth']
        
        # Create synthetic bids
        for i in range(5):
            price = best_bid * (1 - 0.002 * i)
            size = bid_depth * (0.5 ** i)  # Exponential decay
            bids.append({
                'price': price,
                'size': size,
                'order_count': 1
            })
        
        # Create synthetic asks
        for i in range(5):
            price = best_ask * (1 + 0.002 * i)
            size = ask_depth * (0.5 ** i)  # Exponential decay
            asks.append({
                'price': price,
                'size': size,
                'order_count': 1
            })
    
    return {
        'timestamp': metrics['timestamp'],
        'datetime': datetime.fromtimestamp(metrics['timestamp']),
        'bids': bids,
        'asks': asks,
        'metrics': metrics
    }

def find_nearest_timestamp(timestamps, target_timestamp):
    """Find the nearest timestamp to the target"""
    if not timestamps:
        return None
    
    timestamps = np.array(timestamps)
    idx = np.argmin(np.abs(timestamps - target_timestamp))
    return timestamps[idx]

def plot_orderbook(orderbook_data, title="Orderbook Depth Chart"):
    """Create a depth chart visualization of the orderbook"""
    if not orderbook_data:
        return None
    
    # Extract data
    bids = orderbook_data['bids']
    asks = orderbook_data['asks']
    metrics = orderbook_data['metrics']
    
    # Filter out zero-sized orders
    bids = [b for b in bids if b.get('size', 0) > 0]
    asks = [a for a in asks if a.get('size', 0) > 0]
    
    # Sort bids (highest price first) and asks (lowest price first)
    bids_sorted = sorted(bids, key=lambda x: -x['price'])
    asks_sorted = sorted(asks, key=lambda x: x['price'])
    
    # Calculate cumulative sizes
    bid_prices = [b['price'] for b in bids_sorted]
    bid_sizes = [b['size'] for b in bids_sorted]
    bid_cum_sizes = np.cumsum(bid_sizes) if bid_sizes else np.array([])
    
    ask_prices = [a['price'] for a in asks_sorted]
    ask_sizes = [a['size'] for a in asks_sorted]
    ask_cum_sizes = np.cumsum(ask_sizes) if ask_sizes else np.array([])
    
    # Create depth chart
    fig = go.Figure()
    
    # Add bids
    if bid_prices and len(bid_cum_sizes) > 0:
        fig.add_trace(go.Scatter(
            x=bid_prices,
            y=bid_cum_sizes,
            mode='lines',
            line=dict(width=2, color='green'),
            fill='tozeroy',
            name='Bids'
        ))
    
    # Add asks
    if ask_prices and len(ask_cum_sizes) > 0:
        fig.add_trace(go.Scatter(
            x=ask_prices,
            y=ask_cum_sizes,
            mode='lines',
            line=dict(width=2, color='red'),
            fill='tozeroy',
            name='Asks'
        ))
    
    # Add mid price line
    mid_price = metrics['mid_price']
    if mid_price:
        max_depth = max(np.max(bid_cum_sizes) if len(bid_cum_sizes) > 0 else 0, 
                       np.max(ask_cum_sizes) if len(ask_cum_sizes) > 0 else 0)
        
        # Only add mid price line if we have some depth
        if max_depth > 0:
            fig.add_trace(go.Scatter(
                x=[mid_price, mid_price],
                y=[0, max_depth * 1.1],
                mode='lines',
                line=dict(width=1, color='black', dash='dash'),
                name='Mid Price'
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Price",
        yaxis_title="Cumulative Size",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def plot_price_comparison(dfs, candidates):
    """Create a comparison chart of price time series for multiple candidates"""
    if not dfs or any(df.empty for df in dfs.values()):
        return None
    
    # Create figure
    fig = go.Figure()
    
    colors = {
        'trump': 'red',
        'harris': 'blue'
    }
    
    # Add price lines for each candidate
    for candidate, df in dfs.items():
        if df.empty:
            continue
            
        color = colors.get(candidate.lower(), 'green')
        
        if candidate.lower() == 'harris':
            # For Harris, show the "No" price (1-price) after the election date
            election_date = 1699315200  # Nov 6, 2023 00:00:00 UTC
            df.loc[df['timestamp'] > election_date, 'mid_price'] = 1 - df.loc[df['timestamp'] > election_date, 'mid_price']
            name = f"{candidate} No Price"
        else:
            name = f"{candidate} Price"
        
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['mid_price'],
            mode='lines',
            name=name,
            line=dict(color=color, width=2)
        ))
    
    # Update layout
    fig.update_layout(
        title="Price Comparison",
        xaxis_title="Time",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def plot_imbalance_comparison(dfs, candidates):
    """Create a comparison chart of orderbook imbalance for multiple candidates"""
    if not dfs or any(df.empty for df in dfs.values()):
        return None
    
    # Create figure
    fig = go.Figure()
    
    colors = {
        'trump': 'red',
        'harris': 'blue'
    }
    
    # Add imbalance lines for each candidate
    for candidate, df in dfs.items():
        if df.empty or 'imbalance' not in df.columns:
            continue
            
        color = colors.get(candidate.lower(), 'green')
        
        # Apply rolling average to smooth the imbalance line
        df['imbalance_smooth'] = df['imbalance'].rolling(window=10, min_periods=1).mean()
        
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['imbalance_smooth'],
            mode='lines',
            name=f"{candidate} Imbalance",
            line=dict(color=color, width=2)
        ))
    
    # Add a horizontal line at 1.0 (balanced orderbook)
    fig.add_hline(
        y=1.0,
        line=dict(color="gray", width=1, dash="dash"),
        annotation_text="Balanced",
        annotation_position="bottom right"
    )
    
    # Update layout
    fig.update_layout(
        title="Orderbook Imbalance Comparison (>1 means more bids than asks)",
        xaxis_title="Time",
        yaxis_title="Bid/Ask Imbalance",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def plot_spread_comparison(dfs, candidates):
    """Create a comparison chart of orderbook spread for multiple candidates"""
    if not dfs or any(df.empty for df in dfs.values()):
        return None
    
    # Create figure
    fig = go.Figure()
    
    colors = {
        'trump': 'red',
        'harris': 'blue'
    }
    
    # Add spread lines for each candidate
    for candidate, df in dfs.items():
        if df.empty or 'spread' not in df.columns:
            continue
            
        color = colors.get(candidate.lower(), 'green')
        
        # Apply rolling average to smooth the spread
        df['spread_bps'] = df['spread'] * 10000  # Convert to basis points
        df['spread_smooth'] = df['spread_bps'].rolling(window=10, min_periods=1).mean()
        
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['spread_smooth'],
            mode='lines',
            name=f"{candidate} Spread",
            line=dict(color=color, width=2)
        ))
    
    # Update layout
    fig.update_layout(
        title="Orderbook Spread Comparison (in basis points)",
        xaxis_title="Time",
        yaxis_title="Spread (bps)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def plot_side_by_side_orderbooks(orderbooks, candidates):
    """Create side-by-side orderbook depth charts"""
    if not orderbooks or not all(orderbooks.values()):
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=len(orderbooks),
        subplot_titles=[f"{candidate} Orderbook" for candidate in candidates]
    )
    
    colors = {
        'trump': 'red',
        'harris': 'blue'
    }
    
    # Add traces for each candidate
    for i, (candidate, ob) in enumerate(orderbooks.items(), 1):
        if not ob:
            continue
            
        color = colors.get(candidate.lower(), 'green')
        
        # Extract data
        bids = ob['bids']
        asks = ob['asks']
        metrics = ob['metrics']
        
        # Filter out zero-sized orders
        bids = [b for b in bids if b.get('size', 0) > 0]
        asks = [a for a in asks if a.get('size', 0) > 0]
        
        # Sort bids (highest price first) and asks (lowest price first)
        bids_sorted = sorted(bids, key=lambda x: -x['price'])
        asks_sorted = sorted(asks, key=lambda x: x['price'])
        
        # Calculate cumulative sizes
        bid_prices = [b['price'] for b in bids_sorted]
        bid_sizes = [b['size'] for b in bids_sorted]
        bid_cum_sizes = np.cumsum(bid_sizes) if bid_sizes else np.array([])
        
        ask_prices = [a['price'] for a in asks_sorted]
        ask_sizes = [a['size'] for a in asks_sorted]
        ask_cum_sizes = np.cumsum(ask_sizes) if ask_sizes else np.array([])
        
        # Add bids
        if bid_prices and len(bid_cum_sizes) > 0:
            fig.add_trace(
                go.Scatter(
                    x=bid_prices,
                    y=bid_cum_sizes,
                    mode='lines',
                    line=dict(width=2, color='green'),
                    fill='tozeroy',
                    name=f"{candidate} Bids"
                ),
                row=1, col=i
            )
        
        # Add asks
        if ask_prices and len(ask_cum_sizes) > 0:
            fig.add_trace(
                go.Scatter(
                    x=ask_prices,
                    y=ask_cum_sizes,
                    mode='lines',
                    line=dict(width=2, color='red'),
                    fill='tozeroy',
                    name=f"{candidate} Asks"
                ),
                row=1, col=i
            )
        
        # Add mid price line
        mid_price = metrics['mid_price']
        if mid_price:
            max_depth = max(np.max(bid_cum_sizes) if len(bid_cum_sizes) > 0 else 0, 
                          np.max(ask_cum_sizes) if len(ask_cum_sizes) > 0 else 0)
            
            # Only add mid price line if we have some depth
            if max_depth > 0:
                fig.add_trace(
                    go.Scatter(
                        x=[mid_price, mid_price],
                        y=[0, max_depth * 1.1],
                        mode='lines',
                        line=dict(width=1, color='black', dash='dash'),
                        name=f"{candidate} Mid"
                    ),
                    row=1, col=i
                )
    
    # Update layout
    fig.update_layout(
        height=500,
        margin=dict(l=40, r=40, t=80, b=40),
        showlegend=True
    )
    
    return fig

def check_data_directory():
    """Check if the data directory exists and has parquet files"""
    if not os.path.exists(DATA_DIR):
        return False
    
    # Check for at least one metrics file
    metrics_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    return len(metrics_files) > 0

def create_synthetic_orderbook(metrics_row):
    """Create synthetic orderbook from metrics data"""
    metrics = metrics_row.to_dict()
    
    # Create synthetic levels based on metrics
    bids = []
    asks = []
    
    # Check if we have valid data
    if 'best_bid' in metrics and 'best_ask' in metrics and metrics['best_bid'] > 0 and metrics['best_ask'] > 0:
        best_bid = metrics['best_bid']
        best_ask = metrics['best_ask']
        bid_depth = metrics.get('bid_depth', 50)  # Use default if not available
        ask_depth = metrics.get('ask_depth', 50)  # Use default if not available
        
        # Adjust depth based on imbalance for more realistic visualization
        imbalance = metrics.get('imbalance', 1.0)
        
        # Create synthetic bids - 10 levels
        for i in range(10):
            # Adjust price gap based on spread
            price_gap = 0.002 * (1 + i * 0.2)
            price = best_bid * (1 - price_gap * i)
            # Size decays exponentially, affected by imbalance
            decay_factor = 0.7 - 0.05 * min(imbalance, 3) if imbalance > 1 else 0.7 + 0.05 * min(1/imbalance, 3)
            size = bid_depth * (decay_factor ** i)
            bids.append({
                'price': price,
                'size': size 
            })
        
        # Create synthetic asks - 10 levels
        for i in range(10):
            price_gap = 0.002 * (1 + i * 0.2)
            price = best_ask * (1 + price_gap * i)
            # Size decays exponentially, affected by imbalance
            decay_factor = 0.7 + 0.05 * min(imbalance, 3) if imbalance < 1 else 0.7 - 0.05 * min(1/imbalance, 3)
            size = ask_depth * (decay_factor ** i)
            asks.append({
                'price': price,
                'size': size
            })
    
    return {
        'timestamp': metrics.get('timestamp', 0),
        'datetime': pd.to_datetime(metrics.get('timestamp', 0), unit='s'),
        'bids': bids,
        'asks': asks,
        'metrics': metrics
    }

def get_orderbook_frames(df, num_frames=60):
    """Get evenly spaced orderbook frames from a dataframe"""
    if df.empty:
        return []
    
    # Select evenly spaced timestamps
    indices = np.linspace(0, len(df)-1, num_frames, dtype=int)
    frames = []
    
    for idx in indices:
        ob = create_synthetic_orderbook(df.iloc[idx])
        frames.append(ob)
    
    return frames

def create_heatmap_animation(trump_df, harris_df, filename="orderbook_heatmap.mp4"):
    """Create an advanced heatmap animation of the orderbook with both candidates"""
    if trump_df.empty or harris_df.empty:
        print("Missing data for one or both candidates")
        return
    
    # Ensure datetime columns exist
    if 'datetime' not in trump_df.columns and 'timestamp' in trump_df.columns:
        trump_df['datetime'] = pd.to_datetime(trump_df['timestamp'], unit='s')
    if 'datetime' not in harris_df.columns and 'timestamp' in harris_df.columns:
        harris_df['datetime'] = pd.to_datetime(harris_df['timestamp'], unit='s')
    
    # Determine common time range
    start_time = max(trump_df['timestamp'].min(), harris_df['timestamp'].min())
    end_time = min(trump_df['timestamp'].max(), harris_df['timestamp'].max())
    
    # Filter to common time range
    trump_df = trump_df[(trump_df['timestamp'] >= start_time) & (trump_df['timestamp'] <= end_time)]
    harris_df = harris_df[(harris_df['timestamp'] >= start_time) & (harris_df['timestamp'] <= end_time)]
    
    # Select evenly spaced frames
    indices = np.linspace(0, len(trump_df)-1, min(MAX_FRAMES, len(trump_df)), dtype=int)
    
    # Create temporary directory for frames
    temp_dir = os.path.join(OUTPUT_DIR, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create custom colormaps
    red_cmap = LinearSegmentedColormap.from_list("RedBlue", [(1,0.8,0.8), (1,0,0)])
    blue_cmap = LinearSegmentedColormap.from_list("BlueRed", [(0.8,0.8,1), (0,0,1)])
    
    frame_files = []
    print(f"Generating {len(indices)} frames for heatmap animation...")
    
    for i, idx in enumerate(tqdm(indices)):
        # Get timestamp
        ts = trump_df.iloc[idx]['timestamp']
        dt = pd.to_datetime(ts, unit='s')
        
        # Find closest row in Harris data
        harris_idx = np.argmin(np.abs(harris_df['timestamp'] - ts))
        
        # Create orderbooks
        trump_ob = create_synthetic_orderbook(trump_df.iloc[idx])
        harris_ob = create_synthetic_orderbook(harris_df.iloc[harris_idx])
        
        # Setup the figure with gridspec for complex layout
        fig = plt.figure(figsize=(16, 9), dpi=120, facecolor='black')
        gs = GridSpec(3, 2, height_ratios=[1, 3, 1])
        
        # Axis for title and timestamp
        ax_title = plt.subplot(gs[0, :])
        # Main orderbook heatmaps
        ax_trump = plt.subplot(gs[1, 0])
        ax_harris = plt.subplot(gs[1, 1])
        # Price chart at bottom
        ax_price = plt.subplot(gs[2, :])
        
        # Title and timestamp
        title_text = "Polymarket Election Orderbook Visualization"
        timestamp_text = dt.strftime("%Y-%m-%d %H:%M:%S")
        ax_title.text(0.5, 0.7, title_text, ha='center', va='center', fontsize=22, color='white', 
                    fontweight='bold', transform=ax_title.transAxes)
        ax_title.text(0.5, 0.3, timestamp_text, ha='center', va='center', fontsize=16, color='#bbbbbb',
                    transform=ax_title.transAxes)
        ax_title.axis('off')
        
        # Process Trump orderbook for heatmap
        trump_bid_prices = [b['price'] for b in trump_ob['bids']]
        trump_bid_sizes = [b['size'] for b in trump_ob['bids']]
        trump_ask_prices = [a['price'] for a in trump_ob['asks']]
        trump_ask_sizes = [a['size'] for a in trump_ob['asks']]
        
        # Process Harris orderbook for heatmap
        harris_bid_prices = [b['price'] for b in harris_ob['bids']]
        harris_bid_sizes = [b['size'] for b in harris_ob['bids']]
        harris_ask_prices = [a['price'] for a in harris_ob['asks']]
        harris_ask_sizes = [a['size'] for a in harris_ob['asks']]
        
        # Normalize sizes for visual impact
        max_size = max(max(trump_bid_sizes + trump_ask_sizes), max(harris_bid_sizes + harris_ask_sizes))
        
        # Draw Trump heatmap
        for px, sz in zip(trump_bid_prices, trump_bid_sizes):
            intensity = min(1.0, sz / max_size)
            rect = plt.Rectangle((px-0.002, 0), 0.004, intensity,
                                 facecolor=red_cmap(intensity),
                                 alpha=0.8, edgecolor=None)
            ax_trump.add_patch(rect)
            
        for px, sz in zip(trump_ask_prices, trump_ask_sizes):
            intensity = min(1.0, sz / max_size)
            rect = plt.Rectangle((px-0.002, 0), 0.004, -intensity,
                                 facecolor=red_cmap(intensity),
                                 alpha=0.8, edgecolor=None)
            ax_trump.add_patch(rect)
        
        # Draw Harris heatmap
        for px, sz in zip(harris_bid_prices, harris_bid_sizes):
            intensity = min(1.0, sz / max_size)
            rect = plt.Rectangle((px-0.002, 0), 0.004, intensity,
                                 facecolor=blue_cmap(intensity),
                                 alpha=0.8, edgecolor=None)
            ax_harris.add_patch(rect)
            
        for px, sz in zip(harris_ask_prices, harris_ask_sizes):
            intensity = min(1.0, sz / max_size)
            rect = plt.Rectangle((px-0.002, 0), 0.004, -intensity,
                                 facecolor=blue_cmap(intensity),
                                 alpha=0.8, edgecolor=None)
            ax_harris.add_patch(rect)
        
        # Add midprice lines
        trump_mid = trump_ob['metrics']['mid_price']
        harris_mid = harris_ob['metrics']['mid_price']
        
        ax_trump.axvline(x=trump_mid, color='white', linestyle='--', alpha=0.7)
        ax_harris.axvline(x=harris_mid, color='white', linestyle='--', alpha=0.7)
        
        # Set axis limits
        min_price = min(min(trump_bid_prices), min(harris_bid_prices))
        max_price = max(max(trump_ask_prices), max(harris_ask_prices))
        
        ax_trump.set_xlim(min_price*0.98, max_price*1.02)
        ax_harris.set_xlim(min_price*0.98, max_price*1.02)
        ax_trump.set_ylim(-1.1, 1.1)
        ax_harris.set_ylim(-1.1, 1.1)
        
        # Titles and labels
        ax_trump.set_title("Trump Orderbook", color='white', fontsize=16)
        ax_harris.set_title("Harris Orderbook", color='white', fontsize=16)
        
        # X and Y labels
        ax_trump.set_xlabel("Price", color='white')
        ax_harris.set_xlabel("Price", color='white')
        ax_trump.set_ylabel("Order Depth", color='white')
        
        # Price chart
        # Get historical prices leading up to this point
        cutoff_idx = idx
        historical_trump = trump_df.iloc[:cutoff_idx+1]
        historical_harris = harris_df.iloc[:cutoff_idx+1]
        
        # Only show last 100 points for better visualization
        if len(historical_trump) > 100:
            historical_trump = historical_trump.iloc[-100:]
        if len(historical_harris) > 100:
            historical_harris = historical_harris.iloc[-100:]
        
        ax_price.plot(historical_trump['datetime'], historical_trump['mid_price'], 
                     color='red', linewidth=2, label='Trump')
        ax_price.plot(historical_harris['datetime'], historical_harris['mid_price'], 
                     color='blue', linewidth=2, label='Harris')
        
        # Price chart formatting
        ax_price.set_title("Price History", color='white', fontsize=14)
        ax_price.set_xlabel("Date", color='white')
        ax_price.set_ylabel("Price", color='white')
        ax_price.legend(loc='upper left', frameon=False)
        
        # Set grid and background for all axes
        for ax in [ax_trump, ax_harris, ax_price]:
            ax.set_facecolor('black')
            ax.grid(True, color='#333333', linestyle='-', linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_color('#444444')
            ax.tick_params(colors='white')
        
        # Add bid/ask labels on y-axis
        ax_trump.text(-0.05, 0.75, "BIDS", transform=ax_trump.transAxes, 
                     color='white', ha='right', va='center', fontsize=10)
        ax_trump.text(-0.05, 0.25, "ASKS", transform=ax_trump.transAxes, 
                     color='white', ha='right', va='center', fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save frame
        frame_file = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_file, facecolor='black')
        plt.close()
        
        frame_files.append(frame_file)
    
    # Create video
    print(f"Creating animation: {filename}")
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    # Create video using imageio
    with imageio.get_writer(output_path, fps=FPS) as writer:
        for frame_file in frame_files:
            image = imageio.imread(frame_file)
            writer.append_data(image)
    
    print(f"Animation saved to {output_path}")
    
    # Clean up temporary files
    for frame_file in frame_files:
        os.remove(frame_file)
    os.rmdir(temp_dir)

def create_orderbook_animation(candidate, df, filename=None):
    """Create an animation of the orderbook for a single candidate"""
    if df.empty:
        print(f"No data for {candidate}")
        return
    
    # Ensure datetime column exists
    if 'datetime' not in df.columns and 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    if filename is None:
        filename = f"{candidate.lower()}_orderbook.mp4"
    
    frames = get_orderbook_frames(df, num_frames=MAX_FRAMES)
    
    # Create temporary directory for frames
    temp_dir = os.path.join(OUTPUT_DIR, f"temp_{candidate}")
    os.makedirs(temp_dir, exist_ok=True)
    
    frame_files = []
    print(f"Generating {len(frames)} frames for {candidate} animation...")
    
    # Define colors based on candidate
    if candidate.lower() == 'trump':
        bid_color = 'red'
        ask_color = '#ff9999'  # Light red
        title_color = '#ff5555'
    else:
        bid_color = 'blue'
        ask_color = '#9999ff'  # Light blue
        title_color = '#5555ff'
    
    for i, ob in enumerate(tqdm(frames)):
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1]})
        
        # Extract orderbook data
        timestamp = ob['datetime']
        metrics = ob['metrics']
        bids = ob['bids']
        asks = ob['asks']
        
        # Sort bids (descending) and asks (ascending)
        bids_sorted = sorted(bids, key=lambda x: -x['price'])
        asks_sorted = sorted(asks, key=lambda x: x['price'])
        
        # Calculate cumulative sizes
        bid_prices = [b['price'] for b in bids_sorted]
        bid_sizes = [b['size'] for b in bids_sorted]
        bid_cum_sizes = np.cumsum(bid_sizes)
        
        ask_prices = [a['price'] for a in asks_sorted]
        ask_sizes = [a['size'] for a in asks_sorted]
        ask_cum_sizes = np.cumsum(ask_sizes)
        
        # Plot orderbook depth
        ax1.fill_between(bid_prices, 0, bid_cum_sizes, color=bid_color, alpha=0.7, step='post', label='Bids')
        ax1.fill_between(ask_prices, 0, ask_cum_sizes, color=ask_color, alpha=0.7, step='post', label='Asks')
        
        # Add mid price line
        mid_price = metrics['mid_price']
        ax1.axvline(x=mid_price, color='black', linestyle='--', alpha=0.7, label=f'Mid: ${mid_price:.4f}')
        
        # Add 3D effect with additional data
        for j in range(min(len(bid_prices), len(ask_prices))):
            if j < len(bid_prices):
                price = bid_prices[j]
                size = bid_cum_sizes[j] if j < len(bid_cum_sizes) else 0
                alpha = max(0.1, 1 - j * 0.1)
                ax1.plot([price, price], [0, size], color=bid_color, alpha=alpha, linewidth=1)
            
            if j < len(ask_prices):
                price = ask_prices[j]
                size = ask_cum_sizes[j] if j < len(ask_cum_sizes) else 0
                alpha = max(0.1, 1 - j * 0.1)
                ax1.plot([price, price], [0, size], color=ask_color, alpha=alpha, linewidth=1)
        
        # Set title and labels
        ax1.set_title(f"{candidate} Orderbook - {timestamp}", fontsize=16)
        ax1.set_xlabel("Price ($)", fontsize=12)
        ax1.set_ylabel("Cumulative Size", fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Find reasonable y-axis limits
        max_depth = max(max(bid_cum_sizes) if bid_cum_sizes.size > 0 else 0, 
                        max(ask_cum_sizes) if ask_cum_sizes.size > 0 else 0)
        ax1.set_ylim(0, max_depth * 1.1)
        
        # Set reasonable x-axis limits
        spread = metrics.get('spread', 0.01)
        ax1.set_xlim(mid_price - spread * 15, mid_price + spread * 15)
        
        # Plot price time series up to this point
        historical_idx = df[df['timestamp'] <= ob['timestamp']].index
        if len(historical_idx) > 0:
            # Limit to most recent 200 points for better visualization
            if len(historical_idx) > 200:
                historical_idx = historical_idx[-200:]
            
            historical_data = df.loc[historical_idx]
            ax2.plot(historical_data['datetime'], historical_data['mid_price'], 
                     color=title_color, linewidth=2)
            
            # Mark current point
            ax2.scatter([timestamp], [mid_price], color='white', edgecolor='black', s=100, zorder=5)
            
            # Set title and labels
            ax2.set_title("Price History", fontsize=14)
            ax2.set_xlabel("Date", fontsize=12)
            ax2.set_ylabel("Price ($)", fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Set y-axis limits with padding
            y_range = historical_data['mid_price'].max() - historical_data['mid_price'].min()
            if y_range > 0:
                ax2.set_ylim(
                    historical_data['mid_price'].min() - y_range * 0.1,
                    historical_data['mid_price'].max() + y_range * 0.1
                )
        
        # Add metrics display
        metrics_text = f"Spread: {metrics['spread']*10000:.1f} bps | "
        metrics_text += f"Imbalance: {metrics['imbalance']:.2f} | "
        metrics_text += f"Bid Depth: {metrics['bid_depth']:.1f} | "
        metrics_text += f"Ask Depth: {metrics['ask_depth']:.1f}"
        
        fig.suptitle(metrics_text, fontsize=12, y=0.03)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save frame
        frame_file = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_file, dpi=100)
        plt.close(fig)
        
        frame_files.append(frame_file)
    
    # Create video
    print(f"Creating animation: {filename}")
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    # Create video using imageio
    with imageio.get_writer(output_path, fps=FPS) as writer:
        for frame_file in frame_files:
            image = imageio.imread(frame_file)
            writer.append_data(image)
    
    print(f"Animation saved to {output_path}")
    
    # Clean up temporary files
    for frame_file in frame_files:
        os.remove(frame_file)
    os.rmdir(temp_dir)

def create_combined_orderbook_animation(trump_df, harris_df, filename="combined_orderbook.mp4"):
    """Create a combined animation showing both orderbooks side by side"""
    if trump_df.empty or harris_df.empty:
        print("Missing data for one or both candidates")
        return
    
    # Ensure datetime columns exist
    if 'datetime' not in trump_df.columns and 'timestamp' in trump_df.columns:
        trump_df['datetime'] = pd.to_datetime(trump_df['timestamp'], unit='s')
    if 'datetime' not in harris_df.columns and 'timestamp' in harris_df.columns:
        harris_df['datetime'] = pd.to_datetime(harris_df['timestamp'], unit='s')
    
    # Determine common time range
    start_time = max(trump_df['timestamp'].min(), harris_df['timestamp'].min())
    end_time = min(trump_df['timestamp'].max(), harris_df['timestamp'].max())
    
    # Filter to common time range
    trump_df = trump_df[(trump_df['timestamp'] >= start_time) & (trump_df['timestamp'] <= end_time)]
    harris_df = harris_df[(harris_df['timestamp'] >= start_time) & (harris_df['timestamp'] <= end_time)]
    
    # Select evenly spaced frames from Trump data
    indices = np.linspace(0, len(trump_df)-1, min(MAX_FRAMES, len(trump_df)), dtype=int)
    
    # Create temporary directory for frames
    temp_dir = os.path.join(OUTPUT_DIR, "temp_combined")
    os.makedirs(temp_dir, exist_ok=True)
    
    frame_files = []
    print(f"Generating {len(indices)} frames for combined animation...")
    
    for i, idx in enumerate(tqdm(indices)):
        # Get timestamp for this frame
        ts = trump_df.iloc[idx]['timestamp']
        dt = pd.to_datetime(ts, unit='s')
        
        # Find closest match in Harris data
        harris_idx = np.argmin(np.abs(harris_df['timestamp'] - ts))
        
        # Create orderbooks
        trump_ob = create_synthetic_orderbook(trump_df.iloc[idx])
        harris_ob = create_synthetic_orderbook(harris_df.iloc[harris_idx])
        
        # Create figure for side-by-side visualizations
        fig = plt.figure(figsize=(16, 9), dpi=120)
        gs = GridSpec(2, 2, height_ratios=[3, 1])
        
        # Setup axes
        ax_trump = plt.subplot(gs[0, 0])
        ax_harris = plt.subplot(gs[0, 1])
        ax_price = plt.subplot(gs[1, :])
        
        # Process Trump orderbook data
        trump_bids = sorted(trump_ob['bids'], key=lambda x: -x['price'])
        trump_asks = sorted(trump_ob['asks'], key=lambda x: x['price'])
        
        trump_bid_prices = [b['price'] for b in trump_bids]
        trump_bid_sizes = [b['size'] for b in trump_bids]
        trump_bid_cum_sizes = np.cumsum(trump_bid_sizes)
        
        trump_ask_prices = [a['price'] for a in trump_asks]
        trump_ask_sizes = [a['size'] for a in trump_asks]
        trump_ask_cum_sizes = np.cumsum(trump_ask_sizes)
        
        # Process Harris orderbook data
        harris_bids = sorted(harris_ob['bids'], key=lambda x: -x['price'])
        harris_asks = sorted(harris_ob['asks'], key=lambda x: x['price'])
        
        harris_bid_prices = [b['price'] for b in harris_bids]
        harris_bid_sizes = [b['size'] for b in harris_bids]
        harris_bid_cum_sizes = np.cumsum(harris_bid_sizes)
        
        harris_ask_prices = [a['price'] for a in harris_asks]
        harris_ask_sizes = [a['size'] for a in harris_asks]
        harris_ask_cum_sizes = np.cumsum(harris_ask_sizes)
        
        # Plot Trump orderbook
        ax_trump.fill_between(trump_bid_prices, 0, trump_bid_cum_sizes, color='red', alpha=0.7, step='post', label='Bids')
        ax_trump.fill_between(trump_ask_prices, 0, trump_ask_cum_sizes, color='#ff9999', alpha=0.7, step='post', label='Asks')
        
        # Add mid price line for Trump
        trump_mid = trump_ob['metrics']['mid_price']
        ax_trump.axvline(x=trump_mid, color='black', linestyle='--', alpha=0.7)
        
        # Plot Harris orderbook
        ax_harris.fill_between(harris_bid_prices, 0, harris_bid_cum_sizes, color='blue', alpha=0.7, step='post', label='Bids')
        ax_harris.fill_between(harris_ask_prices, 0, harris_ask_cum_sizes, color='#9999ff', alpha=0.7, step='post', label='Asks')
        
        # Add mid price line for Harris
        harris_mid = harris_ob['metrics']['mid_price']
        ax_harris.axvline(x=harris_mid, color='black', linestyle='--', alpha=0.7)
        
        # Set titles and labels
        ax_trump.set_title(f"Trump: ${trump_mid:.4f} (Imb: {trump_ob['metrics']['imbalance']:.2f})", fontsize=14)
        ax_harris.set_title(f"Harris: ${harris_mid:.4f} (Imb: {harris_ob['metrics']['imbalance']:.2f})", fontsize=14)
        
        ax_trump.set_xlabel("Price ($)", fontsize=12)
        ax_harris.set_xlabel("Price ($)", fontsize=12)
        ax_harris.set_ylabel("Cumulative Size", fontsize=12)
        ax_harris.set_ylabel("Cumulative Size", fontsize=12)
        
        # Add legends
        ax_trump.legend(loc='upper right')
        ax_harris.legend(loc='upper right')
        
        # Enable grid
        ax_trump.grid(True, alpha=0.3)
        ax_harris.grid(True, alpha=0.3)
        
        # Set y-axis limits
        max_depth_trump = max(max(trump_bid_cum_sizes) if trump_bid_cum_sizes.size > 0 else 0, 
                             max(trump_ask_cum_sizes) if trump_ask_cum_sizes.size > 0 else 0)
        max_depth_harris = max(max(harris_bid_cum_sizes) if harris_bid_cum_sizes.size > 0 else 0, 
                              max(harris_ask_cum_sizes) if harris_ask_cum_sizes.size > 0 else 0)
        
        max_depth = max(max_depth_trump, max_depth_harris)
        
        ax_trump.set_ylim(0, max_depth * 1.1)
        ax_harris.set_ylim(0, max_depth * 1.1)
        
        # Set x-axis limits
        trump_spread = trump_ob['metrics'].get('spread', 0.01)
        harris_spread = harris_ob['metrics'].get('spread', 0.01)
        
        ax_trump.set_xlim(trump_mid - trump_spread * 15, trump_mid + trump_spread * 15)
        ax_harris.set_xlim(harris_mid - harris_spread * 15, harris_mid + harris_spread * 15)
        
        # Plot price history
        cutoff_idx = idx
        historical_trump = trump_df.iloc[:cutoff_idx+1]
        historical_harris = harris_df.iloc[:harris_idx+1]
        
        # Only show last 200 points for better visualization
        if len(historical_trump) > 200:
            historical_trump = historical_trump.iloc[-200:]
        if len(historical_harris) > 200:
            historical_harris = historical_harris.iloc[-200:]
        
        ax_price.plot(historical_trump['datetime'], historical_trump['mid_price'], 
                     color='red', linewidth=2, label='Trump')
        ax_price.plot(historical_harris['datetime'], historical_harris['mid_price'], 
                     color='blue', linewidth=2, label='Harris')
        
        # Mark current points
        ax_price.scatter([dt], [trump_mid], color='red', edgecolor='black', s=80, zorder=5)
        ax_price.scatter([dt], [harris_mid], color='blue', edgecolor='black', s=80, zorder=5)
        
        # Price chart formatting
        ax_price.set_title("Price History", fontsize=14)
        ax_price.set_xlabel("Date", fontsize=12)
        ax_price.set_ylabel("Price ($)", fontsize=12)
        ax_price.grid(True, alpha=0.3)
        ax_price.legend(loc='upper left')
        
        # Master title with timestamp
        plt.suptitle(f"Presidential Election Orderbook - {dt.strftime('%Y-%m-%d %H:%M:%S')}", 
                     fontsize=16, y=0.98)
        
        # Add candidate faces (emoji placeholders)
        fig.text(0.01, 0.98, "🔴", fontsize=20)  # Trump emoji placeholder
        fig.text(0.5, 0.98, "🔵", fontsize=20)   # Harris emoji placeholder
        
        # Add metrics display
        trump_metrics = trump_ob['metrics']
        harris_metrics = harris_ob['metrics']
        
        metrics_text = f"Trump Spread: {trump_metrics['spread']*10000:.1f} bps | "
        metrics_text += f"Harris Spread: {harris_metrics['spread']*10000:.1f} bps | "
        metrics_text += f"Price Diff: {abs(trump_mid - harris_mid)*100:.2f}%"
        
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=12)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save frame
        frame_file = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_file)
        plt.close()
        
        frame_files.append(frame_file)
    
    # Create video
    print(f"Creating animation: {filename}")
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    # Create video using imageio
    with imageio.get_writer(output_path, fps=FPS) as writer:
        for frame_file in frame_files:
            image = imageio.imread(frame_file)
            writer.append_data(image)
    
    print(f"Animation saved to {output_path}")
    
    # Clean up temporary files
    for frame_file in frame_files:
        os.remove(frame_file)
    os.rmdir(temp_dir)

def create_3d_orderbook_animation(df, candidate, filename=None):
    """Create a 3D animation of the orderbook evolution using matplotlib"""
    if df.empty:
        print(f"No data for {candidate}")
        return
    
    # Ensure datetime column exists
    if 'datetime' not in df.columns and 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    if filename is None:
        filename = f"{candidate.lower()}_3d_orderbook.mp4"
    
    # Select evenly spaced frames
    indices = np.linspace(0, len(df)-1, min(MAX_FRAMES, len(df)), dtype=int)
    
    # Create temporary directory for frames
    temp_dir = os.path.join(OUTPUT_DIR, f"temp_3d_{candidate}")
    os.makedirs(temp_dir, exist_ok=True)
    
    frame_files = []
    print(f"Generating {len(indices)} frames for {candidate} 3D animation...")
    
    # Determine plot colors based on candidate
    if candidate.lower() == 'trump':
        main_color = 'red'
        light_color = '#ffcccc'
        else:
        main_color = 'blue'
        light_color = '#ccccff'
    
    # Initialize price history for 3D surface
    time_data = []
    price_data = []
    bid_volumes = []
    ask_volumes = []
    
    for i, idx in enumerate(tqdm(indices)):
        # Create orderbook for this timestamp
        ob = create_synthetic_orderbook(df.iloc[idx])
        dt = ob['datetime']
        
        # Extract bid and ask data
        for bid in ob['bids']:
            time_data.append(i)
            price_data.append(bid['price'])
            bid_volumes.append(bid['size'])
            
        for ask in ob['asks']:
            time_data.append(i)
            price_data.append(ask['price'])
            ask_volumes.append(-ask['size'])  # Negative for asks
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Separate bids and asks
        bid_points = np.array(list(zip(time_data, price_data, bid_volumes)))
        ask_points = np.array(list(zip(time_data, price_data, [-abs(v) for v in ask_volumes])))
        
        # Filter to only current time point
        current_bid_points = bid_points[bid_points[:, 0] <= i]
        current_ask_points = ask_points[ask_points[:, 0] <= i]
        
        # Plot 3D scatter
        if len(current_bid_points) > 0:
            ax.scatter(current_bid_points[:, 0], current_bid_points[:, 1], current_bid_points[:, 2],
                      color=main_color, alpha=0.6, label='Bids')
            
        if len(current_ask_points) > 0:
            ax.scatter(current_ask_points[:, 0], current_ask_points[:, 1], current_ask_points[:, 2],
                      color=light_color, alpha=0.6, label='Asks')
            
        # Add mid price line
        mid_prices = df.iloc[indices[:i+1]]['mid_price'].values
        time_points = list(range(i+1))
        zero_points = [0] * len(time_points)
        if len(time_points) > 0:
            ax.plot(time_points, mid_prices, zero_points, color='black', linewidth=2, label='Mid Price')
        
        # Add current price marker
        current_mid = ob['metrics']['mid_price']
        ax.scatter([i], [current_mid], [0], color='yellow', s=100, marker='o', 
                   edgecolor='black', linewidth=1, label='Current Price')
        
        # Set labels and title
        ax.set_title(f"{candidate} Orderbook Evolution - {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.set_zlabel('Depth (Bid/Ask)')
        
        # Set the same y-axis (price) range for all frames for consistency
        all_prices = np.array(price_data)
        min_price, max_price = all_prices.min(), all_prices.max()
        ax.set_ylim(min_price, max_price)
        
        # Add colorbar legend
        ax.legend()
        
        # Set the view angle
        ax.view_init(elev=30, azim=i % 360)  # Rotate view for animation effect
        
        # Save frame
        frame_file = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_file, dpi=100)
        plt.close()
        
        frame_files.append(frame_file)
    
    # Create video
    print(f"Creating animation: {filename}")
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    # Create video using imageio
    with imageio.get_writer(output_path, fps=FPS) as writer:
        for frame_file in frame_files:
            image = imageio.imread(frame_file)
            writer.append_data(image)
    
    print(f"Animation saved to {output_path}")
    
    # Clean up temporary files
    for frame_file in frame_files:
        os.remove(frame_file)
    os.rmdir(temp_dir)

def animation_exists(filename):
    """Check if an animation file already exists"""
    full_path = os.path.join(OUTPUT_DIR, filename)
    exists = os.path.exists(full_path)
    if exists:
        print(f"Animation already exists: {full_path}")
    return exists

def convert_frames_to_mp4(temp_dir, output_filename):
    """Convert existing PNG frames to an MP4 file"""
    if not os.path.exists(temp_dir):
        print(f"Temp directory not found: {temp_dir}")
        return False
    
    # Get all PNG files in the directory
    frame_files = sorted(glob.glob(os.path.join(temp_dir, "frame_*.png")))
    
    if not frame_files:
        print(f"No frames found in {temp_dir}")
        return False
    
    print(f"Found {len(frame_files)} frames in {temp_dir}")
    print(f"Converting frames to MP4: {output_filename}")
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Create video using imageio
    with imageio.get_writer(output_path, fps=FPS) as writer:
        for frame_file in tqdm(frame_files):
            image = imageio.imread(frame_file)
            writer.append_data(image)
    
    print(f"Animation saved to {output_path}")
    
    # Don't clean up by default - let the user manually clean up if desired
    # for frame_file in frame_files:
    #     os.remove(frame_file)
    # os.rmdir(temp_dir)
    
    return True

def cleanup_temp_directory(temp_dir):
    """Clean up a temporary directory after successful conversion"""
    if not os.path.exists(temp_dir):
        return
    
    # Ask for confirmation before cleaning up
    print(f"\nWould you like to clean up temporary directory? {temp_dir}")
    confirm = input("Delete temporary frames? (y/n): ").lower().strip()
    
    if confirm in ('y', 'yes'):
        frame_files = glob.glob(os.path.join(temp_dir, "frame_*.png"))
        for frame_file in frame_files:
            os.remove(frame_file)
        os.rmdir(temp_dir)
        print(f"Cleaned up {len(frame_files)} temporary files.")
                else:
        print("Temporary files kept for future use.")

def check_and_convert_incomplete_animations():
    """Check for incomplete animations and convert them to MP4"""
    # Check for each possible temp directory
    candidates = ["Trump", "Harris", "combined", "frames", "3d_Trump", "3d_Harris"]
    
    found_any = False
    
    for candidate in candidates:
        temp_dir = os.path.join(OUTPUT_DIR, f"temp_{candidate}")
        if os.path.exists(temp_dir):
            found_any = True
            
            if candidate == "frames":
                output_filename = "orderbook_heatmap.mp4"
            elif candidate.startswith("3d_"):
                c_name = candidate.split("_")[1]
                output_filename = f"{c_name.lower()}_3d_orderbook.mp4"
            elif candidate == "combined":
                output_filename = "combined_orderbook.mp4"
            else:
                output_filename = f"{candidate.lower()}_orderbook.mp4"
            
            print(f"\nFound incomplete animation: {candidate}")
            
            if animation_exists(output_filename):
                print(f"MP4 file already exists: {output_filename}")
                print(f"Do you want to regenerate it from the frames?")
                regen = input("Regenerate MP4? (y/n): ").lower().strip()
                
                if regen in ('y', 'yes'):
                    convert_frames_to_mp4(temp_dir, output_filename)
                    cleanup_temp_directory(temp_dir)
            else:
                print(f"Converting frames to MP4: {output_filename}")
                convert_frames_to_mp4(temp_dir, output_filename)
                cleanup_temp_directory(temp_dir)
    
    if not found_any:
        print("No incomplete animations found.")
    
    return found_any

def main():
    print("Loading data...")
    trump_df = get_candidate_data("Trump")
    harris_df = get_candidate_data("Harris")
    
    if trump_df.empty or harris_df.empty:
        print("Error: Could not load data for one or both candidates")
        return
    
    # Ensure datetime columns exist
    print("Preparing data...")
    if 'datetime' not in trump_df.columns and 'timestamp' in trump_df.columns:
        trump_df['datetime'] = pd.to_datetime(trump_df['timestamp'], unit='s')
    if 'datetime' not in harris_df.columns and 'timestamp' in harris_df.columns:
        harris_df['datetime'] = pd.to_datetime(harris_df['timestamp'], unit='s')
    
    # These animations have already been created, comment them out
    # print("\nCreating Trump orderbook animation...")
    # create_orderbook_animation("Trump", trump_df)
    
    # print("\nCreating Harris orderbook animation...")
    # create_orderbook_animation("Harris", harris_df)
    
    # print("\nCreating combined orderbook animation...")
    # create_combined_orderbook_animation(trump_df, harris_df)
    
    # print("\nCreating heatmap animation...")
    # create_heatmap_animation(trump_df, harris_df)
    
    # Only run the remaining animations that haven't been created yet
    print("\nCreating 3D Trump orderbook animation...")
    create_3d_orderbook_animation(trump_df, "Trump")
    
    print("\nCreating 3D Harris orderbook animation...")
    create_3d_orderbook_animation(harris_df, "Harris")
    
    print("\nAll animations created successfully!")
    print(f"Animations saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main() 