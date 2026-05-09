import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="Trump Orderbook Visualizer")

# Load Trump data
@st.cache_data
def load_data():
    df = pd.read_parquet('orderbook_data/metrics/Trump.parquet')
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    df_unique = df_sorted.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    return df_unique

st.title("Trump Prediction Market Orderbook Visualizer")

with st.spinner('Loading orderbook data...'):
    df_unique = load_data()

st.success(f"Loaded {len(df_unique)} unique timestamps from {datetime.fromtimestamp(df_unique['timestamp'].min())} to {datetime.fromtimestamp(df_unique['timestamp'].max())}")

# Sidebar controls
st.sidebar.header("Visualization Controls")

# Date range selection
min_date = datetime.fromtimestamp(df_unique['timestamp'].min()).date()
max_date = datetime.fromtimestamp(df_unique['timestamp'].max()).date()

start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

# Convert selected dates to timestamps
start_ts = datetime.combine(start_date, datetime.min.time()).timestamp()
end_ts = datetime.combine(end_date, datetime.max.time()).timestamp()

# Filter the dataframe by selected date range
df_filtered = df_unique[(df_unique['timestamp'] >= start_ts) & (df_unique['timestamp'] <= end_ts)]

if len(df_filtered) == 0:
    st.warning("No data available for the selected date range.")
    st.stop()

st.sidebar.write(f"Selected range has {len(df_filtered)} data points")

# Choose visualization mode
viz_mode = st.sidebar.radio(
    "Visualization Mode",
    ["Static View", "Interactive Exploration", "Time Slider", "Backtest Simulation"]
)

if viz_mode == "Static View":
    st.header("Current Orderbook Snapshot")

    # Let user select a specific timestamp or use the latest
    use_latest = st.sidebar.checkbox("Use latest timestamp", value=True)
    
    if use_latest:
        selected_idx = len(df_filtered) - 1
    else:
        # Create a list of timestamps for selection
        timestamps = [datetime.fromtimestamp(ts) for ts in df_filtered['timestamp']]
        selected_time = st.sidebar.selectbox("Select timestamp", timestamps, index=len(timestamps)-1)
        selected_idx = timestamps.index(selected_time)
    
    # Get data for the selected timestamp
    data = df_filtered.iloc[selected_idx]
    timestamp = datetime.fromtimestamp(data['timestamp'])
    
    # Display timestamp and key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Timestamp", timestamp.strftime("%Y-%m-%d %H:%M:%S"))
    col2.metric("Mid Price", f"${data['mid_price']:.4f}")
    col3.metric("Spread", f"${data['spread']:.4f}")
    col4.metric("Imbalance", f"{data['imbalance']:.4f}")
    
    # Create a figure with matplotlib
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2)
    
    # Bid side
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(0, data['bid_depth'], color='green', alpha=0.7)
    ax1.set_title(f"Bid (${data['best_bid']:.4f})")
    ax1.set_xlim(0, max(data['bid_depth'], data['ask_depth']) * 1.1)
    ax1.set_ylabel("Price")
    ax1.set_xlabel("Depth")
    ax1.invert_xaxis()  # Invert x-axis for bid side
    ax1.set_yticks([])
    
    # Ask side
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(0, data['ask_depth'], color='red', alpha=0.7)
    ax2.set_title(f"Ask (${data['best_ask']:.4f})")
    ax2.set_xlim(0, max(data['bid_depth'], data['ask_depth']) * 1.1)
    ax2.set_yticks([])
    ax2.set_xlabel("Depth")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Additional metrics
    with st.expander("View detailed metrics"):
        st.write(data)

elif viz_mode == "Interactive Exploration":
    st.header("Orderbook Price & Depth History")
    
    # Sample rate for plotting
    sample_rate = st.sidebar.slider("Sampling Rate (1 = all data points)", 1, 1000, 100)
    df_sampled = df_filtered.iloc[::sample_rate].reset_index(drop=True)
    
    # Create interactive plotly figure
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=("Price History", "Depth History"),
                        row_heights=[0.7, 0.3])
    
    # Convert timestamps to datetime for plotting
    dates = [datetime.fromtimestamp(ts) for ts in df_sampled['timestamp']]
    
    # Add price traces
    fig.add_trace(go.Scatter(x=dates, y=df_sampled['mid_price'], mode='lines', name='Mid Price', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df_sampled['best_bid'], mode='lines', name='Best Bid', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df_sampled['best_ask'], mode='lines', name='Best Ask', line=dict(color='red')), row=1, col=1)
    
    # Add depth traces
    fig.add_trace(go.Scatter(x=dates, y=df_sampled['bid_depth'], mode='lines', name='Bid Depth', line=dict(color='green', dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df_sampled['ask_depth'], mode='lines', name='Ask Depth', line=dict(color='red', dash='dot')), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis2_title="Date",
        yaxis_title="Price",
        yaxis2_title="Depth"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Imbalance chart
    st.subheader("Bid/Ask Imbalance")
    imb_fig = go.Figure()
    imb_fig.add_trace(go.Scatter(x=dates, y=df_sampled['imbalance'], mode='lines', name='Imbalance', line=dict(color='purple')))
    imb_fig.update_layout(
        height=400,
        hovermode='x unified',
        xaxis_title="Date",
        yaxis_title="Imbalance Ratio"
    )
    # Add a horizontal line at imbalance = 1
    imb_fig.add_shape(type="line", x0=min(dates), x1=max(dates), y0=1, y1=1,
                    line=dict(color="gray", width=1, dash="dash"))
    
    st.plotly_chart(imb_fig, use_container_width=True)

elif viz_mode == "Time Slider":
    st.header("Orderbook Animation")
    
    # Sample the dataframe to avoid too many data points
    sample_rate = st.sidebar.slider("Sampling Rate", 1, 1000, 100)
    df_sampled = df_filtered.iloc[::sample_rate].reset_index(drop=True)
    
    # Slider to control the current frame
    frame_idx = st.slider("Timestamp", 0, len(df_sampled) - 1, 0)
    
    # Button to animate
    auto_play = st.sidebar.checkbox("Auto-play animation", value=False)
    play_speed = st.sidebar.slider("Play Speed (frames per second)", 1, 30, 5)
    
    # Get data for the current frame
    data = df_sampled.iloc[frame_idx]
    timestamp = datetime.fromtimestamp(data['timestamp'])
    
    # Display timestamp and key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Timestamp", timestamp.strftime("%Y-%m-%d %H:%M:%S"))
    col2.metric("Mid Price", f"${data['mid_price']:.4f}")
    col3.metric("Spread", f"${data['spread']:.4f}")
    col4.metric("Imbalance", f"{data['imbalance']:.4f}")
    
    # Create a plotly figure for the orderbook visualization
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=(f"Bid (${data['best_bid']:.4f})", f"Ask (${data['best_ask']:.4f})"),
                        column_widths=[0.5, 0.5])
    
    # Add bid and ask bars
    fig.add_trace(go.Bar(x=[data['bid_depth']], y=[''], orientation='h', 
                        marker=dict(color='rgba(0, 128, 0, 0.7)')), row=1, col=1)
    fig.add_trace(go.Bar(x=[data['ask_depth']], y=[''], orientation='h',
                        marker=dict(color='rgba(255, 0, 0, 0.7)')), row=1, col=2)
    
    # Update layout
    fig.update_layout(
        height=300,
        showlegend=False,
        xaxis_title="Depth",
        xaxis2_title="Depth",
    )
    
    # Invert x-axis for bid side
    fig.update_xaxes(autorange="reversed", row=1, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Price history chart
    history_window = min(100, frame_idx + 1)
    start_idx = max(0, frame_idx - history_window + 1)
    history = df_sampled.iloc[start_idx:frame_idx+1]
    
    history_dates = [datetime.fromtimestamp(ts) for ts in history['timestamp']]
    
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Scatter(x=history_dates, y=history['mid_price'], mode='lines', name='Mid', line=dict(color='blue')))
    hist_fig.add_trace(go.Scatter(x=history_dates, y=history['best_bid'], mode='lines', name='Bid', line=dict(color='green')))
    hist_fig.add_trace(go.Scatter(x=history_dates, y=history['best_ask'], mode='lines', name='Ask', line=dict(color='red')))
    
    hist_fig.update_layout(
        height=300,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Timestamp",
        yaxis_title="Price",
        yaxis=dict(range=[0, 1])
    )
    
    st.plotly_chart(hist_fig, use_container_width=True)
    
    # Auto-play animation
    if auto_play:
        if frame_idx < len(df_sampled) - 1:
            # Wait for a certain duration
            time.sleep(1.0 / play_speed)
            # Increment the slider value
            new_frame_idx = frame_idx + 1
            st.experimental_set_query_params(frame=new_frame_idx)
            st.experimental_rerun()

elif viz_mode == "Backtest Simulation":
    st.header("Simple Backtest Simulation")
    
    # Backtest parameters
    st.sidebar.subheader("Backtest Parameters")
    entry_threshold = st.sidebar.slider("Entry Imbalance Threshold", 0.01, 0.5, 0.15)
    exit_threshold = st.sidebar.slider("Exit Profit Target", 0.01, 0.2, 0.05)
    stop_loss = st.sidebar.slider("Stop Loss", 0.01, 0.2, 0.05)
    
    # Sample the dataframe to speed up the backtest
    sample_rate = st.sidebar.slider("Data Sampling Rate", 1, 1000, 100)
    df_sampled = df_filtered.iloc[::sample_rate].reset_index(drop=True)
    
    st.info(f"Running backtest on {len(df_sampled)} data points with Entry Threshold: {entry_threshold}, Profit Target: {exit_threshold}, Stop Loss: {stop_loss}")
    
    # Simple trading strategy
    balance = 100.0
    positions = []
    trades = []
    equity_curve = []
    
    # Backtest parameters
    ENTRY_THRESHOLD = entry_threshold
    EXIT_THRESHOLD = exit_threshold
    STOP_LOSS = stop_loss
    TRADE_SIZE = 1.0
    MAX_POSITIONS = 5
    SLIPPAGE = 0.005
    TRANSACTION_FEE = 0.01

    # Run the backtest
    equity_curve.append((df_sampled.iloc[0]['timestamp'], balance))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, row in enumerate(df_sampled.itertuples()):
        # Update progress
        progress = int((i + 1) / len(df_sampled) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing data point {i+1}/{len(df_sampled)} ({progress}%)")
        
        # Check exits
        for position in list(positions):
            exit_reason = None
            
            current_bid = row.best_bid
            current_ask = row.best_ask
            
            if position['type'] == 'long':
                if current_bid >= position['target_price']:
                    exit_reason = "Target Reached"
                elif current_bid <= position['stop_price']:
                    exit_reason = "Stop Loss"
            else:  # short position
                if current_ask <= position['target_price']:
                    exit_reason = "Target Reached"
                elif current_ask >= position['stop_price']:
                    exit_reason = "Stop Loss"
            
            if exit_reason:
                # Close position
                positions.remove(position)
                
                if position['type'] == 'long':
                    exit_price = current_bid - SLIPPAGE
                    exit_price = max(exit_price, 0.0)
                    pnl = (exit_price - position['entry_price']) * position['size']
                else:
                    exit_price = current_ask + SLIPPAGE
                    exit_price = min(exit_price, 1.0)
                    pnl = (position['entry_price'] - exit_price) * position['size']
                
                pnl -= TRANSACTION_FEE
                balance += pnl
                
                trades.append({
                    'type': position['type'],
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'exit_time': row.timestamp,
                    'exit_price': exit_price,
                    'size': position['size'],
                    'pnl': pnl,
                    'exit_reason': exit_reason
                })
        
        # Check entries
        if len(positions) < MAX_POSITIONS:
            if row.imbalance > 1 + ENTRY_THRESHOLD:
                # Long entry
                entry_price = row.best_ask + SLIPPAGE
                entry_price = min(entry_price, 1.0)
                
                positions.append({
                    'type': 'long',
                    'entry_price': entry_price,
                    'entry_time': row.timestamp,
                    'size': TRADE_SIZE,
                    'target_price': min(entry_price + EXIT_THRESHOLD, 1.0),
                    'stop_price': max(entry_price - STOP_LOSS, 0.0),
                })
                
                balance -= TRANSACTION_FEE
                
            elif row.imbalance < (1 - ENTRY_THRESHOLD):
                # Short entry
                entry_price = row.best_bid - SLIPPAGE
                entry_price = max(entry_price, 0.0)
                
                positions.append({
                    'type': 'short',
                    'entry_price': entry_price,
                    'entry_time': row.timestamp,
                    'size': TRADE_SIZE,
                    'target_price': max(entry_price - EXIT_THRESHOLD, 0.0),
                    'stop_price': min(entry_price + STOP_LOSS, 1.0),
                })
                
                balance -= TRANSACTION_FEE
        
        # Calculate unrealized PnL
        unrealized_pnl = 0
        for position in positions:
            if position['type'] == 'long':
                unrealized_pnl += (row.best_bid - position['entry_price']) * position['size']
            else:
                unrealized_pnl += (position['entry_price'] - row.best_ask) * position['size']
        
        # Record equity
        equity_curve.append((row.timestamp, balance + unrealized_pnl))
    
    # Close remaining positions at the last price
    last_row = df_sampled.iloc[-1]
    for position in list(positions):
        if position['type'] == 'long':
            exit_price = last_row['best_bid'] - SLIPPAGE
            exit_price = max(exit_price, 0.0)
            pnl = (exit_price - position['entry_price']) * position['size']
        else:
            exit_price = last_row['best_ask'] + SLIPPAGE
            exit_price = min(exit_price, 1.0)
            pnl = (position['entry_price'] - exit_price) * position['size']
        
        pnl -= TRANSACTION_FEE
        balance += pnl
        
        trades.append({
            'type': position['type'],
            'entry_time': position['entry_time'],
            'entry_price': position['entry_price'],
            'exit_time': last_row['timestamp'],
            'exit_price': exit_price,
            'size': position['size'],
            'pnl': pnl,
            'exit_reason': "Forced Exit"
        })
    
    # Convert results to DataFrames
    equity_df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
    equity_df['datetime'] = equity_df['timestamp'].apply(datetime.fromtimestamp)
    
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        trades_df['entry_datetime'] = trades_df['entry_time'].apply(datetime.fromtimestamp)
        trades_df['exit_datetime'] = trades_df['exit_time'].apply(datetime.fromtimestamp)
    
    # Display results
    status_text.text("Backtest complete!")
    
    if len(trades) == 0:
        st.warning("No trades were executed during the backtest period.")
    else:
        # Calculate statistics
        total_trades = len(trades_df)
        profitable_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        final_balance = balance
        roi = (final_balance - 100.0) / 100.0 * 100
        
        avg_profit = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if profitable_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Trades", f"{total_trades}")
        col2.metric("Win Rate", f"{win_rate:.2%}")
        col3.metric("Total P&L", f"${total_pnl:.2f}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Final Balance", f"${final_balance:.2f}")
        col2.metric("ROI", f"{roi:.2f}%")
        col3.metric("Profit Factor", f"{profit_factor:.2f}")
        
        col1, col2 = st.columns(2)
        col1.metric("Average Profit", f"${avg_profit:.2f}")
        col2.metric("Average Loss", f"${avg_loss:.2f}")
        
        # Plot equity curve
        st.subheader("Equity Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity_df['datetime'], y=equity_df['equity'], mode='lines', name='Equity'))
        fig.update_layout(
            height=400,
            hovermode='x unified',
            xaxis_title="Date",
            yaxis_title="Account Balance ($)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot price with trade markers
        st.subheader("Price Chart with Trades")
        price_fig = go.Figure()
        
        # Price line
        dates = [datetime.fromtimestamp(ts) for ts in df_sampled['timestamp']]
        price_fig.add_trace(go.Scatter(x=dates, y=df_sampled['mid_price'], mode='lines', name='Price', line=dict(color='black', width=1)))
        
        # Add trade markers
        if len(trades_df) > 0:
            # Long entries
            long_entries = trades_df[trades_df['type'] == 'long']
            if len(long_entries) > 0:
                price_fig.add_trace(go.Scatter(
                    x=long_entries['entry_datetime'],
                    y=long_entries['entry_price'],
                    mode='markers',
                    name='Long Entry',
                    marker=dict(color='blue', size=10, symbol='triangle-up')
                ))
            
            # Short entries
            short_entries = trades_df[trades_df['type'] == 'short']
            if len(short_entries) > 0:
                price_fig.add_trace(go.Scatter(
                    x=short_entries['entry_datetime'],
                    y=short_entries['entry_price'],
                    mode='markers',
                    name='Short Entry',
                    marker=dict(color='blue', size=10, symbol='triangle-down')
                ))
            
            # Profitable exits
            profit_exits = trades_df[trades_df['pnl'] > 0]
            if len(profit_exits) > 0:
                price_fig.add_trace(go.Scatter(
                    x=profit_exits['exit_datetime'],
                    y=profit_exits['exit_price'],
                    mode='markers',
                    name='Profitable Exit',
                    marker=dict(color='green', size=8, symbol='square')
                ))
            
            # Loss exits
            loss_exits = trades_df[trades_df['pnl'] <= 0]
            if len(loss_exits) > 0:
                price_fig.add_trace(go.Scatter(
                    x=loss_exits['exit_datetime'],
                    y=loss_exits['exit_price'],
                    mode='markers',
                    name='Loss Exit',
                    marker=dict(color='red', size=8, symbol='square')
                ))
        
        price_fig.update_layout(
            height=500,
            hovermode='x unified',
            xaxis_title="Date",
            yaxis_title="Price",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(price_fig, use_container_width=True)
        
        # Show trade details
        st.subheader("Trade Details")
        st.dataframe(trades_df[['type', 'entry_datetime', 'entry_price', 'exit_datetime', 'exit_price', 'pnl', 'exit_reason']]) 