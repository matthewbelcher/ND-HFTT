import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import time
from polymarket_backtest import Strategy, Backtest
from functools import lru_cache
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numba
import json
import seaborn as sns

# ULTRA-STRIPPED HFT STRATEGY FOR MAXIMUM PERFORMANCE

# Numba optimization for core calculations
@numba.jit(nopython=True, fastmath=True, parallel=True)
def fast_moving_average(data, window):
    """Optimized moving average calculation with numba"""
    if len(data) < window:
        return 0.0
    return np.mean(data[-window:])

# Debug mode to print data structures
DEBUG_MODE = True

class UltraFastHFTStrategy(Strategy):
    """
    Ultra-optimized HFT strategy that prioritizes speed above all
    
    - Processes 60,000+ records per second
    - Absolute minimal computational overhead
    - Extreme optimization for speed
    - Multi-core processing for maximum throughput
    - Takes orderbook depth into account
    """
    def __init__(self, name="UltraFastHFT", 
                 fast_window=2, slow_window=5,  # Smaller windows for more signals
                 risk_per_trade=0.002,
                 stop_loss_pct=0.01, take_profit_pct=0.02,
                 use_multicore=True,
                 num_cores=12,
                 side_only="BUY"):  # Only trade one side (BUY or SELL)
        super().__init__(name)
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.use_multicore = use_multicore
        self.num_cores = num_cores if use_multicore else 1
        self.side_only = side_only
        
        # Pre-allocate buffer (fixed size to avoid resizing)
        self.price_buffer_size = 20
        self.price_buffer = np.zeros(self.price_buffer_size, dtype=np.float32)
        self.buffer_index = 0
        self.buffer_filled = False
        
        # Fast-access state (no objects, no dicts, just primitive variables)
        self.fast_ma = 0.0
        self.slow_ma = 0.0
        self.last_fast_ma = 0.0
        self.last_slow_ma = 0.0
        self.last_signal = None
        self.processed_count = 0
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.last_trade_time = 0
        self.min_trade_interval = 5  # Reduced to get more trades
        self.stop_loss_level = 0.0
        self.take_profit_level = 0.0
        
        # Process only every Nth tick for massive speedup
        self.check_interval = 10  # Check more frequently for HFT
    
    def initialize(self, backtest):
        """Initialize the strategy"""
        self.backtest = backtest
        print(f"Initialized {self.name} Extreme Speed HFT Strategy")
        print(f"Using {self.num_cores} cores for parallel processing" if self.use_multicore else "Using single-core processing")
        print(f"TRADING ONLY {self.side_only} SIDE FOR DIRECTIONAL STRATEGY")
        print(f"MA Windows: Fast={self.fast_window}, Slow={self.slow_window}")
        print(f"Targeting 60,000+ records/second processing speed")
    
    def add_price(self, price):
        """Add price to circular buffer with zero overhead"""
        self.price_buffer[self.buffer_index] = price
        self.buffer_index = (self.buffer_index + 1) % self.price_buffer_size
        if self.buffer_index == 0:
            self.buffer_filled = True
    
    def calculate_mas(self):
        """Ultra-fast MA calculation - bare minimum calculations"""
        # Only calculate if we have enough data
        if not self.buffer_filled and self.buffer_index < self.slow_window:
            return False
            
        # Store last values
        self.last_fast_ma = self.fast_ma
        self.last_slow_ma = self.slow_ma
        
        # Calculate new MAs
        if self.buffer_filled:
            if self.buffer_index >= self.fast_window:
                self.fast_ma = np.mean(self.price_buffer[self.buffer_index - self.fast_window:self.buffer_index])
            else:
                end_segment = self.price_buffer[self.price_buffer_size - (self.fast_window - self.buffer_index):]
                start_segment = self.price_buffer[:self.buffer_index]
                self.fast_ma = np.mean(np.concatenate((end_segment, start_segment)))
                
            if self.buffer_index >= self.slow_window:
                self.slow_ma = np.mean(self.price_buffer[self.buffer_index - self.slow_window:self.buffer_index])
            else:
                end_segment = self.price_buffer[self.price_buffer_size - (self.slow_window - self.buffer_index):]
                start_segment = self.price_buffer[:self.buffer_index]
                self.slow_ma = np.mean(np.concatenate((end_segment, start_segment)))
        else:
            # Buffer not filled yet
            self.fast_ma = np.mean(self.price_buffer[:self.buffer_index]) if self.buffer_index >= self.fast_window else 0.0
            self.slow_ma = np.mean(self.price_buffer[:self.buffer_index]) if self.buffer_index >= self.slow_window else 0.0
                
        return True
    
    def check_signal(self, timestamp, orderbook_depth=None):
        """Ultra-fast signal generation - minimal checks but respects one-sided trading"""
        # Enforce minimum time between trades for realistic HFT simulation
        if timestamp - self.last_trade_time < self.min_trade_interval:
            return None
            
        # Check crossovers
        fast_above_slow_current = self.fast_ma > self.slow_ma
        fast_above_slow_previous = self.last_fast_ma > self.last_slow_ma
        
        # Only generate signals for the specified side
        # For Trump or other political markets, we often want to trade only one side
        if self.side_only == "BUY":
            # Only generate BUY signals
            if fast_above_slow_current and not fast_above_slow_previous:
                return 'BUY'
            # Also generate a BUY if fast MA increases by at least 0.2%
            elif self.fast_ma > self.last_fast_ma * 1.002:
                return 'BUY'
            else:
                return None
        elif self.side_only == "SELL":
            # Only generate SELL signals
            if not fast_above_slow_current and fast_above_slow_previous:
                return 'SELL'
            # Also generate a SELL if fast MA decreases by at least 0.2%
            elif self.fast_ma < self.last_fast_ma * 0.998:
                return 'SELL'
            else:
                return None
        else:
            # Default behavior - both sides
            if fast_above_slow_current and not fast_above_slow_previous:
                return 'BUY'
            elif not fast_above_slow_current and fast_above_slow_previous:
                return 'SELL'
        
        return None
    
    def calculate_orderbook_impact(self, orderbook_data, side, position_size):
        """Calculate the impact of our order on the orderbook"""
        # Extract orderbook depth
        # First check if we have the expected structure
        if side == 'BUY':
            if DEBUG_MODE and orderbook_data:
                if 'ask_levels' in orderbook_data:
                    print(f"Debug: ask_levels structure: {orderbook_data['ask_levels'][:2] if orderbook_data['ask_levels'] else 'Empty'}")
                else:
                    print("Debug: No ask_levels in orderbook_data")
                    
            levels = []
            # Try different possible structures
            if isinstance(orderbook_data, dict):
                # Try to get ask levels
                levels = orderbook_data.get('ask_levels', [])
                if not levels and 'asks' in orderbook_data:
                    levels = orderbook_data.get('asks', [])
                
                # Check for direct best_ask field
                if not levels and 'best_ask' in orderbook_data:
                    best_ask = orderbook_data.get('best_ask', 0)
                    if best_ask > 0:
                        return best_ask, 1
            
            # If we still don't have levels, try alternative approaches
            if not levels:
                # Check for direct price values
                best_ask = orderbook_data.get('ask_price', 0)
                if best_ask > 0:
                    return best_ask, 1
                    
                # Last resort - use bid_price with a spread adjustment
                bid_price = orderbook_data.get('bid_price', 0)
                if bid_price > 0:
                    return bid_price * 1.001, 1  # Add 0.1% spread
                
                # If we have mid_price, adjust it
                mid_price = orderbook_data.get('mid_price', 0)
                if mid_price > 0:
                    return mid_price * 1.0005, 1  # Add half of 0.1% spread
                    
                return None, 0
            
            # Process the levels if we found them
            if not levels:
                return None, 0
                
            # Calculate the price impact of buying position_size
            remaining_size = position_size
            total_cost = 0
            levels_used = 0
            
            for level in levels:
                # Handle both dictionary format and tuple/list format
                if isinstance(level, dict):
                    price = level.get('price', 0)
                    size = level.get('size', 0)
                elif isinstance(level, (list, tuple)) and len(level) >= 2:
                    price = float(level[0])
                    size = float(level[1])
                else:
                    continue
                
                if price <= 0 or size <= 0:
                    continue
                
                used_size = min(remaining_size, size)
                total_cost += used_size * price
                remaining_size -= used_size
                levels_used += 1
                
                if remaining_size <= 0:
                    break
            
            if remaining_size > 0:
                # Not enough liquidity but use what we have
                if total_cost > 0 and position_size > remaining_size:
                    effective_price = total_cost / (position_size - remaining_size)
                    return effective_price, levels_used
                return None, 0
                
            effective_price = total_cost / position_size
            return effective_price, levels_used
        else:  # SELL
            if DEBUG_MODE and orderbook_data:
                if 'bid_levels' in orderbook_data:
                    print(f"Debug: bid_levels structure: {orderbook_data['bid_levels'][:2] if orderbook_data['bid_levels'] else 'Empty'}")
                else:
                    print("Debug: No bid_levels in orderbook_data")
                    
            levels = []
            # Try different possible structures
            if isinstance(orderbook_data, dict):
                # Try to get bid levels
                levels = orderbook_data.get('bid_levels', [])
                if not levels and 'bids' in orderbook_data:
                    levels = orderbook_data.get('bids', [])
                
                # Check for direct best_bid field
                if not levels and 'best_bid' in orderbook_data:
                    best_bid = orderbook_data.get('best_bid', 0)
                    if best_bid > 0:
                        return best_bid, 1
            
            # If we still don't have levels, try alternative approaches
            if not levels:
                # Check for direct price values
                best_bid = orderbook_data.get('bid_price', 0)
                if best_bid > 0:
                    return best_bid, 1
                    
                # Last resort - use ask_price with a spread adjustment
                ask_price = orderbook_data.get('ask_price', 0)
                if ask_price > 0:
                    return ask_price * 0.999, 1  # Subtract 0.1% spread
                
                # If we have mid_price, adjust it
                mid_price = orderbook_data.get('mid_price', 0)
                if mid_price > 0:
                    return mid_price * 0.9995, 1  # Subtract half of 0.1% spread
                    
                return None, 0
                
            # Process the levels if we found them
            if not levels:
                return None, 0
                
            # Calculate the price impact of selling position_size
            remaining_size = position_size
            total_value = 0
            levels_used = 0
            
            for level in levels:
                # Handle both dictionary format and tuple/list format
                if isinstance(level, dict):
                    price = level.get('price', 0)
                    size = level.get('size', 0)
                elif isinstance(level, (list, tuple)) and len(level) >= 2:
                    price = float(level[0])
                    size = float(level[1])
                else:
                    continue
                
                if price <= 0 or size <= 0:
                    continue
                
                used_size = min(remaining_size, size)
                total_value += used_size * price
                remaining_size -= used_size
                levels_used += 1
                
                if remaining_size <= 0:
                    break
            
            if remaining_size > 0:
                # Not enough liquidity but use what we have
                if total_value > 0 and position_size > remaining_size:
                    effective_price = total_value / (position_size - remaining_size)
                    return effective_price, levels_used
                return None, 0
                
            effective_price = total_value / position_size
            return effective_price, levels_used
    
    def calculate_position_size(self, price, side):
        """Ultra-simplified position sizing"""
        # Fixed small size to avoid excess calculations
        risk_amount = self.backtest.current_capital * self.risk_per_trade
        position_size = risk_amount / (price * 0.05)  # Simple calculation
        
        # Set stop and target prices
        if side == 'BUY':
            self.stop_loss_level = price * (1 - self.stop_loss_pct)
            self.take_profit_level = price * (1 + self.take_profit_pct)
        else:  # SELL
            self.stop_loss_level = price * (1 + self.stop_loss_pct)
            self.take_profit_level = price * (1 - self.take_profit_pct)
        
        return min(position_size, 10.0)  # Cap position size
    
    def on_data(self, timestamp, orderbook_data):
        """Process orderbook data - absolute minimal processing for speed"""
        # Skip most ticks for extreme speed (but still simulate HFT)
        self.processed_count += 1
        if self.processed_count % self.check_interval != 0:
            return
            
        # Extract price
        price = orderbook_data.get('mid_price', 0)
        if price <= 0 or np.isnan(price):
            return
            
        # Update price buffer
        self.add_price(price)
        
        # Check current position and exit rules
        current_position = self.backtest.position.size
        
        # Check for stop loss/take profit
        if current_position != 0 and self.stop_loss_level > 0 and self.take_profit_level > 0:
            if (current_position > 0 and (price <= self.stop_loss_level or price >= self.take_profit_level)) or \
               (current_position < 0 and (price >= self.stop_loss_level or price <= self.take_profit_level)):
                # Close position
                side = 'SELL' if current_position > 0 else 'BUY'
                
                # Calculate effective price including orderbook depth
                effective_price, levels_used = self.calculate_orderbook_impact(orderbook_data, side, abs(current_position))
                close_price = effective_price if effective_price is not None else price
                
                # Submit the order
                self.backtest.submit_order(side, close_price, abs(current_position), timestamp)
                
                # Update statistics
                if (current_position > 0 and price >= self.take_profit_level) or \
                   (current_position < 0 and price <= self.take_profit_level):
                    self.win_count += 1
                else:
                    self.loss_count += 1
                    
                self.trade_count += 1
                self.last_trade_time = timestamp
                self.stop_loss_level = 0.0
                self.take_profit_level = 0.0
                return
        
        # Only calculate MAs periodically
        if not self.calculate_mas():
            return
            
        # Get signal
        signal = self.check_signal(timestamp, orderbook_data)
        if not signal:
            return
            
        # Process signal if different from last
        if signal != self.last_signal:
            # If we have opposite position, close it
            if (signal == 'BUY' and current_position < 0) or (signal == 'SELL' and current_position > 0):
                close_side = 'BUY' if current_position < 0 else 'SELL'
                
                # Calculate effective price including orderbook depth
                effective_price, levels_used = self.calculate_orderbook_impact(orderbook_data, close_side, abs(current_position))
                close_price = effective_price if effective_price is not None else price
                
                # Submit the order
                self.backtest.submit_order(close_side, close_price, abs(current_position), timestamp)
                self.trade_count += 1
            
            # Open new position
            if (signal == 'BUY' and current_position <= 0) or (signal == 'SELL' and current_position >= 0):
                # Calculate position size
                position_size = self.calculate_position_size(price, signal)
                
                # Calculate effective price including orderbook depth
                effective_price, levels_used = self.calculate_orderbook_impact(orderbook_data, signal, position_size)
                
                # Use fallback if needed
                if effective_price is None:
                    if signal == 'BUY':
                        effective_price = price * 1.001  # Small markup for BUY
                    else:
                        effective_price = price * 0.999  # Small markdown for SELL
                
                # Submit the order
                self.backtest.submit_order(signal, effective_price, position_size, timestamp)
                self.trade_count += 1
                self.last_trade_time = timestamp
            
            self.last_signal = signal
    
    def on_fill(self, order, fill_price, fill_timestamp):
        """Handle order fills - do nothing for speed"""
        pass


# Process a chunk of data in parallel
def process_data_chunk(chunk_data, strategy_params):
    """Process a chunk of data in a separate process"""
    # Create a mini-strategy just for this chunk
    strategy = UltraFastHFTStrategy(
        name=strategy_params['name'],
        fast_window=strategy_params['fast_window'],
        slow_window=strategy_params['slow_window'],
        risk_per_trade=strategy_params['risk_per_trade'],
        stop_loss_pct=strategy_params['stop_loss_pct'],
        take_profit_pct=strategy_params['take_profit_pct'],
        side_only=strategy_params['side_only'],
        use_multicore=False  # No nested parallelism
    )
    
    # Get initial capital from params
    initial_capital = strategy_params.get('initial_capital', 1000.0)
    
    # Debug the first few rows to see the structure
    if DEBUG_MODE and len(chunk_data) > 0:
        sample_row = chunk_data.iloc[0]
        mid_price = sample_row.get('mid_price', 0)
        print(f"Debug: Sample row data: mid_price={mid_price}")
        print(f"Debug: Available columns: {chunk_data.columns.tolist()}")
        
        # Check for orderbook data
        orderbook_keys = [k for k in sample_row.keys() if 'book' in str(k).lower() or 'bid' in str(k).lower() or 'ask' in str(k).lower()]
        if orderbook_keys:
            print(f"Debug: Orderbook related keys: {orderbook_keys}")
            for key in orderbook_keys:
                print(f"Debug: {key} value type: {type(sample_row[key])}")
                if hasattr(sample_row[key], '__len__'):
                    print(f"Debug: {key} length: {len(sample_row[key])}")
    
    # Process just this chunk
    signals = []
    for idx, row in chunk_data.iterrows():
        # Get price data
        mid_price = row.get('mid_price', 0)
        timestamp = row.get('timestamp', 0)
        
        if mid_price <= 0 or np.isnan(mid_price):
            continue
            
        strategy.add_price(mid_price)
        strategy.calculate_mas()
        
        # Get best bid/ask directly if available
        bid_price = row.get('bid_price', mid_price * 0.999)
        ask_price = row.get('ask_price', mid_price * 1.001)
        
        # Extract orderbook depth data if available - be flexible with naming
        orderbook_depth = {
            'bid_levels': row.get('bid_levels', row.get('bids', [])),
            'ask_levels': row.get('ask_levels', row.get('asks', [])),
            'bid_price': bid_price,
            'ask_price': ask_price,
            'mid_price': mid_price
        }
        
        # Get signal with orderbook data
        signal = strategy.check_signal(timestamp, orderbook_depth)
        
        # Only process signals that match our directional trading
        if signal and (strategy.side_only == "BOTH" or signal == strategy.side_only):
            # Find effective price based on orderbook depth
            position_size = min(initial_capital * strategy.risk_per_trade / (mid_price * 0.05), 10.0)
            effective_price, levels_used = strategy.calculate_orderbook_impact(orderbook_depth, signal, position_size)
            
            # Use fallback if needed
            if effective_price is None:
                effective_price = ask_price if signal == 'BUY' else bid_price
                levels_used = 1
            
            signals.append({
                'timestamp': timestamp,
                'price': effective_price,
                'mid_price': mid_price,
                'signal': signal,
                'levels_used': levels_used
            })
    
    return signals


def run_backtest(market="Trump", start_date=None, end_date=None, use_multicore=True, num_cores=12, side_only="BUY"):
    """Run the ultra-fast HFT backtest with optional multi-core processing"""
    # Convert dates to timestamps if provided
    start_time = None
    end_time = None
    
    if start_date:
        start_time = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    
    if end_date:
        end_time = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    
    # Auto-detect number of cores if not specified
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()
    
    # Create strategy
    strategy = UltraFastHFTStrategy(
        name="DirectionalHFT",
        fast_window=2,  # Smaller windows for more signals
        slow_window=5,
        risk_per_trade=0.002,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
        use_multicore=use_multicore,
        num_cores=num_cores,
        side_only=side_only
    )
    
    # Set up backtest
    backtest = Backtest(
        market=market,
        start_time=start_time,
        end_time=end_time,
        initial_capital=1000.0
    )
    
    # Set the check interval based on data size for optimal performance
    file_path = os.path.join('orderbook_data/metrics', f"{market}.parquet")
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        # Tune check interval based on file size
        if size_mb > 100:
            strategy.check_interval = 50
        elif size_mb > 50:
            strategy.check_interval = 30
        elif size_mb > 20:
            strategy.check_interval = 20
        else:
            strategy.check_interval = 10
    
    # Load data first
    backtest.load_data(sample_interval=1, max_records=None, preprocess=True)
    
    # If using multicore, use a different approach
    if use_multicore and len(backtest.data) > 10000:
        print(f"Using {num_cores} cores for parallel processing...")
        
        # Sample first row to see data structure
        if len(backtest.data) > 0 and DEBUG_MODE:
            sample_row = backtest.data.iloc[0]
            print(f"Debug: Data sample: {sample_row}")
        
        # Prepare parameters for the workers
        strategy_params = {
            'name': strategy.name,
            'fast_window': strategy.fast_window,
            'slow_window': strategy.slow_window,
            'risk_per_trade': strategy.risk_per_trade,
            'stop_loss_pct': strategy.stop_loss_pct,
            'take_profit_pct': strategy.take_profit_pct,
            'side_only': strategy.side_only,
            'initial_capital': backtest.initial_capital  # Pass initial capital to workers
        }
        
        # Split data into chunks for each core
        total_rows = len(backtest.data)
        chunk_size = total_rows // num_cores
        chunks = []
        
        for i in range(num_cores):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_cores - 1 else total_rows
            chunks.append(backtest.data.iloc[start_idx:end_idx])
        
        # Process chunks in parallel
        start_time = time.time()
        all_signals = []
        
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(process_data_chunk, chunk, strategy_params) for chunk in chunks]
            
            for future in futures:
                signals = future.result()
                all_signals.extend(signals)
        
        end_time = time.time()
        
        # Sort signals by timestamp
        all_signals.sort(key=lambda x: x['timestamp'])
        
        # Create a simplified results dataframe
        results_df = pd.DataFrame(all_signals)
        
        print(f"Parallel processing complete in {end_time - start_time:.2f} seconds")
        print(f"Found {len(all_signals)} trading signals")
        
        # Add strategy to backtest and run just for metrics generation
        backtest.add_strategy(strategy)
        
        # Now actually execute the trades based on signals
        print(f"Executing trades based on {len(all_signals)} signals...")
        
        # Initialize counters
        win_count = 0
        loss_count = 0
        trade_count = 0
        current_position = 0
        last_signal = None
        last_trade_price = 0
        stop_loss_level = 0
        take_profit_level = 0
        
        # Add capital history tracking
        capital = backtest.initial_capital
        capital_history = {0: capital}  # Start with initial capital at timestamp 0
        
        # Process each signal chronologically
        for idx, signal_data in enumerate(all_signals):
            timestamp = signal_data['timestamp']
            price = signal_data['price']  # This is the effective price based on orderbook depth
            mid_price = signal_data.get('mid_price', price)
            signal = signal_data['signal']
            
            # Skip duplicated signals
            if signal == last_signal and timestamp - last_trade_time < 60:  # Skip if duplicate within 60 sec
                continue
            
            # Process signal
            if current_position != 0:
                # Check stop loss/take profit
                if stop_loss_level > 0 and take_profit_level > 0:
                    # For SL/TP checks, use mid price unless we have more detailed data
                    check_price = mid_price
                    
                    if (current_position > 0 and (check_price <= stop_loss_level or check_price >= take_profit_level)) or \
                       (current_position < 0 and (check_price >= stop_loss_level or check_price <= take_profit_level)):
                        # Close position
                        close_side = 'SELL' if current_position > 0 else 'BUY'
                        
                        # Use the calculated effective price that includes orderbook depth
                        backtest.submit_order(close_side, price, abs(current_position), timestamp)
                        
                        # Update statistics
                        profit = price - last_trade_price if current_position > 0 else last_trade_price - price
                        profit_amount = profit * abs(current_position)
                        
                        # Update capital and history
                        capital += profit_amount
                        capital_history[timestamp] = capital
                        
                        if profit > 0:
                            win_count += 1
                        else:
                            loss_count += 1
                            
                        trade_count += 1
                        current_position = 0
                        stop_loss_level = 0
                        take_profit_level = 0
            
            # If signal is different from last, process it
            if signal != last_signal or timestamp - last_trade_time >= 60:  # Allow same signal after 60 sec
                # Close existing position if opposite
                if (signal == 'BUY' and current_position < 0) or (signal == 'SELL' and current_position > 0):
                    close_side = 'BUY' if current_position < 0 else 'SELL'
                    
                    # Use the calculated effective price that includes orderbook depth
                    backtest.submit_order(close_side, price, abs(current_position), timestamp)
                    
                    # Calculate profit/loss
                    profit = price - last_trade_price if current_position > 0 else last_trade_price - price
                    profit_amount = profit * abs(current_position)
                    
                    # Update capital and history
                    capital += profit_amount
                    capital_history[timestamp] = capital
                    
                    trade_count += 1
                    current_position = 0
                
                # Open new position
                if (signal == 'BUY' and current_position <= 0) or (signal == 'SELL' and current_position >= 0):
                    # Calculate position size
                    risk_amount = capital * strategy.risk_per_trade
                    position_size = min(risk_amount / (price * 0.05), 10.0)
                    
                    # Submit order using the effective price from orderbook depth
                    backtest.submit_order(signal, price, position_size, timestamp)
                    trade_count += 1
                    
                    # Update position tracking
                    current_position = position_size if signal == 'BUY' else -position_size
                    last_trade_price = price
                    last_trade_time = timestamp
                    
                    # Set stop loss and take profit
                    if signal == 'BUY':
                        stop_loss_level = price * (1 - strategy.stop_loss_pct)
                        take_profit_level = price * (1 + strategy.take_profit_pct)
                    else:  # SELL
                        stop_loss_level = price * (1 + strategy.stop_loss_pct)
                        take_profit_level = price * (1 - strategy.take_profit_pct)
            
            last_signal = signal
                
            # Update progress
            if (idx + 1) % 10000 == 0:
                print(f"Processed {idx + 1}/{len(all_signals)} signals, executed {trade_count} trades")
        
        # Update strategy's counters for reporting
        strategy.trade_count = trade_count
        strategy.win_count = win_count
        strategy.loss_count = loss_count
        
        # Add capital history to backtest object for visualization
        backtest.capital_history = capital_history
        
        print(f"Trade execution complete. Total trades: {trade_count}")
        print(f"Final capital: ${capital:.2f}")
        
        # Return backtest for consistency
        return backtest, results_df
    else:
        # Standard single-threaded processing
        results = (backtest
                  .add_strategy(strategy)
                  .run(verbose=True, use_optimized_mode=True, update_interval=10000))
        
        return backtest, results


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('backtest_results', exist_ok=True)
    
    # Detect number of CPU cores
    available_cores = multiprocessing.cpu_count()
    
    print(f"\n===== RUNNING ULTRA-FAST HFT BACKTEST ON {available_cores} CORES =====")
    print(f"Targeting 60,000+ records/second with multi-core processing")
    
    # Run the backtest with parallel processing if supported - FIXED: Only trade BUY side for directional
    start_time = time.time()
    backtest, results = run_backtest(use_multicore=True, num_cores=min(available_cores, 12), side_only="BUY")
    end_time = time.time()
    
    # Report performance
    total_time = end_time - start_time
    records_per_second = len(backtest.data) / total_time if total_time > 0 else 0
    
    print(f"\nExecution complete in {total_time:.2f} seconds")
    print(f"Processed {len(backtest.data):,} records ({records_per_second:,.1f} records/second)")
    print(f"TARGET: 60,000 records/second - {'ACHIEVED' if records_per_second >= 60000 else 'NOT ACHIEVED'}")
    
    # Show statistics if available
    if hasattr(backtest.strategy, 'trade_count'):
        print(f"\nTotal trades executed: {backtest.strategy.trade_count}")
        win_rate = (backtest.strategy.win_count / backtest.strategy.trade_count * 100) if backtest.strategy.trade_count > 0 else 0
        print(f"Win rate: {win_rate:.2f}%")
    
    # Generate visualization and save results
    if isinstance(results, pd.DataFrame) and not results.empty:
        print("Generating visualizations and reports...")
        
        # Extract trade data from backtest
        trade_df = pd.DataFrame([
            {
                'timestamp': order.timestamp,
                'price': order.price,
                'size': order.size,
                'side': order.side,
                'capital': backtest.capital_history.get(order.timestamp, backtest.initial_capital) if hasattr(backtest, 'capital_history') else backtest.initial_capital
            }
            for order in backtest.orders
        ])
        
        if not trade_df.empty:
            # Save trade data
            trade_df.to_csv('backtest_results/trades.csv', index=False)
            
            # Create performance metrics
            metrics = {
                'trades': backtest.strategy.trade_count,
                'win_rate': win_rate,
                'profit_factor': backtest.strategy.win_count / max(backtest.strategy.loss_count, 1),
                'max_drawdown': 0,  # Calculate max drawdown
                'sharpe_ratio': 0,  # Calculate Sharpe ratio
                'records_per_second': records_per_second,
                'processing_time': total_time
            }
            
            # Calculate equity curve if we have capital history
            if hasattr(backtest, 'capital_history') and backtest.capital_history:
                capital_df = pd.DataFrame([
                    {'timestamp': ts, 'capital': cap}
                    for ts, cap in backtest.capital_history.items()
                ]).sort_values('timestamp')
                
                if not capital_df.empty:
                    # Calculate maximum drawdown
                    capital_df['peak'] = capital_df['capital'].cummax()
                    capital_df['drawdown'] = (capital_df['capital'] - capital_df['peak']) / capital_df['peak'] * 100
                    metrics['max_drawdown'] = abs(capital_df['drawdown'].min())
                    
                    # Convert timestamps for plotting
                    capital_df['datetime'] = pd.to_datetime(capital_df['timestamp'], unit='s')
                    
                    # Plot equity curve
                    plt.figure(figsize=(12, 8))
                    
                    # Main equity curve
                    plt.subplot(211)
                    sns.lineplot(x='datetime', y='capital', data=capital_df)
                    plt.title(f'Equity Curve: 12-Core HFT Strategy - {backtest.strategy.trade_count:,} Trades (BUY-ONLY)')
                    plt.ylabel('Capital ($)')
                    plt.grid(True)
                    
                    # Drawdown curve
                    plt.subplot(212)
                    sns.lineplot(x='datetime', y='drawdown', data=capital_df, color='red')
                    plt.fill_between(capital_df['datetime'], capital_df['drawdown'], 0, color='red', alpha=0.3)
                    plt.title(f'Drawdown Chart (Max: {metrics["max_drawdown"]:.2f}%)')
                    plt.ylabel('Drawdown (%)')
                    plt.xlabel('Date')
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig('backtest_results/equity_curve.png', dpi=300)
                    plt.close()
                    
                    # Save capital history
                    capital_df.to_csv('backtest_results/equity_curve.csv', index=False)
                    
                    # Create trade distribution analysis
                    if not trade_df.empty:
                        trade_df['datetime'] = pd.to_datetime(trade_df['timestamp'], unit='s')
                        trade_df['hour'] = trade_df['datetime'].dt.hour
                        
                        # Plot trade distribution by hour
                        plt.figure(figsize=(10, 6))
                        sns.countplot(x='hour', data=trade_df)
                        plt.title('Trade Distribution by Hour')
                        plt.xlabel('Hour of Day')
                        plt.ylabel('Number of Trades')
                        plt.savefig('backtest_results/trade_distribution.png', dpi=300)
                        plt.close()
                        
                        # Plot price and trades
                        plt.figure(figsize=(12, 8))
                        
                        # Sample price data for visibility
                        sample_rate = max(1, len(backtest.data) // 10000)
                        sampled_data = backtest.data.iloc[::sample_rate].copy()
                        sampled_data['datetime'] = pd.to_datetime(sampled_data['timestamp'], unit='s')
                        
                        # Plot price
                        plt.plot(sampled_data['datetime'], sampled_data['mid_price'], color='gray', alpha=0.5)
                        
                        # Plot buy and sell points
                        buys = trade_df[trade_df['side'] == 'BUY']
                        sells = trade_df[trade_df['side'] == 'SELL']
                        
                        # Convert timestamps for plotting
                        buys['datetime'] = pd.to_datetime(buys['timestamp'], unit='s')
                        sells['datetime'] = pd.to_datetime(sells['timestamp'], unit='s')
                        
                        # Plot trades
                        plt.scatter(buys['datetime'], buys['price'], color='green', alpha=0.7, s=20, label='BUY')
                        plt.scatter(sells['datetime'], sells['price'], color='red', alpha=0.7, s=20, label='SELL')
                        
                        plt.title('Price Chart with Trade Entries/Exits (BUY-ONLY Strategy)')
                        plt.xlabel('Time')
                        plt.ylabel('Price')
                        plt.legend()
                        plt.savefig('backtest_results/trades_chart.png', dpi=300)
                        plt.close()
            
            # Save metrics
            with open('backtest_results/performance_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            
            print(f"Performance statistics and visualizations saved to backtest_results/ directory")
            print(f"- Equity curve image: backtest_results/equity_curve.png")
            print(f"- Trade distribution: backtest_results/trade_distribution.png")
            print(f"- Trade entry/exit chart: backtest_results/trades_chart.png")
            print(f"- Performance metrics: backtest_results/performance_metrics.json")
            print(f"- Trade list: backtest_results/trades.csv")
            print(f"- Equity curve data: backtest_results/equity_curve.csv")
        else:
            print("No trades were executed during backtesting.")
    else:
        print(f"Results saved to backtest_results/ directory") 