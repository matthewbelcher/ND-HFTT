import os
import pandas as pd
from datetime import datetime
import csv
import glob

def load_coinbase_trades(file_path='/Users/juanporras/Desktop/Cash&Carry/coinbase_trades.csv'):
    """Load the Coinbase trades data."""
    try:
        trades_df = pd.read_csv(file_path)
        # Convert time to datetime
        trades_df['time'] = pd.to_datetime(trades_df['time'])
        # Extract second-level timestamp for matching with orderbook data
        trades_df['timestamp_second'] = trades_df['time'].dt.floor('s')
        # Add a flag to track if the trade has been used for arbitrage
        trades_df['used_for_arbitrage'] = False
        # Filter to only include SELL orders
        trades_df = trades_df[trades_df['side'] == 'SELL']
        return trades_df
    except Exception as e:
        print(f"Error loading Coinbase trades: {e}")
        return None

def load_binance_orderbooks(folder_path='/Users/juanporras/Desktop/Cash&Carry/binance_orderbook_data'):
    """Load all Binance orderbook CSV files."""
    try:
        # Get all CSV files in the directory
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        
        if not csv_files:
            print(f"No CSV files found in {folder_path}")
            return None
        
        print(f"Found {len(csv_files)} CSV files in {folder_path}")
        
        orderbooks = []
        for file_path in csv_files:
            try:
                print(f"Processing {os.path.basename(file_path)}...")
                df = pd.read_csv(file_path)
                # Extract filename without extension as a new column
                file_name = os.path.basename(file_path)
                df['source_file'] = file_name
                
                # Print the first few column names for debugging
                print(f"Columns in file: {', '.join(df.columns[:5])}...")
                
                # Handle the timestamp format in the provided data
                if 'timestamp_utc' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp_utc'])
                    df['timestamp_second'] = df['timestamp'].dt.floor('s')
                elif 'timestamp' in df.columns:
                    # Convert milliseconds to seconds if needed
                    try:
                        if df['timestamp'].iloc[0] > 1e12:  # Likely milliseconds
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        else:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df['timestamp_second'] = df['timestamp'].dt.floor('s')
                    except:
                        print(f"Could not convert timestamp in {file_name}, trying different format...")
                        try:
                            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
                            df['timestamp_second'] = df['timestamp'].dt.floor('s')
                        except:
                            print(f"Failed to convert timestamp in {file_name}")
                            continue
                else:
                    # Try to find any column that might contain timestamp data
                    time_cols = [col for col in df.columns if any(term in col.lower() for term in ['time', 'date'])]
                    if time_cols:
                        print(f"Trying alternative time column: {time_cols[0]}")
                        try:
                            df['timestamp'] = pd.to_datetime(df[time_cols[0]])
                            df['timestamp_second'] = df['timestamp'].dt.floor('s')
                        except:
                            print(f"Could not convert {time_cols[0]} to timestamp in {file_name}")
                            continue
                    else:
                        print(f"No timestamp column found in {file_name}")
                        continue
                
                # Add fields for best bid/ask if they don't exist
                if 'best_bid' not in df.columns:
                    if 'bidPrice' in df.columns:
                        df['best_bid'] = df['bidPrice'].astype(float)
                    elif 'bid' in df.columns:
                        df['best_bid'] = df['bid'].astype(float)
                    else:
                        print(f"No bid price column found in {file_name}")
                        continue
                
                if 'best_ask' not in df.columns:
                    if 'askPrice' in df.columns:
                        df['best_ask'] = df['askPrice'].astype(float)
                    elif 'ask' in df.columns:
                        df['best_ask'] = df['ask'].astype(float)
                    else:
                        print(f"No ask price column found in {file_name}")
                        continue
                
                # Add fields to track available liquidity
                if 'best_bid_size' not in df.columns:
                    if 'bidQty' in df.columns:
                        df['best_bid_size'] = df['bidQty'].astype(float)
                    elif 'bidSize' in df.columns:
                        df['best_bid_size'] = df['bidSize'].astype(float)
                    else:
                        df['best_bid_size'] = 1.0  # Default value if not available
                
                if 'best_ask_size' not in df.columns:
                    if 'askQty' in df.columns:
                        df['best_ask_size'] = df['askQty'].astype(float)
                    elif 'askSize' in df.columns:
                        df['best_ask_size'] = df['askSize'].astype(float)
                    else:
                        df['best_ask_size'] = 1.0  # Default value if not available
                
                print(f"Successfully processed {file_name} with {len(df)} rows")
                orderbooks.append(df)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Combine all dataframes
        if orderbooks:
            combined_df = pd.concat(orderbooks, ignore_index=True)
            print(f"Combined {len(orderbooks)} files into DataFrame with {len(combined_df)} rows")
            return combined_df
        else:
            print("No valid orderbook data found")
            return None
    except Exception as e:
        print(f"Error loading Binance orderbooks: {e}")
        return None

def print_cash_carry_opportunity(second, coinbase_trade, binance_data):
    """Print details of a matching cash-and-carry arbitrage opportunity."""
    # Trading fees
    COINBASE_FEE = 0.0005  # 0.05%
    BINANCE_FEE = 0.000225  # 0.0225%
    
    trade_price = float(coinbase_trade['price'])
    trade_size = float(coinbase_trade['size'])
    trade_side = coinbase_trade['side']
    
    # Ensure we're only processing SELL orders
    if trade_side != 'SELL':
        return False, 0
    
    # Extract contract expiry date if available in the product_id (format: BIT-25APR25-CDE)
    product_id = coinbase_trade['product_id']
    contract_date = "Unknown"
    if "-" in product_id:
        parts = product_id.split("-")
        if len(parts) > 1:
            contract_date = parts[1]  # Extract the date part (e.g., 25APR25)
    
    # Calculate prices including fees
    coinbase_price_with_fee = trade_price * (1 - COINBASE_FEE)  # SELL on Coinbase
    
    # Ensure binance_ask exists and is a number
    if pd.isna(binance_data['best_ask']) or binance_data['best_ask'] == 0:
        return False, 0
        
    binance_ask = float(binance_data['best_ask'])
    binance_ask_with_fee = binance_ask * (1 + BINANCE_FEE)
    
    is_cash_carry = False
    spread = None
    profit = 0
    
    # For cash-and-carry: We need spot price (Binance) < futures price (Coinbase)
    # We buy spot on Binance and sell futures on Coinbase
    # For true cash-and-carry, ensure spot price < futures price
    if binance_ask_with_fee < coinbase_price_with_fee:
        spread = coinbase_price_with_fee - binance_ask_with_fee
        is_cash_carry = True
        profit = spread * (trade_size/100)  # Adjust for BTC size
    
    # Print output only if it's a cash-and-carry opportunity
    if is_cash_carry:
        print("\n" + "-"*50)
        print(f"CASH-AND-CARRY OPPORTUNITY AT {second}")
        print("-"*50)
        print(f"Coinbase Futures: {product_id} ({contract_date}) | Trade #{coinbase_trade['trade_id']} | {trade_side}")
        print(f"Prices: Coinbase Future ${trade_price:.2f} | Binance Spot Ask ${binance_ask:.2f}")
        print(f"With Fees: Coinbase ${coinbase_price_with_fee:.2f} | Binance ${binance_ask_with_fee:.2f}")
        print(f"Size: {trade_size/100:.4f} BTC")
        print(f"ARBITRAGE: Spread ${spread:.2f} | Potential Profit ${profit:.2f}")
        print(f"Strategy: Buy on Binance spot, sell on Coinbase futures")
        print(f"Source: {binance_data['source_file']}")
        print("-"*50)
        
        return is_cash_carry, profit
    
    return False, 0

def find_cash_carry_opportunities(coinbase_trades, binance_orderbooks):
    """Find cash-and-carry arbitrage opportunities."""
    opportunities = []
    profits_by_date = {}  # To track profits by date
    profits_by_contract = {}  # To track profits by Coinbase contract
    
    if coinbase_trades is None or binance_orderbooks is None:
        print("Missing data, cannot find arbitrage opportunities")
        return opportunities, profits_by_date, profits_by_contract
    
    print(f"Initial Coinbase trades count: {len(coinbase_trades)}")
    print(f"SELL trades count: {len(coinbase_trades[coinbase_trades['side'] == 'SELL'])}")
    
    # Convert timestamp_second columns to string format for comparison
    coinbase_trades['timestamp_str'] = coinbase_trades['timestamp_second'].dt.strftime('%Y-%m-%d %H:%M:%S')
    binance_orderbooks['timestamp_str'] = binance_orderbooks['timestamp_second'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Create a dictionary of Binance data for efficient lookup
    binance_dict = {}
    for timestamp_str, group in binance_orderbooks.groupby('timestamp_str'):
        binance_dict[timestamp_str] = group.to_dict('records')
    
    # Counter for tracked opportunities
    opportunity_count = 0
    trade_count = 0
    
    # Process each Coinbase trade
    for i, trade in coinbase_trades.iterrows():
        trade_count += 1
        if trade_count % 10000 == 0:
            print(f"Processed {trade_count} trades, found {opportunity_count} opportunities")
        
        # Skip trades already used for arbitrage or not SELL
        if trade['used_for_arbitrage'] or trade['side'] != 'SELL':
            continue
        
        timestamp_str = trade['timestamp_str']
        # Extract date from timestamp for daily profit tracking
        date = timestamp_str.split()[0]  # Just the date part (YYYY-MM-DD)
        
        # Check if we have matching Binance data for this timestamp
        if timestamp_str in binance_dict:
            binance_entries = binance_dict[timestamp_str]
            
            # Track if we found an opportunity for this trade
            found_opportunity = False
            
            for orderbook in binance_entries:
                # Skip if this particular Binance entry has been processed for a specific opportunity
                if orderbook.get('used_for_this_opportunity', False):
                    continue
                
                # Check for cash-and-carry opportunity
                is_opportunity, profit = print_cash_carry_opportunity(timestamp_str, trade, orderbook)
                
                if is_opportunity:
                    # Extract info for recording
                    trade_price = float(trade['price'])
                    trade_size = float(trade['size'])
                    product_id = trade['product_id']
                    
                    # Extract contract date if available
                    contract_date = "Unknown"
                    if "-" in product_id:
                        parts = product_id.split("-")
                        if len(parts) > 1:
                            contract_date = parts[1]  # e.g., 25APR25
                    
                    # Update profits by date
                    if date not in profits_by_date:
                        profits_by_date[date] = 0
                    profits_by_date[date] += profit
                    
                    # Update profits by contract
                    if contract_date not in profits_by_contract:
                        profits_by_contract[contract_date] = 0
                    profits_by_contract[contract_date] += profit
                    
                    opportunities.append({
                        'timestamp': timestamp_str,
                        'coinbase_product': product_id,
                        'contract_date': contract_date,
                        'coinbase_trade_id': trade['trade_id'],
                        'coinbase_price': trade_price,
                        'binance_ask': float(orderbook['best_ask']),
                        'spread': profit * (100/trade_size),  # Per BTC spread
                        'trade_size': trade_size/100,  # In BTC
                        'profit_potential': profit,
                        'binance_file': orderbook['source_file']
                    })
                    
                    opportunity_count += 1
                    
                    # Mark this trade as used
                    coinbase_trades.at[i, 'used_for_arbitrage'] = True
                    # Mark this Binance entry as used for this specific opportunity
                    orderbook['used_for_this_opportunity'] = True
                    
                    found_opportunity = True
                    break  # Move to the next trade once we find an opportunity
            
            if found_opportunity:
                continue  # Move to the next trade
    
    print(f"Finished processing {trade_count} trades, found {opportunity_count} total opportunities")
    return opportunities, profits_by_date, profits_by_contract

def save_opportunities(opportunities, output_file='cash_carry_opportunities.csv'):
    """Save arbitrage opportunities to a CSV file."""
    if not opportunities:
        print("No arbitrage opportunities found to save")
        return
    
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=opportunities[0].keys())
            writer.writeheader()
            writer.writerows(opportunities)
        print(f"Successfully saved {len(opportunities)} arbitrage opportunities to {output_file}")
    except Exception as e:
        print(f"Error saving opportunities: {e}")

def save_profit_summaries(profits_by_date, profits_by_contract, date_file='profits_by_date.csv', contract_file='profits_by_contract.csv'):
    """Save profit summaries to CSV files."""
    try:
        # Save profits by date
        with open(date_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Profit'])
            for date, profit in sorted(profits_by_date.items()):
                writer.writerow([date, f"${profit:.2f}"])
        print(f"Successfully saved profits by date to {date_file}")
        
        # Save profits by contract
        with open(contract_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Contract', 'Profit'])
            for contract, profit in sorted(profits_by_contract.items()):
                writer.writerow([contract, f"${profit:.2f}"])
        print(f"Successfully saved profits by contract to {contract_file}")
    except Exception as e:
        print(f"Error saving profit summaries: {e}")

def main():
    print("Loading Coinbase trades data...")
    coinbase_trades = load_coinbase_trades()
    
    print("Loading Binance orderbook data...")
    binance_orderbooks = load_binance_orderbooks()
    
    if coinbase_trades is not None:
        print(f"Loaded {len(coinbase_trades)} Coinbase trades (SELL orders only)")
    
    if binance_orderbooks is not None:
        print(f"Loaded {len(binance_orderbooks)} Binance orderbook entries")
    
    print("Finding cash-and-carry arbitrage opportunities...")
    print("Strategy: Buy on Binance spot, sell on Coinbase futures")
    print("Including trading fees: Coinbase 0.05%, Binance 0.0225%")
    opportunities, profits_by_date, profits_by_contract = find_cash_carry_opportunities(coinbase_trades, binance_orderbooks)
    
    if opportunities:
        print("\n" + "="*50)
        print("CASH-AND-CARRY ARBITRAGE SUMMARY")
        print("="*50)
        print(f"Total opportunities found: {len(opportunities)}")
        total_profit = sum(opp['profit_potential'] for opp in opportunities)
        print(f"Total potential profit: ${total_profit:.2f}")
        
        # Calculate average spread
        avg_spread = sum(opp['spread'] for opp in opportunities) / len(opportunities)
        print(f"Average spread per BTC: ${avg_spread:.2f}")
        
        # Calculate total BTC traded
        total_btc = sum(opp['trade_size'] for opp in opportunities)
        print(f"Total BTC traded: {total_btc:.4f}")
        
        # Display profits by date
        print("\nProfits by Date:")
        for date, profit in sorted(profits_by_date.items()):
            print(f"{date}: ${profit:.2f}")
        
        # Display profits by contract
        print("\nProfits by Contract Expiry:")
        for contract, profit in sorted(profits_by_contract.items()):
            print(f"{contract}: ${profit:.2f}")
        
        # Top opportunities
        if len(opportunities) > 0:
            sorted_opps = sorted(opportunities, key=lambda x: x['profit_potential'], reverse=True)
            print("\nTop 5 Opportunities:")
            for i, opp in enumerate(sorted_opps[:5], 1):
                print(f"{i}. ${opp['profit_potential']:.2f} at {opp['timestamp']} | {opp['coinbase_product']} | Spread: ${opp['spread']:.2f}/BTC | Size: {opp['trade_size']:.4f} BTC")
        
        print("\nNote: This data represents a short sampling period. Actual opportunities")
        print("over longer periods would likely yield different results.")
        print("="*50)
    else:
        print("\nNo cash-and-carry arbitrage opportunities found")
    
    # Save opportunities to CSV
    save_opportunities(opportunities)
    
    # Also save profit summaries to CSV
    save_profit_summaries(profits_by_date, profits_by_contract)

if __name__ == "__main__":
    main()