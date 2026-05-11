import math
from urllib import response
import uuid

import time
import requests
from datetime import date, datetime
from decimal import Decimal
import numpy as np
from scipy.stats import norm
import test_order  # Import the authentication and API request functions from test_order.py

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


def get_todays_5pm_event_ticker():
    """Build today's 5pm EST BTC event ticker dynamically."""
    today = date.today()
    current_time = datetime.now()
    current_hour = current_time.hour
    if current_hour >= 17:  # If it's already past 5pm, show tomorrow's event
        today = today.replace(day=today.day + 1)
    # Format: KXBTC-YYMMDD17  (17 = 17:00 EST = 5pm)
    return f"KXBTC-{today.strftime('%y%b%d').upper()}17"

def get_todays_above_below_5pm_event_ticker():
    """Build today's 5pm EST BTC event ticker dynamically."""
    today = date.today()
    current_time = datetime.now()
    current_hour = current_time.hour
    if current_hour >= 17:  # If it's already past 5pm, show tomorrow's event
        today = today.replace(day=today.day + 1)
    # Format: KXBTCD-YYMMDD17  (17 = 17:00 EST = 5pm)
    return f"KXBTCD-{today.strftime('%y%b%d').upper()}17"

def get_todays_BTC_hourly_event_ticker():
    """Build today's 5pm EST BTC event ticker dynamically."""
    today = date.today()
    # Format: KXBTC-YYMMDD14  (14 = 14:00 EST = 2pm)
    current_time = datetime.now()
    current_hour = current_time.hour + 1  # Next hour's event, e.g. if it's 2:30pm, show 3pm event
    return f"KXBTC-{today.strftime('%y%b%d').upper()}{current_hour:02d}"

def get_todays_BTC_hourly_above_below_event_ticker():
    """Build today's 5pm EST BTC event ticker dynamically."""
    today = date.today()
    # Format: KXBTCD-YYMMDD14  (14 = 14:00 EST = 2pm)
    current_time = datetime.now()
    current_hour = current_time.hour + 1  # Next hour's event, e.g. if it's 2:30pm, show 3pm event
    return f"KXBTCD-{today.strftime('%y%b%d').upper()}{current_hour:02d}"

def get_todays_BTC_15min_event_ticker():
    """Build today's 5pm EST BTC event ticker dynamically."""
    today = date.today()
    # Format: KXBTC15M-YYMMDD1430  (1430 = 2:30pm)
    current_time = datetime.now()
    current_hour = current_time.hour
    current_minute = current_time.minute
    next_15min_mark = ((current_minute // 15) + 1) * 15
    if next_15min_mark == 60:
        next_15min_mark = 0
        current_hour += 1
    return f"KXBTC15M-{today.strftime('%y%b%d').upper()}{current_hour:02d}{next_15min_mark:02d}"

def get_todays_ETH_15min_event_ticker():
    """Build today's 5pm EST BTC event ticker dynamically."""
    today = date.today()
    # Format: KXBTC15M-YYMMDD1430  (1430 = 2:30pm)
    current_time = datetime.now()
    current_hour = current_time.hour
    current_minute = current_time.minute
    next_15min_mark = ((current_minute // 15) + 1) * 15
    if next_15min_mark == 60:
        next_15min_mark = 0
        current_hour += 1
    return f"KXETH15M-{today.strftime('%y%b%d').upper()}{current_hour:02d}{next_15min_mark:02d}"

def get_todays_XRP_15min_event_ticker():
    """Build today's 5pm EST BTC event ticker dynamically."""
    today = date.today()
    # Format: KXBTC15M-YYMMDD1430  (1430 = 2:30pm)
    current_time = datetime.now()
    current_hour = current_time.hour
    current_minute = current_time.minute
    next_15min_mark = ((current_minute // 15) + 1) * 15
    if next_15min_mark == 60:
        next_15min_mark = 0
        current_hour += 1
    return f"KXXRP15M-{today.strftime('%y%b%d').upper()}{current_hour:02d}{next_15min_mark:02d}"

def get_todays_SOL_15min_event_ticker():
    """Build today's 5pm EST BTC event ticker dynamically."""
    today = date.today()
    # Format: KXBTC15M-YYMMDD1430  (1430 = 2:30pm)
    current_time = datetime.now()
    current_hour = current_time.hour
    current_minute = current_time.minute
    next_15min_mark = ((current_minute // 15) + 1) * 15
    if next_15min_mark == 60:
        next_15min_mark = 0
        current_hour += 1
    return f"KXSOL15M-{today.strftime('%y%b%d').upper()}{current_hour:02d}{next_15min_mark:02d}"


def get_event_markets(event_ticker):
    """Fetch all markets (strike options) within a specific event."""
    url = f"{BASE_URL}/events/{event_ticker}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data.get("markets", [])


def get_orderbook(ticker, depth=10):
    """Fetch the full orderbook for a market."""
    url = f"{BASE_URL}/markets/{ticker}/orderbook"
    response = requests.get(url, params={"depth": depth})
    response.raise_for_status()
    return response.json().get("orderbook_fp", {})


def derive_best_prices(orderbook):
    """
    Derive best bid/ask from the orderbook.
    YES ask = $1.00 - best NO bid (and vice versa).
    Bids are sorted worst-to-best, so the last element is the best bid.
    """
    yes_levels = orderbook.get("yes_dollars", [])
    no_levels = orderbook.get("no_dollars", [])

    best_yes_bid = Decimal(str(yes_levels[-1][0])) if yes_levels else None
    best_no_bid = Decimal(str(no_levels[-1][0])) if no_levels else None

    best_yes_ask = (Decimal("1.00") - best_no_bid) if best_no_bid is not None else None
    best_no_ask = (Decimal("1.00") - best_yes_bid) if best_yes_bid is not None else None

    return best_yes_bid, best_yes_ask, best_no_bid, best_no_ask


def fmt(val):
    return f"${val:.2f}" if val is not None else "  N/A "

def get_best_bids_asks(event_ticker):
    try:
        markets = get_event_markets(event_ticker)
    except requests.HTTPError as e:
        print(f"Error fetching event: {e}")
        print("The market may not be open yet, or the ticker format may differ.")
        print(f"Try checking: https://kalshi.com/markets/{event_ticker.lower()}/bitcoin-range")
        return

    if not markets:
        print("No markets found for this event.")
        return

    # Sort markets by their subtitle/strike for a clean ordered table
    markets.sort(key=lambda m: m.get("yes_sub_title", ""))

    print(f"Found {len(markets)} strike options.\n")

    # Header
    col = 28
    print(f"  {'Strike / Range':<{col}} {'YES Bid':>8} {'YES Ask':>8}  |  {'NO Bid':>8} {'NO Ask':>8}  {'Last':>8}")
    print(f"  {'-' * (col + 52)}")

    yes_bids = []
    yes_asks = []
    no_bids = []
    no_asks = []

    for market in markets:
        ticker = market["ticker"]
        label = market.get("yes_sub_title", ticker)[:col]

        # Use the pre-computed best prices from the markets list first (fast path)
        yes_bid = market.get("yes_bid_dollars")
        yes_ask = market.get("yes_ask_dollars")
        no_bid = market.get("no_bid_dollars")
        no_ask = market.get("no_ask_dollars")
        last = market.get("last_price_dollars")

        #print(f"Debug: {ticker} - yes_bid: {yes_bid}, yes_ask: {yes_ask}, no_bid: {no_bid}, no_ask: {no_ask}, last: {last}")
        #if Decimal(str(yes_bid)) == Decimal("0.0000") or Decimal(str(no_ask)) == Decimal("1.0000") or yes_bid is None or no_ask is None:
        #    continue  # Skip zero bid or ask of 1.00 which indicates no liquidity
        
        # Fall back to orderbook if the market-level prices are missing
        if yes_bid is None and no_bid is None:
            try:
                ob = get_orderbook(ticker)
                yb, ya, nb, na = derive_best_prices(ob)
                yes_bid, yes_ask, no_bid, no_ask = yb, ya, nb, na
            except Exception:
                pass
        
        yes_bids.append(yes_bid)
        yes_asks.append(yes_ask)
        no_bids.append(no_bid)
        no_asks.append(no_ask)
        
        return yes_bids, yes_asks, no_bids, no_asks

###
### Probability
### P(S_T > K) = N( -dΓéé )
### where dΓéé = [ ln(K/SΓéÇ) + ╧â┬▓T/2 ] / (╧âΓêÜT)  
### K is threshold
### T is time to expiration
### SΓéÇ is current price
### ╧â is volatility calculated by taking the standrad deviation of the returns of the past
### 200 candles and multiplying by the square root of time T in minutes
### r_i = ln(P_i / P_{i-1})
### ╧â_15min = ╧â_1min ├ù ΓêÜ15
###  EWMA (Exponentially Weighted Moving Average) ΓÇö this is what JPMorgan's RiskMetrics model uses, and it's simple:
### ╧â┬▓_t = ╬╗┬╖╧â┬▓_{t-1} + (1-╬╗)┬╖r┬▓_t
### Or could use derebit DVOL (REST endpoint)
### N() is the cumulative distribution function of the standard normal distribution  

def calculate_probability(S0, K, T, sigma):
    print(f"Calculating probability with S0={S0}, K={K}, T={T}, sigma={sigma}")
    d2 = (np.log(K / S0) + (sigma**2 * T / 2)) / (sigma * np.sqrt(T))
    print(f"Calculated d2: {d2}")
    probability = 1 - norm.cdf(d2)  # P(S_T > K) = 1 - N(d2)
    return probability  

def new_calculate_probability(S0, K, T, sigma):
    stddev = sigma * S0
    probability = 1 - norm.cdf(K, loc=S0, scale=stddev)
    return probability

def get_btc_vol_1min(n_candles=200, lam=0.97):
    url = "https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD/candles"
    params = {"granularity": "ONE_MINUTE", "limit": n_candles}
    data = requests.get(url, params=params).json()
    
    closes = np.array([float(c["close"]) for c in data["candles"]])
    log_returns = np.diff(np.log(closes))
    
    var = np.var(log_returns[:20])
    for r in log_returns[20:]:
        var = lam * var + (1 - lam) * r**2
    
    return np.sqrt(var)

def get_eth_vol_1min(n_candles=200, lam=0.97):
    url = "https://api.coinbase.com/api/v3/brokerage/market/products/ETH-USD/candles"
    params = {"granularity": "ONE_MINUTE", "limit": n_candles}
    data = requests.get(url, params=params).json()
    
    closes = np.array([float(c["close"]) for c in data["candles"]])
    log_returns = np.diff(np.log(closes))
    
    var = np.var(log_returns[:20])
    for r in log_returns[20:]:
        var = lam * var + (1 - lam) * r**2
    
    return np.sqrt(var)

def get_sol_vol_1min(n_candles=200, lam=0.97):
    url = "https://api.coinbase.com/api/v3/brokerage/market/products/SOL-USD/candles"
    params = {"granularity": "ONE_MINUTE", "limit": n_candles}
    data = requests.get(url, params=params).json()
    
    closes = np.array([float(c["close"]) for c in data["candles"]])
    log_returns = np.diff(np.log(closes))
    
    var = np.var(log_returns[:20])
    for r in log_returns[20:]:
        var = lam * var + (1 - lam) * r**2
    
    return np.sqrt(var)

def get_xrp_vol_1min(n_candles=200, lam=0.97):
    url = "https://api.coinbase.com/api/v3/brokerage/market/products/XRP-USD/candles"
    params = {"granularity": "ONE_MINUTE", "limit": n_candles}
    data = requests.get(url, params=params).json()
    
    closes = np.array([float(c["close"]) for c in data["candles"]])
    log_returns = np.diff(np.log(closes))
    
    var = np.var(log_returns[:20])
    for r in log_returns[20:]:
        var = lam * var + (1 - lam) * r**2
    
    return np.sqrt(var)

def get_btc_price_simple():
    url = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
    data = requests.get(url).json()
    return float(data["data"]["amount"])

def get_eth_price_simple():
    url = "https://api.coinbase.com/v2/prices/ETH-USD/spot"
    data = requests.get(url).json()
    return float(data["data"]["amount"])

def get_sol_price_simple():
    url = "https://api.coinbase.com/v2/prices/SOL-USD/spot"
    data = requests.get(url).json()
    return float(data["data"]["amount"])

def get_xrp_price_simple():
    url = "https://api.coinbase.com/v2/prices/XRP-USD/spot"
    data = requests.get(url).json()
    return float(data["data"]["amount"])

def time_to_next_15min():
    now = datetime.now()
    next_15min_mark = ((now.minute // 15) + 1) * 15
    if next_15min_mark == 60:
        next_15min_mark = 0
        now = now.replace(hour=now.hour + 1)
    next_time = now.replace(minute=next_15min_mark, second=0, microsecond=0)
    return (next_time - datetime.now()).total_seconds() / 60


def query_15m_chart(size, difference_threshold, threshold_mins, execution):
    
    event_ticker = get_todays_BTC_15min_event_ticker()
    current_time = datetime.now()
    current_hour = current_time.hour
    current_minute = current_time.minute
    next_15min_mark = ((current_minute // 15) + 1) * 15
    if next_15min_mark == 60:
        next_15min_mark = 0
        current_hour += 1
    market_ticker = event_ticker + f"-{next_15min_mark:02d}"
    
    position = test_order.get_position(market_ticker)
    print(position)
    market_exposure = float(position.get('market_exposure_dollars')) if position is not None else 0
    position_fp = int(abs(float(position.get('position_fp')))) if position is not None else 0
    
    ## Add feature to check if we should close the position
    
    time_to_next = time_to_next_15min() - 0.4 # Subtract to account for 1 minute moving average
    print(f"\nTime to next 15-minute event: {time_to_next:.1f} minutes")
    if time_to_next < threshold_mins:
        print("Too little time to make buys. SKIPPING EVERYTHING ELSE")
        return
    
    
    try:
        markets = get_event_markets(event_ticker)
    except requests.HTTPError as e:
        print(f"Error fetching event: {e}")
        print("The market may not be open yet, or the ticker format may differ.")
        print(f"Try checking: https://kalshi.com/markets/{event_ticker.lower()}/bitcoin-range")
        return

    if not markets:
        print("No markets found for this event.")
        return

    current_btc_price = get_btc_price_simple()
    print(f"Current BTC price: ${current_btc_price:.2f}")
    
    # Sort markets by their subtitle/strike for a clean ordered table
    markets.sort(key=lambda m: m.get("yes_sub_title", ""))

    #print(f"Found {len(markets)} strike options.\n")
    target_price = None
    # Header
    col = 28
    #print(f"  {'Strike / Range':<{col}} {'YES Bid':>8} {'YES Ask':>8}  |  {'NO Bid':>8} {'NO Ask':>8}  {'Last':>8}")
    #print(f"  {'-' * (col + 52)}")
    for market in markets:
        #print(market)
        target_price = market.get("floor_strike")
        yes_bid = float(market.get("yes_bid_dollars"))
        #yes_bid_size = market.get("yes_bid_size_fp")
        yes_ask = float(market.get("yes_ask_dollars"))
        #yes_ask_size = market.get("yes_ask_size_fp")
        no_bid = float(market.get("no_bid_dollars"))
        no_ask = float(market.get("no_ask_dollars"))
        #no_bid_size = market.get("no_bid_size_fp")
        #no_ask_size = market.get("no_ask_size_fp")
        #print(f"Debug: {event_ticker} - target_price: {target_price}, yes_bid: {yes_bid} ({yes_bid_size}), yes_ask: {yes_ask} ({yes_ask_size}), no_bid: {no_bid} ({no_bid_size}), no_ask: {no_ask} ({no_ask_size})")

    ## DETERMINE USING LIMIT OR MARKET ORDER
    # LIMIT uses midpoint
    yes_midpoint = math.floor((yes_bid + yes_ask) * 50)/100 if yes_bid is not None and yes_ask is not None else None
    real_midpoint = (yes_bid + yes_ask) / 2 if yes_bid is not None and yes_ask is not None else None
    no_midpoint = math.floor((no_bid + no_ask) * 50)/100 if no_bid is not None and no_ask is not None else None
    if yes_midpoint is None or no_midpoint is None:
        print("Insufficient price data to calculate midpoints, skipping this cycle.")
        return
    print(f"Target price for next 15-minute event: ${target_price:.2f}" if target_price else "Target price not found")
    
    # Possibly get second BTC price and average it
    sigma_1min = get_btc_vol_1min()
    sigma = sigma_1min * np.sqrt(time_to_next)
    print(f"Estimated 1-minute volatility (╧â): {sigma_1min:.4f}")
    print(f"Scaled volatility for {time_to_next:.1f} minutes (╧âΓêÜT): {sigma:.4f}")
    
    #probability_up = calculate_probability(S0=current_btc_price, K=target_price, T=time_to_next, sigma=sigma)
    #print(f"Implied probability of BTC being above ${target_price:.2f} in {time_to_next:.1f} minutes: {probability_up:.2%}")
    print(f"Cost to buy YES at ${yes_ask:.2f} implies an implied probability of {yes_ask:.2%}")
    new_probability_up = new_calculate_probability(S0=current_btc_price, K=target_price, T=time_to_next, sigma=sigma)
    print(f"Implied probability of BTC being above ${target_price:.2f} in {time_to_next:.1f} minutes: {new_probability_up:.2%}")
    difference = (new_probability_up - real_midpoint) if real_midpoint is not None else 0
    print(f"Difference between new method and implied probability from YES ask: {difference:.2%}")
    #if position is not None:
    #    if market_exposure > 0:
    #        if position_fp > 0: # yes position
    #            print(f"Already have a position in {market_ticker}, skipping trading logic.")
    #            if difference < -difference_threshold/2: # Can maybe adjust these values
                    # Close yes position by selling
    #                print(f"Closing YES position in {market_ticker} by selling at ${yes_bid:.2f}")
                    #test_order.create_yes_order(market_ticker, "sell", position_fp, str(yes_bid))
    #            return
    #        elif position_fp < 0: # no position
    #            print(f"Already have a position in {market_ticker}, skipping trading logic.")
    #            if difference > difference_threshold/2:
                    # Close no position by selling
    #                print(f"Closing NO position in {market_ticker} by selling at ${no_bid:.2f}")
                    #test_order.create_no_order(market_ticker, "sell", position_fp, str(no_bid))
    #            return
    if difference > difference_threshold:
        #BUY yes
        if execution:
            order_size = size // yes_ask if yes_ask > 0 else size
            order_size = int(order_size)
            test_order.create_yes_order(market_ticker, "buy", order_size, str(yes_ask))
        else:
            print(f"Would place BUY YES order for {size} contracts at ${yes_ask:.2f} based on a difference of {difference:.2%}")
    elif difference < -difference_threshold:
        #BUY no
        if execution:
            order_size = size // no_ask if no_ask > 0 else size
            order_size = int(order_size)
            test_order.create_no_order(market_ticker, "buy", order_size, str(no_ask))
        else:
            print(f"Would place BUY NO order for {size} contracts at ${no_ask:.2f} based on a difference of {difference:.2%}")
    else:
        print("No clear edge based on the difference between model probability and market price.")

def main():
    # POSSIBLY COULD ADD CHECKING BITCOIN PRICE FOR LARGE MOVEMENTS AND BUYING THEN
    # Could look at adding take profit/stop loss points
    # Possibly scale things better relative to time and percent chance
    
    ### VARIABLES TO SET BEFORE RUNNING ###
    size = 1.5
    difference_threshold = 0.09  # 10% difference threshold for deciding to buy/sell
    threshold_mins = 1  # Minimum time to next event in minutes to consider trading
    execution = True
    while True:
        try:
            query_15m_chart(size, difference_threshold, threshold_mins, execution)
        except Exception as e:
            print(f"An error occurred: {e}")
        # Wait for a minute before checking again
        time.sleep(30)

if __name__ == "__main__":
    main()
