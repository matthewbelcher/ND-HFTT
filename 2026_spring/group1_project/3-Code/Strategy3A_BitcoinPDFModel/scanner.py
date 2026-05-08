import requests
from datetime import date, datetime
from decimal import Decimal
import numpy as np

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
    current_hour = date.today().hour
    current_minute = date.today().minute
    next_15min_mark = ((current_minute // 15) + 1) * 15
    if next_15min_mark == 60:
        next_15min_mark = 0
        current_hour += 1
    return f"KXBTC15M-{today.strftime('%y%b%d').upper()}{current_hour:02d}{next_15min_mark:02d}"

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

def print_market_data(event_ticker):
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

    market_data = []
    index = 0
    first_price = None
    
    total_implied_probability = Decimal("0.00")
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
        if Decimal(str(yes_bid)) == Decimal("0.0000") or Decimal(str(no_ask)) == Decimal("1.0000") or yes_bid is None or no_ask is None:
            continue  # Skip zero bid or ask of 1.00 which indicates no liquidity
        
        
        # Fall back to orderbook if the market-level prices are missing
        if yes_bid is None and no_bid is None:
            try:
                ob = get_orderbook(ticker)
                yb, ya, nb, na = derive_best_prices(ob)
                yes_bid, yes_ask, no_bid, no_ask = yb, ya, nb, na
            except Exception:
                pass

        yes_bid_s = fmt(Decimal(str(yes_bid))) if yes_bid is not None else "  N/A "
        yes_ask_s = fmt(Decimal(str(yes_ask))) if yes_ask is not None else "  N/A "
        no_bid_s  = fmt(Decimal(str(no_bid)))  if no_bid  is not None else "  N/A "
        no_ask_s  = fmt(Decimal(str(no_ask)))  if no_ask  is not None else "  N/A "
        last_s    = fmt(Decimal(str(last)))     if last    is not None else "  N/A "
        midpoint_s = fmt((Decimal(str(yes_bid)) + Decimal(str(yes_ask))) / 2) if yes_bid is not None and yes_ask is not None else "  N/A "
        implied_probability = (Decimal(str(yes_bid)) + Decimal(str(yes_ask))) / 2 if yes_bid is not None and yes_ask is not None else None
        if 'BTCD' in event_ticker.upper():
            implied_probability = 1 - implied_probability if implied_probability is not None else None
        label_floor = label.split()[0]  # Get the strike price part for sorting
        print(f"  {label:<{col}} {label_floor:<10} {midpoint_s:>8} {implied_probability:.2%} {total_implied_probability:.2%}")
        if first_price is None:
            first_price = label_floor
        market_data.append({
            "label": label,
            "label_floor": label_floor,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
            "last": last,
            "midpoint": (Decimal(str(yes_bid)) + Decimal(str(yes_ask))) / 2 if yes_bid is not None and yes_ask is not None else None,
            "implied_probability": implied_probability,
            "total_implied_probability": total_implied_probability,
            "index": index
        })
        total_implied_probability += implied_probability if implied_probability is not None else Decimal("0.00")
        #print(f"  {label:<{col}} {yes_bid_s:>8} {yes_ask_s:>8}  |  {no_bid_s:>8} {no_ask_s:>8}  {last_s:>8}")
        index += 1
    return market_data, first_price

###
### Probability
### P(S_T > K) = N( -d₂ )
### where d₂ = [ ln(K/S₀) + σ²T/2 ] / (σ√T)  
### K is threshold
### T is time to expiration
### S₀ is current price
### σ is volatility calculated by taking the standrad deviation of the returns of the past
### 200 candles and multiplying by the square root of time T in minutes
### r_i = ln(P_i / P_{i-1})
### σ_15min = σ_1min × √15
###  EWMA (Exponentially Weighted Moving Average) — this is what JPMorgan's RiskMetrics model uses, and it's simple:
### σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t
### Or could use derebit DVOL (REST endpoint)
### N() is the cumulative distribution function of the standard normal distribution  
    
    
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

    
def main():
    #print(f"\nFetching markets for event: {event_ticker}")
    print(f"(Bitcoin price range — today at 5pm EST)\n")
    
    event_ticker_upper = get_todays_above_below_5pm_event_ticker()
    #event_ticker_upper = get_todays_BTC_hourly_above_below_event_ticker()
    hourly_ab_data, some_price = print_market_data(event_ticker_upper)
    event_ticker = get_todays_5pm_event_ticker()
    #event_ticker = get_todays_BTC_hourly_event_ticker()
    hourly_data, first_price = print_market_data(event_ticker)
    
    most_negative_diff = Decimal("0.00")
    most_negative_diff_market = None
    most_positive_diff = Decimal("0.00")
    most_positive_diff_market = None
    most_negative_diff_index = None
    most_positive_diff_index = None
    most_negative_diff_cost = Decimal("0.00")
    most_positive_diff_cost = Decimal("0.00")
    
    total_estimated_cost = Decimal("0.00")
    base_cost = Decimal("0.00")
    
    for row in hourly_ab_data: # Row is data in an above/below market
        if row["label_floor"] == first_price:
            print(f"First price in AB market: {row['label_floor']} with implied probability {row['implied_probability']:.2%}")
            base_cost = Decimal(1-(row['midpoint'])) if row['midpoint'] is not None else Decimal("0.00")
            print(f"Base cost to buy AB NO at this strike: ${base_cost:.2f}")
            
    for row in hourly_data: # Row is data in a range
        print(row["label"])
        for col in hourly_ab_data: # Col is data in an above/below market
            if row["label_floor"] == col["label_floor"]:
                print("  Matched AB data:", col["label"])
                # difference is cumulative range probability minus the AB probability for that strike
                print(f" Difference in Probability: {(row['total_implied_probability']+base_cost) - col['implied_probability']}")
                if row['total_implied_probability'] is not None and col['implied_probability'] is not None:
                    diff = row['total_implied_probability'] - col['implied_probability']
                    if diff < most_negative_diff:
                        most_negative_diff = diff
                        most_negative_diff_market = row['label']
                        most_negative_diff_index = row['index']
                        most_negative_diff_cost = Decimal(1-(col['midpoint'])) if col['midpoint'] is not None else Decimal("0.00")
                    if diff > most_positive_diff:
                        most_positive_diff = diff
                        most_positive_diff_market = row['label']
                        most_positive_diff_index = row['index']
                        most_positive_diff_cost = Decimal((col['midpoint'])) if col['midpoint'] is not None else Decimal("0.00")
    # Negative Difference means it is better to buy the AB market
    # Positive Difference means it is better to buy the range market  
    
    total_estimated_cost = most_negative_diff_cost + most_positive_diff_cost
    # Buy AB NO at most negative
    print(f"\nMost Negative Difference: {most_negative_diff:.2%} in market {most_negative_diff_market}")
    # BUY AB YES at most positive
    print(f"Most Positive Difference: {most_positive_diff:.2%} in market {most_positive_diff_market}")
    # BUY Range YES in between the two
    
    # Determine which range markets are in between the two AB markets and print those as well
    print("\nRange markets in between:")
    for row in hourly_data:
        if most_negative_diff_index is not None and most_positive_diff_index is not None:
            if row['index'] >= most_negative_diff_index and row['index'] < most_positive_diff_index:
                total_estimated_cost += Decimal(str(row['midpoint'])) if row['midpoint'] is not None else Decimal("0.00")
                print(f"  {row['label']} with cumulative probability {row['total_implied_probability']:.2%}")
    
    print(f"\nEstimated Cost to Buy AB NO at {most_negative_diff_cost:.2f} and AB YES at {most_positive_diff_cost:.2f} and all range markets in between: ${total_estimated_cost:.2f}")
    print(f"\n  All prices in dollars (max payout = $1.00 per contract)")
    
    
    current_time = datetime.now()
    current_hour = current_time.hour
    print("Current Hour:", current_hour)
    
    
    sigma_1min = get_btc_vol_1min()
    sigma_10min = sigma_1min * np.sqrt(10)


if __name__ == "__main__":
    main()