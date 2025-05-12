import sqlite3
import os
import heapq
import json
from abc import ABC, abstractmethod
import random
import datetime as dt
import pandas as pd

class Order:
    '''Meant for strategy to submit individual orders to backtest engine'''
    def __init__(self, symbol, quantity, price, order_type, side):
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.order_type = order_type
        self.side = side

class PriceLevel:
    '''Holds information about the price, quantity at a given price level'''
    def __init__(self, side, price, quantity):
        self.side = side
        self.price = price
        self.quantity = quantity

    def __lt__(self, other):
        if self.side == 'bid':
            return self.price > other.price
        else:
            return self.price < other.price

class OrderBook:
    '''Holds all pricelevel items for a given market'''
    def __init__(self):
        self.bids = []
        self.asks = []
        self.bid_map = dict()
        self.ask_map = dict()
    
    def update_order_book(self, side, price, d_quantity):
        if side == 'yes':
            if price in self.bid_map:
                self.bid_map[price].quantity += d_quantity

                # remove pricelevel if qty <= 0
                if self.bid_map[price].quantity <= 0:
                    idx = self.bids.index(self.bid_map[price])
                    self.bids[idx] = self.bids[-1]
                    self.bids.pop()
                    heapq.heapify(self.bids)
                    del self.bid_map[price]
            elif d_quantity > 0:
                pl = PriceLevel('bid', price, d_quantity)
                heapq.heappush(self.bids, pl)
                self.bid_map[price] = pl
        elif side == 'no':
            if 100-price in self.ask_map:
                self.ask_map[100-price].quantity += d_quantity

                # remove pricelevel if qty <= 0
                if self.ask_map[100-price].quantity <= 0:
                    idx = self.asks.index(self.ask_map[100-price])
                    self.asks[idx] = self.asks[-1]
                    self.asks.pop()
                    heapq.heapify(self.asks)
                    del self.ask_map[100-price]
            elif d_quantity > 0:
                pl = PriceLevel('ask', 100-price, d_quantity)
                heapq.heappush(self.asks, pl)
                self.ask_map[100-price] = pl

    def get_best_bid(self):
        if self.bids:
            return self.bids[0].price, self.bids[0].quantity
        return 0, 0
    
    def get_best_ask(self):
        if self.asks:
            return self.asks[0].price, self.asks[0].quantity
        return 100, 0
    
    def get_mid_price(self):
        best_a = self.get_best_ask()[0]
        best_b = self.get_best_bid()[0]
        return (best_a + best_b) / 2

class Strategy(ABC):
    '''Generates signals based on orderbook'''
    def __init__(self, cash):
        self.position = 0
        self.cost_basis = None
        self.cash = cash

    @abstractmethod
    def generate_signal(orderbook):
        pass

    def update_position(self, prc, qty):
        # initiating a position
        if self.cost_basis is None or self.position == 0:
            self.cost_basis = prc
            self.position += qty
            self.cash -= prc * qty

        # changing position
        else:
            self.position += qty
            self.cash -= prc * qty

            # switching direction
            if (self.position < 0 and self.position + qty > 0) or (self.position > 0 and self.position + qty < 0):
                self.cost_basis = prc

            # extending position
            elif self.position * qty > 0:
                self.cost_basis = (self.cost_basis * self.position + prc * qty) / (self.position + qty)

    def get_pnl(self, market_price):  
        if self.cost_basis is None:
            return self.cash  
        
        return self.cash + self.position * (market_price - self.cost_basis)

class Backtest:
    '''Main backtest engine'''
    def __init__(self, strategy, market_ticker, start_time, end_time, latency):
        self.strategy = strategy
        self.market_ticker = market_ticker
        self.start_time = start_time
        self.end_time = end_time
        self.latency = latency
        self.pnl = []
        self.yes_orders = []
        self.no_orders = []
        self.price_history = []
        self.db_path = os.path.abspath(os.path.join('historical_data', 'kalshi_orderbooks.db'))
        if not os.path.exists(self.db_path):
            print(f"Database file not found at: {self.db_path}")
            return None

    def run(self):
        '''Runs the backtest'''
        orderbook = self.initiate_orderbook()
        outstanding_orders = []

        if orderbook is None:
            print('Backtest failed to initiate orderbook')
            return
        
        # run backtest
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp, price, delta, side FROM orderbook_deltas \
                           WHERE market_ticker=? AND timestamp BETWEEN ? AND ?", 
                           (self.market_ticker, self.start_time, self.end_time))
            rows = cursor.fetchall()

            # backtest loop
            for timestamp, price, quantity, side in rows:
                # submit orders after latency period
                idx = 0
                while outstanding_orders and outstanding_orders[idx][0] <= (dt.datetime.fromisoformat(timestamp) + dt.timedelta(seconds=self.latency)).isoformat():
                    
                    # handle market orders
                    if outstanding_orders[idx][1].order_type == 'market':
                        if outstanding_orders[idx][1].side == 'yes':
                            purchase_prc, purchase_qty = orderbook.get_best_ask()

                            if purchase_qty == 0:
                                outstanding_orders.pop(idx)
                                print('Unable to fill yes market order: no liquidity')
                                continue

                            # fully filled
                            if purchase_qty >= outstanding_orders[idx][1].quantity:
                                orderbook.update_order_book('no', 100-purchase_prc, -outstanding_orders[idx][1].quantity) # update order book
                                self.strategy.update_position(purchase_prc, outstanding_orders[idx][1].quantity) # update position
                                outstanding_orders.pop(idx) # remove filled order
                            # partially filled
                            else:
                                orderbook.update_order_book('no', 100-purchase_prc, -purchase_qty)
                                self.strategy.update_position(purchase_prc, purchase_qty)
                                outstanding_orders[idx][1].quantity -= purchase_qty # update remaining quantity
                                continue

                            self.yes_orders.append((timestamp, purchase_prc))

                        elif outstanding_orders[idx][1].side == 'no':
                            sale_prc, sale_qty = orderbook.get_best_bid()

                            if sale_qty == 0:
                                outstanding_orders.pop(idx)
                                print('Unable to fill no market order: no liquidity')
                                continue

                            if sale_qty >= outstanding_orders[idx][1].quantity:
                                orderbook.update_order_book('yes', sale_prc, -outstanding_orders[idx][1].quantity)
                                self.strategy.update_position(sale_prc, -outstanding_orders[idx][1].quantity)
                                outstanding_orders.pop(idx)
                            else:
                                orderbook.update_order_book('yes', sale_prc, -sale_qty)
                                self.strategy.update_position(sale_prc, -sale_qty)
                                outstanding_orders[idx][1].quantity -= sale_qty
                                continue

                            self.no_orders.append((timestamp, sale_prc))
                    
                    # handle ioc orders
                    elif outstanding_orders[idx][1].order_type == 'ioc':
                        if outstanding_orders[idx][1].side == 'yes':
                            purchase_prc, purchase_qty = orderbook.get_best_ask()

                            if purchase_qty == 0:
                                outstanding_orders.pop(idx)
                                continue
                            
                            # filled
                            if purchase_prc <= outstanding_orders[idx][1].price:
                                # fully
                                if purchase_qty >= outstanding_orders[idx][1].quantity:
                                    orderbook.update_order_book('no', 100-purchase_prc, -outstanding_orders[idx][1].quantity)
                                    self.strategy.update_position(purchase_prc, outstanding_orders[idx][1].quantity)
                                    outstanding_orders.pop(idx)
                                # partially
                                else:
                                    orderbook.update_order_book('no', 100-purchase_prc, -purchase_qty)
                                    self.strategy.update_position(purchase_prc, purchase_qty)
                            else:
                                outstanding_orders.pop(idx)
                                continue

                            self.yes_orders.append((timestamp, purchase_prc))

                        if outstanding_orders[idx][1].side == 'no':
                            sale_prc, sale_qty = orderbook.get_best_bid()

                            if sale_qty == 0:
                                outstanding_orders.pop(idx)
                                continue
                            
                            # filled
                            if sale_prc >= outstanding_orders[idx][1].price:
                                # fully
                                if sale_qty >= outstanding_orders[idx][1].quantity:
                                    orderbook.update_order_book('yes', sale_prc, -outstanding_orders[idx][1].quantity)
                                    self.strategy.update_position(sale_prc, -outstanding_orders[idx][1].quantity)
                                    outstanding_orders.pop(idx)
                                # partially
                                else:
                                    orderbook.update_order_book('yes', sale_prc, -sale_qty)
                                    self.strategy.update_position(sale_prc, -sale_qty)
                            else:
                                outstanding_orders.pop(idx)
                                continue

                            self.no_orders.append((timestamp, sale_prc))

                # Update the order book with the delta
                orderbook.update_order_book(side, price, quantity)

                # calculate pnl
                self.pnl.append((timestamp, self.strategy.get_pnl(orderbook.get_mid_price())))

                # record price
                self.price_history.append((timestamp, orderbook.get_best_ask()[0], orderbook.get_best_bid()[0]))

                # Generate signals using the strategy
                orders = self.strategy.generate_signal(orderbook)
                if hasattr(orders, '__iter__'):
                    for order in orders:
                        if order is not None:
                            outstanding_orders.append((timestamp, order))
                elif orders is not None:
                        outstanding_orders.append((timestamp, orders))

    def report(self):
        '''Reports the results of the backtest'''
        pd.DataFrame(self.pnl, columns=['timestamp', 'PnL']).to_csv('./backtest/pnl.csv', index=False)
        pd.DataFrame(self.price_history, columns=['timestamp', 'best_a', 'best_b']).to_csv('./backtest/price_history.csv', index=False)
        pd.DataFrame(self.yes_orders, columns=['timestamp', 'price']).to_csv('./backtest/yes_orders.csv', index=False)
        pd.DataFrame(self.no_orders, columns=['timestamp', 'price']).to_csv('./backtest/no_orders.csv', index=False)
        
        return self.pnl

    def initiate_orderbook(self):
        '''Initializes the orderbook to the state at the start time'''        
        orderbook = OrderBook()
        
        with sqlite3.connect(self.db_path) as conn:
            # Get the most recent snapshot before the start time
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp, snapshot_data FROM orderbook_snapshots \
                        WHERE market_ticker=? AND timestamp <=? \
                        ORDER BY timestamp DESC \
                        LIMIT 1", (self.market_ticker, self.start_time))
            row = cursor.fetchone()
            if row is None:
                print("No data found for the specified market ticker and time range.")
                return None
            
            timestamp = row[0]
            snapshot_data = json.loads(row[1])
            try:
                for prc, qty in snapshot_data["no"]:
                    orderbook.update_order_book('no', prc, qty)

                for prc, qty in snapshot_data["yes"]:
                    orderbook.update_order_book('yes', prc, qty)
            except KeyError:
                print("Market is not two-sided.")
                return None

            # Fill in orderbook deltas between snapshot and start time
            cursor.execute("SELECT price, delta, side FROM orderbook_deltas \
                           WHERE market_ticker=? AND timestamp BETWEEN ? AND ?", 
                           (self.market_ticker, timestamp, self.start_time))
            rows = cursor.fetchall()
            for price, quantity, side in rows:
                if side == "yes":
                    orderbook.update_order_book('yes', price, quantity)
                elif side == "no":
                    orderbook.update_order_book('no', price, quantity)

        return orderbook              

def main():
    class RandomStrategy(Strategy):
        def __init__(self, cash):
            super().__init__(cash)
            buy = Order('KXBTC-25APR2117-B87250', 1, 0, 'market', 'yes')
            sell = Order('KXBTC-25APR2117-B87250', 1, 0, 'market', 'no')
            self.orders = [buy] + [None] * 8 + [sell]
        
        def generate_signal(self, orderbook):
            # Generate random buy/sell signals
            return [random.choice(self.orders)]
        
    strat = RandomStrategy(0)
    backtest = Backtest(strat, 'KXBTC-25APR2117-B87250', '2025-04-21 12:00:00', '2025-04-21 21:00:00', 0.1)
    backtest.run()
    backtest.report()

if __name__=='__main__':
    main()