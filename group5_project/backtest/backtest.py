import pandas as pd
from datetime import datetime, time
from abc import ABC, abstractmethod
from enum import Enum

class Side(Enum):
    BUY = 'BUY'
    SELL = 'SELL'

class Order:
    def __init__(self, symbol: str, side: Side, quantity: int, limit: float, time: datetime):
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.limit = limit
        self.order_time = time

        self.fill_price = None
    
    def fill(self, price: float):
        self.fill_price = price

class Strategy(ABC):
    @abstractmethod
    def execute(self, data: pd.DataFrame) -> Order | None:
        pass

class Backtest:
    def __init__(self, strategy: Strategy, symbol: str, start_time: datetime, end_time: datetime, tick_file:str): 
        self.strategy = strategy
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        self.tick_file = tick_file
        self.data = None

        self.__load_data()
    
    def __load_data(self):
        self.data = pd.read_csv(self.tick_file)
        self.data['timestamp'] = pd.to_datetime(self.data['DATE'] + ' ' + self.data['TIME_M'])
        self.data.drop(columns=['DATE', 'SYM_SUFFIX'], inplace=True)
        self.data.set_index('timestamp', inplace=True)
        self.data = self.data[(self.data['SYM_ROOT'] == self.symbol) & (self.data.index >= pd.to_datetime(self.start_time)) & (self.data.index <= pd.to_datetime(self.end_time))]
        self.data['WMID'] = (self.data['BID'] * self.data['BIDSIZ'] + self.data['ASK'] * self.data['ASKSIZ']) / (self.data['BIDSIZ'] + self.data['ASKSIZ'])
        self.orders = []
    
    def run(self):
        for row in self.data.itertuples(name=None):
            order = self.strategy.execute(row)
            if order is not None:
                self.orders.append(order)

class ExampleStrategy(Strategy):
    def __init__(self, lower_limit: float, upper_limit: float):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.has_position = False

    def execute(self, data: tuple) -> Order | None:
        print(data)
        if not self.has_position and data[-1] < self.lower_limit:
            self.has_position = True
            return Order(data[6], Side.BUY, 1, self.lower_limit)
        elif self.has_position == True and data[-1] > self.upper_limit:
            self.has_position = False
            return Order(data[6], Side.SELL, 1, self.upper_limit)
        return None

    
if __name__ == '__main__':
    example_strategy = ExampleStrategy(515, 518)
    backtest = Backtest(example_strategy, 'VOO', datetime(2025,3,19,13,59,51), datetime(2025,3,19,14,0,5), 'data/market_ticks/03_19_25.csv')
    backtest.run()
    print("Orders executed:")
    for order in backtest.orders:
        print(f"{order.side.value} {order.quantity} of {order.symbol} at limit {order.limit}")