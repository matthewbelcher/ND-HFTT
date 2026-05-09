import backtest_engine as be

def main():
    class TrendStrategy(be.Strategy):
        def __init__(self, ticker, multiplier, trigger):
            super().__init__(0)
            self.ema = 0
            self.multiplier = multiplier
            self.trigger = trigger
            self.last_price = None
            self.buy = be.Order(ticker, 1, 0, 'market', 'yes')
            self.sell = be.Order(ticker, 1, 0, 'market', 'no')

        def generate_signal(self, orderbook):
            mid = orderbook.get_mid_price()
            if not self.last_price:
                self.last_price = mid
                return None
            
            ret = (mid - self.last_price) / self.last_price
            
            self.last_price = mid
            self.ema = ret + self.multiplier * self.ema

            if orderbook.get_best_ask()[0] - orderbook.get_best_bid()[0] > 30:
                return None

            if self.ema > self.trigger:
                self.ema = self.ema * 0.5
                return self.buy
            elif self.ema < -self.trigger:
                self.ema = self.ema * 0.5
                return self.sell
            
    strat = TrendStrategy('KXBTC-25APR2117-B87250', 0.8, 0.05)
    backtest = be.Backtest(strat, 'KXBTC-25APR2117-B87250', '2025-04-21 12:00:00', '2025-04-21 21:00:00', 10)
    backtest.run()
    backtest.report()

if __name__ == "__main__":
    main()
