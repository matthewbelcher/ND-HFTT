import pandas as pd
import statsmodels.api as sm
import logging

'''
This file handles trading signal generation and backtesting, including hedge ratio calculation and transaction costs.
'''

class StatArbStrategy:
    def __init__(self, price_df: pd.DataFrame, hedge_ratio=None, transaction_cost=0.0001):
        self.price_df = price_df
        self.hedge_ratio = hedge_ratio
        self.spread = None
        self.transaction_cost = transaction_cost
        logging.info(f"Initialized StatArbStrategy with transaction cost: {transaction_cost}")

    def fit_hedge_ratio(self):
        logging.debug("Fitting hedge ratio")
        y = self.price_df.iloc[:, 0]
        x = sm.add_constant(self.price_df.iloc[:, 1])
        try:
            model = sm.OLS(y, x).fit()
            self.hedge_ratio = model.params.iloc[1]
            logging.info(f"Hedge ratio: {self.hedge_ratio:.4f}")
            return self.hedge_ratio
        except Exception as e:
            logging.error(f"Failed to fit hedge ratio: {str(e)}")
            raise

    def compute_spread(self):
        if self.spread is None:
            if self.hedge_ratio is None:
                self.fit_hedge_ratio()
            s1, s2 = self.price_df.iloc[:, 0], self.price_df.iloc[:, 1]
            self.spread = s1 - self.hedge_ratio * s2
            logging.debug("Computed spread")
        return self.spread

    def generate_signals(self, entry_z=2.0, exit_z=1.0, window=20):
        logging.debug(f"Generating signals with entry_z={entry_z}, exit_z={exit_z}, window={window}")
        spread = self.compute_spread()
        mu = spread.rolling(window).mean()
        sigma = spread.rolling(window).std()
        z = (spread - mu) / sigma

        signals = pd.Series(0, index=z.index)
        signals[z > entry_z] = -1
        signals[z < -entry_z] = 1
        signals[abs(z) < exit_z] = 0
        signals = signals.ffill().fillna(0)
        logging.info(f"Generated {len(signals)} signals")
        return signals

    def backtest_returns(self, signals):
        logging.debug("Backtesting returns")
        s1 = self.price_df.iloc[:, 0].pct_change().fillna(0)
        s2 = self.price_df.iloc[:, 1].pct_change().fillna(0)
        returns = signals.shift(1) * (s1 - self.hedge_ratio * s2)
        
        signal_changes = signals.diff().fillna(0).abs() > 0
        costs = signal_changes * self.transaction_cost
        net_returns = (returns - costs).dropna()
        logging.info(f"Backtest complete, net returns length: {len(net_returns)}")
        return net_returns