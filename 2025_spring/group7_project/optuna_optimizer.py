import numpy as np
import optuna
import pandas as pd
import logging
from stat_arb_strat import StatArbStrategy

'''
This file tunes entry/exit z-scores & window via Optuna, balancing trade frequency and PnL.
'''

class OptunaOptimizer:
    def __init__(self, train_df, trade_target=5000, penalty=1e6):
        self.train_df = train_df
        self.trade_target = trade_target
        self.penalty = penalty
        self.base_strat = StatArbStrategy(train_df)
        self.hedge = self.base_strat.fit_hedge_ratio()
        self.spread = self.base_strat.compute_spread()
        logging.info(f"Initialized OptunaOptimizer with trade_target={trade_target}, penalty={penalty}")

    def compute_trades_per_window(self, signals, freq="4h"):
        df = (
            signals
            .to_frame("sig")
            .assign(trade=lambda d: d.sig.diff().fillna(0).abs() > 0,
                    dt=lambda d: d.index)
        )
        counts = [
            grp.trade.sum()
            for _, grp in df.groupby(pd.Grouper(key="dt", freq=freq))
            if not grp.empty
        ]
        return np.array(counts)

    def objective(self, trial):
        entry_z = trial.suggest_float("entry_z", 0.05, 1.0)
        exit_z = trial.suggest_float("exit_z", 0.0, entry_z)
        window = trial.suggest_int("window", 5, 30)
        
        strat = StatArbStrategy(self.train_df)
        strat.fit_hedge_ratio()
        strat.compute_spread()
        signals = strat.generate_signals(entry_z, exit_z, window)
        pnl = strat.backtest_returns(signals).sum()
        
        trades4h = self.compute_trades_per_window(signals)
        min_trades = trades4h.min() if len(trades4h) > 0 else 0
        
        trade_penalty = max(0, (self.trade_target - min_trades)) * self.penalty / self.trade_target
        objective = -pnl + trade_penalty
        logging.debug(f"Trial: entry_z={entry_z:.3f}, exit_z={exit_z:.3f}, window={window}, pnl={pnl:.4f}, min_trades={min_trades}, objective={objective:.4f}")
        return objective

    def optimize(self, n_trials=50):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        logging.info(f"Best parameters: {study.best_params}, value: {study.best_value:.4f}")
        return study.best_params