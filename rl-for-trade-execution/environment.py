import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class OrderExecutionEnv(gym.Env):

    # Action meanings:
    # 0: do nothing
    # 1: passive small  — 10% of remaining, post at bid (low fill prob)
    # 2: passive large  — 20% of remaining, post at ask (moderate fill)
    # 3: aggressive     — 30% of remaining, take ask
    # 4: market order   — 50% of remaining, take ask immediately

    FILL_PROBS  = [0.0, 0.25, 0.60, 1.0, 1.0]   # probability of fill per action
    ORDER_FRACS = [0.0, 0.10, 0.20, 0.30, 0.50]  # fraction of remaining inventory

    def __init__(self, book_df, parent_order_size=100,
                 episode_length=50, tick_size=0.05,
                 inventory_penalty=2.0):
        super().__init__()
        self.book         = book_df.reset_index(drop=True)
        self.order_size   = parent_order_size
        self.ep_len       = episode_length
        self.tick         = tick_size
        self.inv_penalty  = inventory_penalty

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        max_start = len(self.book) - self.ep_len - 1
        self.idx          = np.random.randint(0, max_start)
        self.step_num     = 0
        self.remaining    = self.order_size
        self.total_cost   = 0.0
        self.total_filled = 0
        self.decision_price = self.book.at[self.idx, 'mid_price']
        return self._obs(), {}

    def _obs(self):
        r = self.book.iloc[self.idx]
        #State space
        return np.array([
            self.remaining / self.order_size,                        # inventory urgency
            1.0 - self.step_num / self.ep_len,                       # time urgency
            r['spread'],                                             # bid-ask spread
            r['imbalance_l1'],                                       # L1 book imbalance
            r['imbalance_l3'],                                       # L3 book imbalance
            r['ask_qty_1'] / self.order_size,                        # ask liquidity
            r['bid_qty_1'] / self.order_size,                        # bid liquidity
        ], dtype=np.float32)

    def _simulate_fill(self, action, row):
        if self.remaining <= 0 or action == 0:
            return 0, 0.0

        ask_p = row['ask_price_1']
        ask_q = row['ask_qty_1']
        if ask_p <= 0 or ask_q <= 0:
            return 0, 0.0

        # Determine order size
        qty = max(1, int(self.ORDER_FRACS[action] * self.remaining))
        qty = min(qty, self.remaining)

        # For passive orders, fill probabilistically and cap at available qty
        if action in (1, 2):
            if np.random.random() > self.FILL_PROBS[action]:
                return 0, 0.0
            fill_qty  = min(qty, int(ask_q))
            # Never fill below mid
            fill_price = max(ask_p - self.tick, row['mid_price']) if action == 1 else ask_p
            return fill_qty, fill_price

        # For aggressive/market orders, always fill, walk book if needed
        fill_qty  = min(qty, int(ask_q))
        fill_price = ask_p
        return fill_qty, fill_price

    def step(self, action):
        row = self.book.iloc[self.idx]
        fill_qty, fill_price = self._simulate_fill(action, row)

        reward = 0.0
        if fill_qty > 0:
            self.total_cost   += fill_qty * fill_price
            self.total_filled += fill_qty
            self.remaining    -= fill_qty
            # Per-step IS penalty (cost above decision price)
            reward -= (fill_price - self.decision_price) * fill_qty

        self.step_num += 1
        self.idx      += 1

        terminated = self.remaining <= 0
        truncated  = self.step_num >= self.ep_len

        # Terminal inventory penalty
        if truncated and self.remaining > 0:
            worst_price = self.book.iloc[self.idx - 1]['ask_price_1'] + self.book.iloc[self.idx - 1]['spread']
            penalty = self.inv_penalty * self.remaining * max(worst_price - self.decision_price, self.tick)
            reward -= penalty

        return self._obs(), reward, terminated, truncated, {}

    def implementation_shortfall(self):
        if self.total_filled == 0:
            return 0.0
        avg_exec = self.total_cost / self.total_filled
        return avg_exec - self.decision_price