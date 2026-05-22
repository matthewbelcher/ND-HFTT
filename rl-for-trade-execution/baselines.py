import pandas as pd
from environment import OrderExecutionEnv
 
def run_twap(env, n_episodes=100):
    results = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            # Equal slices at every step (always action 3 - aggressive, 30%)
            action = 3
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        results.append({
            'IS': env.implementation_shortfall(),
            'filled': env.total_filled,
            'pct_filled': env.total_filled / env.order_size
        })
    return pd.DataFrame(results)

def run_vwap(env, n_episodes=100):
    results = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done:
            # aggression scales with remaining time urgency
            time_remaining = 1.0 - step / env.ep_len
            if time_remaining > 0.6:
                action = 1  # early: be passive
            elif time_remaining > 0.3:
                action = 2  # mid: moderate
            else:
                action = 3  # late: aggressive to ensure completion
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1
        results.append({
            'IS': env.implementation_shortfall(),
            'filled': env.total_filled,
            'pct_filled': env.total_filled / env.order_size
        })
    return pd.DataFrame(results)

def print_baseline_summary(twap_results, vwap_results):
    for label, df in [("TWAP", twap_results), ("VWAP", vwap_results)]:
        print(f"=== {label} ===")
        print(f"  Mean IS:    {df['IS'].mean():.4f}")
        print(f"  Std IS:     {df['IS'].std():.4f}")
        print(f"  Pct Filled: {df['pct_filled'].mean():.2%}\n")