import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environment import OrderExecutionEnv


def evaluate_ppo(model, env, n_episodes=200):
    results = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        results.append({
            'IS':         env.implementation_shortfall(),
            'filled':     env.total_filled,
            'pct_filled': env.total_filled / env.order_size,
        })
    return pd.DataFrame(results)


def plot_reward_curve(reward_logger, twap_results, vwap_results, window=100):
    rewards  = reward_logger.episode_rewards
    smoothed = pd.Series(rewards).rolling(window).mean()
    plt.figure(figsize=(12, 4))
    plt.plot(smoothed, color='steelblue', label=f'Smoothed (window={window})')
    plt.axhline(y=-twap_results['IS'].mean()*100, color='r', linestyle='--', label=f'TWAP IS={twap_results["IS"].mean():.4f}')
    plt.axhline(y=-vwap_results['IS'].mean()*100, color='g', linestyle='--', label=f'VWAP IS={vwap_results["IS"].mean():.4f}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO Training Reward Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reward_curve.png', dpi=150)
    plt.show()


def plot_is_distribution(twap_results, vwap_results, ppo_results):
    bins = np.linspace(-0.05, 0.15, 50)
    plt.figure(figsize=(12, 4))
    plt.hist(twap_results['IS'], bins=bins, alpha=0.5, label='TWAP', color='red')
    plt.hist(vwap_results['IS'], bins=bins, alpha=0.5, label='VWAP', color='green')
    plt.hist(ppo_results['IS'],  bins=bins, alpha=0.5, label='PPO',  color='steelblue')
    plt.axvline(twap_results['IS'].mean(), color='red',       linestyle='--')
    plt.axvline(vwap_results['IS'].mean(), color='green',     linestyle='--')
    plt.axvline(ppo_results['IS'].mean(),  color='steelblue', linestyle='--')
    plt.xlabel('Implementation Shortfall')
    plt.ylabel('Frequency')
    plt.title('IS Distribution: TWAP vs VWAP vs PPO')
    plt.legend()
    plt.tight_layout()
    plt.savefig('is_distribution.png', dpi=150)
    plt.show()


def print_summary(twap_results, vwap_results, ppo_results):
    summary = pd.DataFrame({
        'Strategy':   ['TWAP', 'VWAP', 'PPO'],
        'Mean IS':    [twap_results['IS'].mean(), vwap_results['IS'].mean(), ppo_results['IS'].mean()],
        'Std IS':     [twap_results['IS'].std(),  vwap_results['IS'].std(),  ppo_results['IS'].std()],
        'Pct Filled': [twap_results['pct_filled'].mean(), vwap_results['pct_filled'].mean(), ppo_results['pct_filled'].mean()],
    })
    summary['Mean IS']    = summary['Mean IS'].map('{:.4f}'.format)
    summary['Std IS']     = summary['Std IS'].map('{:.4f}'.format)
    summary['Pct Filled'] = summary['Pct Filled'].map('{:.2%}'.format)
    print(summary.to_string(index=False))


def plot_action_distribution(model, book_df, n_episodes=200):
    actions = []
    env = OrderExecutionEnv(book_df)
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            actions.append(int(action))
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    labels = ['Do Nothing', 'Passive Small', 'Passive Large', 'Aggressive', 'Market']
    plt.figure(figsize=(8, 4))
    plt.bar(labels, [actions.count(i) for i in range(5)], color='steelblue')
    plt.ylabel('Count')
    plt.title('PPO Action Distribution')
    plt.tight_layout()
    plt.savefig('action_distribution.png', dpi=150)
    plt.show()


def plot_action_over_time(model, book_df, n_episodes=200):
    action_by_step = [[] for _ in range(50)]
    env = OrderExecutionEnv(book_df)
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, step = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_by_step[step].append(int(action))
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1
    plt.figure(figsize=(10, 4))
    plt.plot([np.mean(a) for a in action_by_step if a], color='steelblue')
    plt.xlabel('Step')
    plt.ylabel('Average Action')
    plt.title('PPO Average Aggression Level Over Episode')
    plt.yticks([0,1,2,3,4], ['Nothing','Passive Sm','Passive Lg','Aggressive','Market'])
    plt.tight_layout()
    plt.savefig('action_over_time.png', dpi=150)
    plt.show()