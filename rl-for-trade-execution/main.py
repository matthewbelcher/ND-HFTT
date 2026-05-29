import warnings
warnings.filterwarnings("ignore")
 
from data_loader import get_client, fetch_book, fetch_trades, process_book, process_trades
from environment import OrderExecutionEnv
from baselines   import run_twap, run_vwap, print_baseline_summary
from train       import train_ppo
from evaluate    import evaluate_ppo, plot_reward_curve, plot_is_distribution, print_summary, plot_action_distribution, plot_action_over_time

# Configuration
START      = "2026-01-26T14:30:00"
END        = "2026-01-30T21:00:00"
LIMIT      = 50_000
TIMESTEPS  = 2_000_000


if __name__ == '__main__':
    # Data
    client = get_client()
    book   = process_book(fetch_book(client, START, END, limit=LIMIT))
    trades = process_trades(fetch_trades(client, START, END))

    # Baselines
    env          = OrderExecutionEnv(book)
    twap_results = run_twap(env)
    vwap_results = run_vwap(env)
    print_baseline_summary(twap_results, vwap_results)

    # Training
    model, reward_logger = train_ppo(book, total_timesteps=TIMESTEPS)

    # Evaluation
    ppo_results = evaluate_ppo(model, OrderExecutionEnv(book))
    print_summary(twap_results, vwap_results, ppo_results)

    # Plots
    plot_reward_curve(reward_logger, twap_results, vwap_results)
    plot_is_distribution(twap_results, vwap_results, ppo_results)
    plot_action_distribution(model, book)
    plot_action_over_time(model, book)
