import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from environment import OrderExecutionEnv


class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True


def train_ppo(book_df, total_timesteps=1_000_000, n_envs=8):
    def make_env():
        return Monitor(OrderExecutionEnv(book_df))

    vec_env  = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    eval_env = Monitor(OrderExecutionEnv(book_df))

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        device="cpu",
    )

    reward_logger = RewardLoggerCallback()
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([EvalCallback(eval_env, eval_freq=5000, verbose=1), reward_logger]),
    )

    return model, reward_logger