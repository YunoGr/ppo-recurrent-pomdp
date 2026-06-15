from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import gymnasium as gym

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.envs.pomdp_wrappers import MaskVelocityWrapper


class EpisodeReturnLogger(gym.Wrapper):
    """Wrapper simple pour enregistrer les returns épisodiques."""

    def __init__(self, env):
        super().__init__(env)
        self.episode_returns = []
        self.current_return = 0.0

    def reset(self, **kwargs):
        self.current_return = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_return += reward

        if terminated or truncated:
            self.episode_returns.append(self.current_return)
            info = dict(info)
            info["episode_return"] = self.current_return

        return obs, reward, terminated, truncated, info


def make_cartpole_env(use_masked=False):
    def _init():
        env = gym.make("CartPole-v1")
        if use_masked:
            env = MaskVelocityWrapper(env)
        env = Monitor(env)
        return env
    return _init


def train_and_evaluate(config_name, use_masked=False, total_timesteps=200_000, seed=1):
    print("\n" + "=" * 70)
    print(f"Training {config_name}")
    print(f"use_masked={use_masked}, total_timesteps={total_timesteps}, seed={seed}")
    print("=" * 70)

    env = make_vec_env(
        make_cartpole_env(use_masked=use_masked),
        n_envs=4,
        seed=seed,
    )

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        verbose=1,
        seed=seed,
        device="auto",
    )

    start = time.time()
    model.learn(total_timesteps=total_timesteps)
    duration = time.time() - start

    # Évaluation avec gestion correcte des états LSTM
    eval_env = gym.make("CartPole-v1")
    if use_masked:
        eval_env = MaskVelocityWrapper(eval_env)

    n_eval_episodes = 20
    eval_returns = []

    for ep in range(n_eval_episodes):
        obs, info = eval_env.reset(seed=seed + 1000 + ep)
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        done = False
        total_reward = 0.0

        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
            episode_starts = np.array([done], dtype=bool)

        eval_returns.append(total_reward)

    eval_env.close()
    env.close()

    final_mean = float(np.mean(eval_returns))
    final_std = float(np.std(eval_returns))

    out_dir = ROOT / "results" / "csv" / "sb3_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "config": [config_name] * len(eval_returns),
        "seed": [seed] * len(eval_returns),
        "episode": list(range(1, len(eval_returns) + 1)),
        "eval_return": eval_returns,
    })

    out_csv = out_dir / f"{config_name}_seed{seed}.csv"
    df.to_csv(out_csv, index=False)

    model_path = out_dir / f"{config_name}_seed{seed}.zip"
    model.save(model_path)

    print("\nResult:")
    print(f"Mean eval return: {final_mean:.2f} ± {final_std:.2f}")
    print(f"Duration: {duration:.1f}s")
    print(f"CSV saved: {out_csv}")
    print(f"Model saved: {model_path}")

    return final_mean, final_std


if __name__ == "__main__":
    for seed in [2, 3]:
        train_and_evaluate(
            config_name="sb3_recurrentppo_pomdp",
            use_masked=True,
            total_timesteps=200_000,
            seed=seed,
        )