"""
Phase H — MiniGrid-MemoryS7 : la seconde moitié de l'argument "horizon de mémoire".

Compare, sur un POMDP a MEMOIRE LONGUE, deux approches dans sb3 (comparaison controlee) :
  1. PPO + FrameStack(4)  -> memoire EXPLICITE COURTE  -> doit plafonner au hasard (~0.5)
  2. RecurrentPPO (LSTM)  -> memoire RECURRENTE         -> doit depasser le hasard

Pourquoi sb3 pour les DEUX ici : sur MiniGrid on veut une comparaison parfaitement
controlee entre "memoire courte" et "memoire recurrente", avec la meme librairie de
reference. (Sur CartPole, le coeur du projet reste l'implementation from-scratch.)

Astuce cle : espace d'actions reduit a 3 (left, right, forward), ce qui accelere
fortement l'apprentissage sur les taches MiniGrid (les 4 autres actions sont inutiles ici).

Usage :
    python phase_H_minigrid_memory_sb3.py                      # defauts (3 seeds, 400k)
    python phase_H_minigrid_memory_sb3.py --seeds 1 2 --timesteps 300000
    python phase_H_minigrid_memory_sb3.py --configs recurrent  # un seul des deux

Duree indicative (8 envs) :
    ~130-150 fps CPU -> 400k ≈ 45-55 min / run.  GPU recommande pour RecurrentPPO.
"""
from pathlib import Path
import argparse
import time
import numpy as np
import pandas as pd
import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ENV_ID = "MiniGrid-MemoryS7-v0"


# ----------------------------- Wrappers -----------------------------
class ThreeActions(gym.ActionWrapper):
    """Reduit l'espace d'actions a {0:left, 1:right, 2:forward}."""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(3)
    def action(self, a):
        return int(a)


class FlattenImg(gym.ObservationWrapper):
    """Image symbolique 7x7x3 -> vecteur 147 normalise (/10)."""
    def __init__(self, env):
        super().__init__(env)
        n = int(np.prod(env.observation_space.shape))
        self.observation_space = gym.spaces.Box(0.0, 1.0, (n,), dtype=np.float32)
    def observation(self, obs):
        return obs.flatten().astype(np.float32) / 10.0


def make_env_fn(seed_offset=0):
    def _init():
        env = gym.make(ENV_ID)
        env = ImgObsWrapper(env)
        env = ThreeActions(env)
        env = FlattenImg(env)
        env = Monitor(env)
        return env
    return _init


def build_vec_env(n_envs=8, framestack=False):
    venv = DummyVecEnv([make_env_fn(i) for i in range(n_envs)])
    if framestack:
        venv = VecFrameStack(venv, n_stack=4)
    return venv


# ----------------------------- Logging callback -----------------------------
class CurveLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rows = []
    def _on_step(self):
        return True
    def _on_rollout_end(self):
        if len(self.model.ep_info_buffer) > 0:
            r = float(np.mean([e["r"] for e in self.model.ep_info_buffer]))
            l = float(np.mean([e["l"] for e in self.model.ep_info_buffer]))
            self.rows.append({"step": int(self.num_timesteps), "ep_rew_mean": r, "ep_len_mean": l})


# ----------------------------- Deterministic eval -----------------------------
def eval_ppo(model, framestack, n_ep=30, seed=0):
    venv = build_vec_env(n_envs=1, framestack=framestack)
    rets = []
    for ep in range(n_ep):
        obs = venv.reset()
        done = False; total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            total += float(reward[0]); done = bool(dones[0])
        rets.append(total)
    venv.close()
    return float(np.mean(rets)), float(np.std(rets))


def eval_recurrent(model, n_ep=30, seed=0):
    venv = build_vec_env(n_envs=1, framestack=False)
    rets = []
    for ep in range(n_ep):
        obs = venv.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        done = False; total = 0.0
        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            total += float(reward[0]); done = bool(dones[0])
            episode_starts = np.array([done], dtype=bool)
        rets.append(total)
    venv.close()
    return float(np.mean(rets)), float(np.std(rets))


# ----------------------------- Training -----------------------------
def train_one(kind, seed, timesteps, n_envs, device):
    common = dict(learning_rate=3e-4, n_steps=256, batch_size=256, n_epochs=4,
                  gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2,
                  seed=seed, verbose=0, device=device)
    if kind == "framestack":
        env = build_vec_env(n_envs=n_envs, framestack=True)
        model = PPO("MlpPolicy", env, **common)
    elif kind == "recurrent":
        env = build_vec_env(n_envs=n_envs, framestack=False)
        model = RecurrentPPO("MlpLstmPolicy", env, **common)
    else:
        raise ValueError(kind)

    cb = CurveLogger()
    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)
    dur = time.time() - t0
    env.close()

    if kind == "framestack":
        det_mean, det_std = eval_ppo(model, framestack=True, n_ep=30)
    else:
        det_mean, det_std = eval_recurrent(model, n_ep=30)

    return model, cb.rows, dur, det_mean, det_std


# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--timesteps", type=int, default=400_000)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--configs", nargs="+", default=["framestack", "recurrent"],
                    choices=["framestack", "recurrent"])
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    out_csv = ROOT / "results" / "csv" / "minigrid_memory_sb3"
    out_models = ROOT / "results" / "models" / "minigrid_memory_sb3"
    fig_dir = ROOT / "report" / "figures"
    for d in (out_csv, out_models, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    label = {"framestack": "PPO+FrameStack(4)", "recurrent": "RecurrentPPO-LSTM"}
    summary = []

    print("=" * 72)
    print("PHASE H — MiniGrid-MemoryS7 (POMDP a memoire longue)")
    print(f"Configs: {args.configs} | seeds: {args.seeds} | timesteps: {args.timesteps}")
    print("=" * 72)

    for kind in args.configs:
        for seed in args.seeds:
            print(f"\n[{label[kind]}] seed={seed} ...")
            model, rows, dur, det_m, det_s = train_one(
                kind, seed, args.timesteps, args.n_envs, args.device)
            # Save curve
            df = pd.DataFrame(rows); df["config"] = kind; df["seed"] = seed
            df.to_csv(out_csv / f"{kind}__seed{seed}.csv", index=False)
            # Save model
            model.save(out_models / f"{kind}__seed{seed}.zip")
            final_train = rows[-1]["ep_rew_mean"] if rows else float("nan")
            summary.append({"config": kind, "label": label[kind], "seed": seed,
                            "final_train_rew": round(final_train, 3),
                            "det_eval_mean": round(det_m, 3), "det_eval_std": round(det_s, 3)})
            print(f"   done in {dur:.0f}s | train_rew(fin)={final_train:.3f} | "
                  f"DET EVAL={det_m:.3f} +/- {det_s:.3f}")

    sdf = pd.DataFrame(summary)
    sdf.to_csv(out_csv / "minigrid_memory_summary.csv", index=False)

    print("\n" + "=" * 72)
    print("RECAP (eval deterministe, moyenne sur seeds)")
    for kind in args.configs:
        sub = sdf[sdf["config"] == kind]
        if len(sub):
            print(f"  {label[kind]:22s} : {sub['det_eval_mean'].mean():.3f} "
                  f"+/- {sub['det_eval_mean'].std():.3f}  (seeds: "
                  f"{', '.join(f'{v:.2f}' for v in sub['det_eval_mean'])})")

    # ---- Figure : courbes d'apprentissage ----
    plt.figure(figsize=(10, 6))
    colors = {"framestack": "#2E9E5B", "recurrent": "#1F6FB2"}
    for kind in args.configs:
        # moyenne des courbes sur seeds (interpolation sur grille commune)
        curves = []
        gmax = 0
        for seed in args.seeds:
            f = out_csv / f"{kind}__seed{seed}.csv"
            if not f.exists(): continue
            d = pd.read_csv(f)
            if len(d) < 2: continue
            curves.append((d["step"].values, d["ep_rew_mean"].values))
            gmax = max(gmax, d["step"].values[-1])
        if not curves: continue
        grid = np.linspace(curves[0][0][0], gmax, 100)
        interp = np.array([np.interp(grid, s, r) for s, r in curves])
        mean = interp.mean(0); std = interp.std(0)
        plt.plot(grid, mean, color=colors[kind], linewidth=2, label=label[kind])
        if len(curves) > 1:
            plt.fill_between(grid, mean - std, mean + std, color=colors[kind], alpha=0.18)
    plt.axhline(0.5, color="gray", ls=":", alpha=0.7, label="Hasard (memoire impossible)")
    plt.xlabel("Timesteps"); plt.ylabel("Recompense episodique moyenne")
    plt.title("MiniGrid-MemoryS7 : memoire recurrente vs frame-stacking (memoire courte)")
    plt.legend(); plt.grid(alpha=0.3); plt.ylim(0, 1)
    plt.tight_layout()
    fp = fig_dir / "minigrid_memory_recurrent_vs_framestack.png"
    plt.savefig(fp, dpi=150)
    print(f"\nFigure -> {fp}")
    print("=" * 72)


if __name__ == "__main__":
    main()
