"""
Phase B — Multi-seeds CartPole.

Re-entraine les 4 configurations CartPole avec 3 seeds chacune.
Sortie : 12 CSVs dans results/csv/multiseed/ que NB d'analyse va agreger.

Usage :
    python scripts/phase_B_multiseeds.py

Duree estimee :
    - CPU : ~45 min total (4 configs x 3 seeds x ~3-5 min/run)
    - GPU : ~15 min total
"""
from __future__ import annotations
import os, sys, time, random, argparse
from pathlib import Path
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# Ajout chemin projet
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.envs.pomdp_wrappers import MaskVelocityWrapper, FrameStackWrapper
from src.agents.ppo_lstm import AgentLSTM
import pandas as pd


# =================== Hyperparameters ===================
@dataclass
class CartPoleArgs:
    seed: int = 1
    cuda: bool = True
    total_timesteps: int = 200_000
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 128
    lstm_hidden_size: int = 128

    @property
    def batch_size(self): return self.num_envs * self.num_steps
    @property
    def minibatch_size(self): return self.batch_size // self.num_minibatches
    @property
    def num_iterations(self): return self.total_timesteps // self.batch_size


def set_global_seed(seed, det=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = det
    torch.backends.cudnn.benchmark = not det


def layer_init(layer, std=np.sqrt(2), b=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, b)
    return layer


class AgentMLP(nn.Module):
    """Tanh - pour CartPole."""
    def __init__(self, obs_dim, n_actions, h=64):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, h)), nn.Tanh(),
            layer_init(nn.Linear(h, h)), nn.Tanh(),
            layer_init(nn.Linear(h, 1), std=1.0))
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, h)), nn.Tanh(),
            layer_init(nn.Linear(h, h)), nn.Tanh(),
            layer_init(nn.Linear(h, n_actions), std=0.01))
    def get_value(self, x): return self.critic(x)
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None: action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)


def make_env(seed, idx, use_masked=False, use_framestack=False, k=4):
    def thunk():
        env = gym.make("CartPole-v1")
        if use_masked:
            env = MaskVelocityWrapper(env)
        if use_framestack:
            env = FrameStackWrapper(env, k=k)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env
    return thunk


# =================== Training PPO-MLP ===================
def train_ppo_mlp(args: CartPoleArgs, use_masked=False, use_framestack=False, k=4, verbose=False):
    set_global_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv([
        make_env(args.seed, i, use_masked, use_framestack, k) for i in range(args.num_envs)
    ])
    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n
    obs_dim = int(np.prod(obs_shape))

    agent = AgentMLP(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_buf = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    values_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    history = {"step": [], "return": []}
    global_step = 0; start = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1) / args.num_iterations
            for pg in optimizer.param_groups: pg["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs; dones_buf[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()
            actions_buf[step] = action; logprobs_buf[step] = logprob
            obs_np, reward, term, trunc, info = envs.step(action.cpu().numpy())
            rewards_buf[step] = torch.tensor(reward, dtype=torch.float32, device=device)
            done_np = np.logical_or(term, trunc)
            next_obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(done_np, dtype=torch.float32, device=device)
            if "final_info" in info:
                for ei in info["final_info"]:
                    if ei and "episode" in ei:
                        history["step"].append(global_step)
                        history["return"].append(float(ei["episode"]["r"]))
            elif "episode" in info and "_episode" in info:
                mask = info["_episode"]
                if np.any(mask):
                    for idx in np.where(mask)[0]:
                        history["step"].append(global_step)
                        history["return"].append(float(info["episode"]["r"][idx]))

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterm = 1.0 - next_done; nextvals = next_value
                else:
                    nextnonterm = 1.0 - dones_buf[t+1]; nextvals = values_buf[t+1]
                delta = rewards_buf[t] + args.gamma * nextvals * nextnonterm - values_buf[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterm * lastgaelam
            returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1,) + obs_shape)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1).long()
        b_adv = advantages.reshape(-1); b_ret = returns.reshape(-1); b_val = values_buf.reshape(-1)
        inds = np.arange(args.batch_size)

        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for s in range(0, args.batch_size, args.minibatch_size):
                mb = inds[s:s+args.minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_actions[mb])
                logratio = newlogprob - b_logprobs[mb]; ratio = logratio.exp()
                ma = b_adv[mb]
                if args.norm_adv: ma = (ma - ma.mean()) / (ma.std() + 1e-8)
                pg1 = -ma * ratio; pg2 = -ma * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                newvalue = newvalue.view(-1)
                v_clip = b_val[mb] + torch.clamp(newvalue - b_val[mb], -args.clip_coef, args.clip_coef)
                v_loss = 0.5 * torch.max((newvalue-b_ret[mb])**2, (v_clip-b_ret[mb])**2).mean()
                ent = entropy.mean()
                loss = pg_loss - args.ent_coef * ent + args.vf_coef * v_loss
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if verbose and (iteration % 50 == 0 or iteration == 1):
            r50 = np.mean(history["return"][-50:]) if history["return"] else 0
            sps = int(global_step / (time.time() - start))
            print(f"    iter {iteration:4d} | step {global_step:6d} | return(50)={r50:6.1f} | SPS={sps}")

    envs.close()
    duration = time.time() - start
    return agent, history, duration


# =================== Training PPO-LSTM ===================
def train_ppo_lstm(args: CartPoleArgs, use_masked=True, verbose=False):
    set_global_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    def mk(seed, idx):
        def thunk():
            env = gym.make("CartPole-v1")

            if use_masked:
                env = MaskVelocityWrapper(env)

            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed + idx)
            return env
        return thunk

    envs = gym.vector.SyncVectorEnv([mk(args.seed, i) for i in range(args.num_envs)])
    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n
    obs_dim = int(np.prod(obs_shape))
    # Debug:
    print(f"[DEBUG LSTM] use_masked={use_masked} | obs_shape={obs_shape} | obs_dim={obs_dim}")

    # LSTM hidden=64 (config qui marche le mieux) avec ReLU dans encoder
    agent = AgentLSTM(
        
        obs_dim,
        n_actions,
        hidden_size=args.hidden_size,
        lstm_hidden_size=args.lstm_hidden_size,
        
        ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    history = {"step": [], "return": []}
    global_step = 0; start = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_lstm_state = agent.initial_state(args.num_envs, device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1) / args.num_iterations
            for pg in optimizer.param_groups: pg["lr"] = frac * args.learning_rate
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs; dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                    next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action; logprobs[step] = logprob
            obs_np, reward, term, trunc, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_done_np = np.logical_or(term, trunc)
            next_obs = torch.Tensor(obs_np).to(device)
            next_done = torch.Tensor(next_done_np).to(device)
            if "final_info" in info:
                for ei in info["final_info"]:
                    if ei and "episode" in ei:
                        history["step"].append(global_step)
                        history["return"].append(float(ei["episode"]["r"]))
            elif "episode" in info and "_episode" in info:
                mask = info["_episode"]
                if np.any(mask):
                    for idx in np.where(mask)[0]:
                        history["step"].append(global_step)
                        history["return"].append(float(info["episode"]["r"][idx]))

        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_lstm_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterm = 1.0 - next_done; nextvals = next_value
                else:
                    nextnonterm = 1.0 - dones[t+1]; nextvals = values[t+1]
                delta = rewards[t] + args.gamma * nextvals * nextnonterm - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterm * lastgaelam
            returns_ = advantages + values

        b_obs = obs.reshape((-1,) + obs_shape); b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1).long(); b_dones = dones.reshape(-1)
        b_adv = advantages.reshape(-1); b_ret = returns_.reshape(-1); b_val = values.reshape(-1)
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)

        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for s in range(0, args.num_envs, envsperbatch):
                mbenvinds = envinds[s:s+envsperbatch]
                mb_inds = flatinds[:, mbenvinds].ravel()
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]; ratio = logratio.exp()
                ma = b_adv[mb_inds]
                if args.norm_adv: ma = (ma - ma.mean()) / (ma.std() + 1e-8)
                pg1 = -ma * ratio; pg2 = -ma * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                newvalue = newvalue.view(-1)
                v_clip = b_val[mb_inds] + torch.clamp(newvalue - b_val[mb_inds], -args.clip_coef, args.clip_coef)
                v_loss = 0.5 * torch.max((newvalue-b_ret[mb_inds])**2, (v_clip-b_ret[mb_inds])**2).mean()
                ent = entropy.mean()
                loss = pg_loss - args.ent_coef * ent + args.vf_coef * v_loss
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if verbose and (iteration % 50 == 0 or iteration == 1):
            r50 = np.mean(history["return"][-50:]) if history["return"] else 0
            sps = int(global_step / (time.time() - start))
            print(f"    iter {iteration:4d} | step {global_step:6d} | return(50)={r50:6.1f} | SPS={sps}")

    envs.close()
    duration = time.time() - start
    return agent, history, duration


# =================== Main loop ===================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args_cli = parser.parse_args()

    OUT = Path(args_cli.out_dir) if args_cli.out_dir else ROOT / "results" / "csv" / "multiseed"
    OUT.mkdir(parents=True, exist_ok=True)

    configs = [
        ("ppo_mlp_mdp",         dict(use_masked=False, use_framestack=False), "mlp"),
        ("ppo_mlp_pomdp",       dict(use_masked=True,  use_framestack=False), "mlp"),
        ("ppo_framestack_pomdp",dict(use_masked=True,  use_framestack=True,  k=4), "mlp"),
        ("ppo_lstm_pomdp",      dict(), "lstm"),
    ]

    total_start = time.time()
    print("="*70)
    print(f"PHASE B - Multi-seeds CartPole")
    print(f"Seeds: {args_cli.seeds}, total_timesteps: {args_cli.total_timesteps}")
    print(f"Total runs: {len(configs) * len(args_cli.seeds)}")
    print("="*70)

    for cfg_name, kwargs, agent_type in configs:
        print(f"\n[CONFIG] {cfg_name}")
        for seed in args_cli.seeds:
            t0 = time.time()
            args = CartPoleArgs(seed=seed, total_timesteps=args_cli.total_timesteps)
            print(f"  [seed={seed}] training...")
            if agent_type == "mlp":
                _, history, dur = train_ppo_mlp(args, verbose=args_cli.verbose, **kwargs)
            else:
                _, history, dur = train_ppo_lstm(args, verbose=args_cli.verbose)

            # Save CSV
            df = pd.DataFrame({
                "step": history["step"],
                "return": history["return"],
                "config": cfg_name,
                "seed": seed,
            })
            out_csv = OUT / f"{cfg_name}__seed{seed}.csv"
            df.to_csv(out_csv, index=False)
            r50 = np.mean(history["return"][-50:]) if history["return"] else 0
            print(f"  [seed={seed}] DONE in {dur:.0f}s | return(50 derniers)={r50:.1f} | -> {out_csv.name}")

    total_dur = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"PHASE B TERMINEE en {total_dur/60:.1f} min")
    print(f"CSVs sauvegardes dans : {OUT}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
