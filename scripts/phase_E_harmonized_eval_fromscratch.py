from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.phase_B_multiseeds import CartPoleArgs, train_ppo_mlp
from src.envs.pomdp_wrappers import MaskVelocityWrapper, FrameStackWrapper


def make_eval_env(use_masked=False, use_framestack=False, k=4):
    env = gym.make("CartPole-v1")

    if use_masked:
        env = MaskVelocityWrapper(env)

    if use_framestack:
        env = FrameStackWrapper(env, k=k)

    return env


@torch.no_grad()
def evaluate_mlp_agent(
    agent,
    config_name,
    seed,
    use_masked=False,
    use_framestack=False,
    k=4,
    n_eval_episodes=20,
):
    device = next(agent.parameters()).device
    agent.eval()

    rows = []

    for episode in range(1, n_eval_episodes + 1):
        env = make_eval_env(
            use_masked=use_masked,
            use_framestack=use_framestack,
            k=k,
        )

        obs, info = env.reset(seed=seed + 1000 + episode)

        done = False
        total_reward = 0.0
        episode_length = 0

        while not done:
            obs_tensor = torch.tensor(
                obs,
                dtype=torch.float32,
                device=device,
            ).view(1, -1)

            logits = agent.actor(obs_tensor)
            action = int(torch.argmax(logits, dim=-1).item())

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            episode_length += 1
            done = terminated or truncated

        env.close()

        rows.append({
            "config": config_name,
            "seed": seed,
            "episode": episode,
            "eval_return": total_reward,
            "episode_length": episode_length,
        })

    return pd.DataFrame(rows)


def main():
    seeds = [1, 2, 3]
    total_timesteps = 200_000
    n_eval_episodes = 20

    out_dir = ROOT / "results" / "csv" / "harmonized_eval"
    model_dir = ROOT / "results" / "models" / "harmonized_eval"
    fig_dir = ROOT / "report" / "figures"

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "config_name": "ppo_mlp_mdp",
            "label": "PPO-MLP / MDP",
            "use_masked": False,
            "use_framestack": False,
            "k": 4,
        },
        {
            "config_name": "ppo_mlp_pomdp",
            "label": "PPO-MLP / POMDP",
            "use_masked": True,
            "use_framestack": False,
            "k": 4,
        },
        {
            "config_name": "ppo_framestack_pomdp",
            "label": "PPO-FrameStack / POMDP",
            "use_masked": True,
            "use_framestack": True,
            "k": 4,
        },
    ]

    all_eval_rows = []
    summary_rows = []

    print("=" * 80)
    print("PHASE E — Harmonized evaluation for from-scratch PPO agents")
    print(f"Seeds: {seeds}")
    print(f"Training timesteps per run: {total_timesteps}")
    print(f"Evaluation episodes per seed: {n_eval_episodes}")
    print("=" * 80)

    for cfg in configs:
        config_name = cfg["config_name"]

        print("\n" + "-" * 80)
        print(f"CONFIG: {config_name}")
        print("-" * 80)

        for seed in seeds:
            print(f"\n[TRAIN] {config_name} | seed={seed}")

            args = CartPoleArgs(
                seed=seed,
                total_timesteps=total_timesteps,
                num_envs=4,
                num_steps=128,
                learning_rate=2.5e-4,
                update_epochs=4,
                ent_coef=0.01,
            )

            start = time.time()

            agent, history, train_duration = train_ppo_mlp(
                args,
                use_masked=cfg["use_masked"],
                use_framestack=cfg["use_framestack"],
                k=cfg["k"],
                verbose=True,
            )

            train_r50 = (
                float(np.mean(history["return"][-50:]))
                if len(history["return"]) >= 50
                else float(np.mean(history["return"]))
                if len(history["return"]) > 0
                else 0.0
            )

            model_path = model_dir / f"{config_name}__seed{seed}.pt"

            torch.save(
                {
                    "config_name": config_name,
                    "seed": seed,
                    "state_dict": agent.state_dict(),
                    "args": args.__dict__,
                    "use_masked": cfg["use_masked"],
                    "use_framestack": cfg["use_framestack"],
                    "k": cfg["k"],
                    "train_r50": train_r50,
                },
                model_path,
            )

            print(f"[EVAL] {config_name} | seed={seed}")

            eval_df = evaluate_mlp_agent(
                agent=agent,
                config_name=config_name,
                seed=seed,
                use_masked=cfg["use_masked"],
                use_framestack=cfg["use_framestack"],
                k=cfg["k"],
                n_eval_episodes=n_eval_episodes,
            )

            eval_csv = out_dir / f"{config_name}__seed{seed}.csv"
            eval_df.to_csv(eval_csv, index=False)

            mean_eval = float(eval_df["eval_return"].mean())
            std_eval = float(eval_df["eval_return"].std())

            all_eval_rows.append(eval_df)

            summary_rows.append({
                "config": config_name,
                "label": cfg["label"],
                "implementation": "from_scratch",
                "seed": seed,
                "train_return_last50": train_r50,
                "mean_eval_return": mean_eval,
                "std_eval_return": std_eval,
                "n_eval_episodes": n_eval_episodes,
                "model_path": str(model_path),
                "eval_csv": str(eval_csv),
            })

            total_duration = time.time() - start

            print(
                f"[DONE] {config_name} | seed={seed} | "
                f"train_r50={train_r50:.2f} | "
                f"eval={mean_eval:.2f} ± {std_eval:.2f} | "
                f"duration={total_duration:.1f}s"
            )

    all_eval = pd.concat(all_eval_rows, ignore_index=True)
    per_seed_summary = pd.DataFrame(summary_rows)

    all_eval_csv = out_dir / "fromscratch_all_eval_episodes.csv"
    per_seed_csv = out_dir / "fromscratch_per_seed_summary.csv"

    all_eval.to_csv(all_eval_csv, index=False)
    per_seed_summary.to_csv(per_seed_csv, index=False)

    global_summary = (
        per_seed_summary
        .groupby(["config", "label", "implementation"], as_index=False)
        .agg(
            mean_eval_return=("mean_eval_return", "mean"),
            std_across_seeds=("mean_eval_return", "std"),
            mean_train_return_last50=("train_return_last50", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )

    # Ajouter RecurrentPPO si le résumé SB3 existe
    sb3_summary_path = ROOT / "results" / "csv" / "sb3_sanity" / "sb3_recurrentppo_pomdp_summary.csv"

    if sb3_summary_path.exists():
        sb3_summary = pd.read_csv(sb3_summary_path)

        sb3_global = pd.DataFrame([{
            "config": "sb3_recurrentppo_pomdp",
            "label": "RecurrentPPO-LSTM / POMDP",
            "implementation": "sb3-contrib",
            "mean_eval_return": float(sb3_summary["mean_eval_return"].mean()),
            "std_across_seeds": float(sb3_summary["mean_eval_return"].std()),
            "mean_train_return_last50": np.nan,
            "n_seeds": int(sb3_summary["seed"].nunique()),
        }])

        global_summary = pd.concat([global_summary, sb3_global], ignore_index=True)

    final_csv = out_dir / "final_harmonized_results_summary.csv"
    global_summary.to_csv(final_csv, index=False)

    print("\n" + "=" * 80)
    print("FINAL HARMONIZED RESULTS")
    print("=" * 80)
    print(global_summary)
    print(f"\nCSV global sauvegardé : {final_csv}")

    # Figure finale
    plot_df = global_summary.copy()

    plt.figure(figsize=(9, 5))
    plt.bar(
        plot_df["label"],
        plot_df["mean_eval_return"],
        yerr=plot_df["std_across_seeds"],
        capsize=5,
    )
    plt.ylabel("Mean evaluation return")
    plt.title("Final comparison on CartPole variants")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 520)
    plt.tight_layout()

    fig_path = fig_dir / "final_harmonized_eval_comparison.png"
    plt.savefig(fig_path, dpi=220)
    plt.close()

    print(f"Figure finale sauvegardée : {fig_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()