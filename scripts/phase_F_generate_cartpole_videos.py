from pathlib import Path
import sys
import numpy as np
import torch
import gymnasium as gym
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.phase_B_multiseeds import AgentMLP
from src.envs.pomdp_wrappers import MaskVelocityWrapper, FrameStackWrapper

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None


def make_env(render_mode="rgb_array", use_masked=False, use_framestack=False, k=4):
    env = gym.make("CartPole-v1", render_mode=render_mode)

    if use_masked:
        env = MaskVelocityWrapper(env)

    if use_framestack:
        env = FrameStackWrapper(env, k=k)

    return env


def save_video(frames, out_path, fps=10, hold_seconds=2):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(frames) > 0:
        hold_frames = [frames[-1]] * int(fps * hold_seconds)
        frames = frames + hold_frames

    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Vidéo sauvegardée : {out_path}")


def record_random_agent(out_path, seed=1, max_steps=500):
    env = make_env(render_mode="rgb_array")
    obs, info = env.reset(seed=seed)

    frames = []
    done = False
    total_reward = 0

    for _ in range(max_steps):
        frames.append(env.render())
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated

        if done:
            break

    env.close()
    save_video(frames, out_path)

    print(f"[RANDOM] return={total_reward:.1f}, frames={len(frames)}")


def load_mlp_checkpoint(model_path, use_masked=False, use_framestack=False, k=4):
    env = make_env(
        render_mode="rgb_array",
        use_masked=use_masked,
        use_framestack=use_framestack,
        k=k,
    )

    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n
    env.close()

    checkpoint = torch.load(model_path, map_location="cpu")

    agent = AgentMLP(obs_dim, n_actions)
    agent.load_state_dict(checkpoint["state_dict"])
    agent.eval()

    return agent


@torch.no_grad()
def record_mlp_agent(
    model_path,
    out_path,
    seed=1,
    use_masked=False,
    use_framestack=False,
    k=4,
    max_steps=500,
):
    agent = load_mlp_checkpoint(
        model_path=model_path,
        use_masked=use_masked,
        use_framestack=use_framestack,
        k=k,
    )

    env = make_env(
        render_mode="rgb_array",
        use_masked=use_masked,
        use_framestack=use_framestack,
        k=k,
    )

    obs, info = env.reset(seed=seed)

    frames = []
    done = False
    total_reward = 0

    for _ in range(max_steps):
        frames.append(env.render())

        obs_tensor = torch.tensor(obs, dtype=torch.float32).view(1, -1)
        logits = agent.actor(obs_tensor)
        action = int(torch.argmax(logits, dim=-1).item())

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated

        if done:
            break

    env.close()
    save_video(frames, out_path)

    print(f"[MLP] {out_path.name} | return={total_reward:.1f}, frames={len(frames)}")


def record_recurrentppo_agent(
    model_path,
    out_path,
    seed=1,
    use_masked=True,
    max_steps=500,
):
    if RecurrentPPO is None:
        raise ImportError("sb3-contrib n'est pas installé. Fais : pip install sb3-contrib")

    model = RecurrentPPO.load(model_path)

    env = make_env(
        render_mode="rgb_array",
        use_masked=use_masked,
        use_framestack=False,
    )

    obs, info = env.reset(seed=seed)

    frames = []
    done = False
    total_reward = 0

    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    for _ in range(max_steps):
        frames.append(env.render())

        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated
        episode_starts = np.array([done], dtype=bool)

        if done:
            break

    env.close()
    save_video(frames, out_path)

    print(f"[RecurrentPPO] {out_path.name} | return={total_reward:.1f}, frames={len(frames)}")


def main():
    video_dir = ROOT / "results" / "videos"
    model_dir = ROOT / "results" / "models" / "harmonized_eval"
    sb3_dir = ROOT / "results" / "csv" / "sb3_sanity"

    video_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE F — Génération des vidéos CartPole")
    print("=" * 80)

    # 1. Agent aléatoire
    record_random_agent(
        out_path=video_dir / "random_cartpole.mp4",
        seed=1,
    )

    # 2. PPO-MLP sur CartPole complet
    record_mlp_agent(
        model_path=model_dir / "ppo_mlp_mdp__seed1.pt",
        out_path=video_dir / "ppo_mlp_mdp_seed1.mp4",
        seed=1001,
        use_masked=False,
        use_framestack=False,
    )

    # 3. PPO-MLP sur CartPole-MaskedVelocity
    record_mlp_agent(
        model_path=model_dir / "ppo_mlp_pomdp__seed1.pt",
        out_path=video_dir / "ppo_mlp_pomdp_seed1.mp4",
        seed=1001,
        use_masked=True,
        use_framestack=False,
    )

    # 4. PPO-FrameStack sur CartPole-MaskedVelocity
    record_mlp_agent(
        model_path=model_dir / "ppo_framestack_pomdp__seed3.pt",
        out_path=video_dir / "ppo_framestack_pomdp_seed3.mp4",
        seed=1003,
        use_masked=True,
        use_framestack=True,
        k=4,
    )

    # 5. RecurrentPPO-LSTM sur CartPole-MaskedVelocity
    record_recurrentppo_agent(
        model_path=sb3_dir / "sb3_recurrentppo_pomdp_seed2.zip",
        out_path=video_dir / "sb3_recurrentppo_pomdp_seed2.mp4",
        seed=1002,
        use_masked=True,
    )

    print("=" * 80)
    print(f"Vidéos disponibles dans : {video_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()