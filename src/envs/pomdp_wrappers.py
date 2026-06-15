"""
Wrappers pour transformer des MDPs en POMDPs.

Le wrapper principal de ce projet est `MaskVelocityWrapper`, qui supprime
les composantes de vitesse de l'observation de CartPole. L'agent ne reçoit
plus que les positions (x, theta), pas les dérivées (x_dot, theta_dot).

Ce design suit l'esprit de Ni et al. (2022), "Recurrent Model-Free RL Can Be
a Strong Baseline for Many POMDPs", où des MDPs classiques sont transformés
en POMDPs en masquant l'information dynamique.

CartPole-v1 observation:
  [x, x_dot, theta, theta_dot]
  Index 0 : position du chariot
  Index 1 : vitesse du chariot
  Index 2 : angle du pole
  Index 3 : vitesse angulaire du pole

MaskVelocityWrapper supprime les indices 1 et 3 → l'obs devient [x, theta] (dim 2).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class MaskVelocityWrapper(gym.ObservationWrapper):
    """Masque les composantes de vitesse de CartPole-v1.

    L'agent reçoit (x, theta) au lieu de (x, x_dot, theta, theta_dot).
    Pour décider de l'action optimale, il doit donc INFÉRER les vitesses
    à partir de l'historique — ce qu'un MLP ne peut pas faire, mais un LSTM oui.
    """

    # Indices à GARDER (positions, pas vitesses)
    KEEP_INDICES = np.array([0, 2], dtype=np.int64)

    def __init__(self, env: gym.Env):
        super().__init__(env)
        low = env.observation_space.low[self.KEEP_INDICES]
        high = env.observation_space.high[self.KEEP_INDICES]
        self.observation_space = gym.spaces.Box(
            low=low, high=high, shape=(2,), dtype=env.observation_space.dtype
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation[self.KEEP_INDICES].astype(self.observation_space.dtype)


class FrameStackWrapper(gym.ObservationWrapper):
    """Empile les k dernières observations pour fournir un historique court à l'agent.

    Pour un MLP, c'est la baseline "low-tech" du LSTM : au lieu d'utiliser une
    mémoire récurrente, on donne explicitement les k derniers états en input.

    Attention : on aplatit pour rester compatible avec un MLP. L'observation
    finale est de dimension (k * obs_dim,).
    """

    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        self.k = k
        obs_dim = env.observation_space.shape[0]
        self._frames: list[np.ndarray] = []
        self.observation_space = gym.spaces.Box(
            low=np.tile(env.observation_space.low, k),
            high=np.tile(env.observation_space.high, k),
            shape=(k * obs_dim,),
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Au reset, on initialise le buffer avec la même obs k fois.
        self._frames = [obs.copy() for _ in range(self.k)]
        return self.observation(obs), info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self._frames.append(observation.copy())
        self._frames = self._frames[-self.k :]
        return np.concatenate(self._frames, axis=0).astype(
            self.observation_space.dtype
        )


def make_cartpole_pomdp(env_id: str = "CartPole-v1") -> gym.Env:
    """Helper : CartPole avec masquage des vitesses."""
    env = gym.make(env_id)
    env = MaskVelocityWrapper(env)
    return env


def make_cartpole_framestack(env_id: str = "CartPole-v1", k: int = 4) -> gym.Env:
    """Helper : CartPole-MaskedVelocity + FrameStack k."""
    env = gym.make(env_id)
    env = MaskVelocityWrapper(env)
    env = FrameStackWrapper(env, k=k)
    return env
