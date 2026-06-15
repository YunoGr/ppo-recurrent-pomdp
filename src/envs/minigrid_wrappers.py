"""
Wrappers pour MiniGrid : aplatir l'observation pour usage MLP.

MiniGrid-MemoryS7-v0 est un POMDP a memoire longue :
- L'agent commence dans une salle, voit un objet (cle coloree)
- Doit traverser un couloir (5-7 pas)
- A la fin, choisir entre 2 portes (matching avec l'objet vu au debut)

Ce POMDP est PARFAIT pour distinguer FrameStack (memoire courte) de LSTM (memoire longue) :
- L'objet n'est plus dans le champ de vision apres ~2-3 pas
- Frame Stacking k=4 ne suffit pas pour s'en souvenir
- Un LSTM peut encoder l'identite de l'objet dans son etat cache
"""
from __future__ import annotations
import gymnasium as gym
import numpy as np


class FlatMiniGridWrapper(gym.ObservationWrapper):
    """Aplatit l'image 7x7x3 de MiniGrid en vecteur 147-dim normalise.

    Garde seulement l'image (ignore direction et mission). L'observation
    devient utilisable directement par un MLP standard.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        img_shape = env.observation_space["image"].shape
        flat_dim = int(np.prod(img_shape))
        # On peut utiliser low=0/high=255 mais on normalise dans observation()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=10.0,
            shape=(flat_dim,), dtype=np.float32
        )

    def observation(self, obs):
        # /10 = normalisation simple. L'image MiniGrid contient des indices < 10
        # pour les types d'objets, et 0-5 pour les couleurs. Diviser par 10 garde
        # tout dans [0, 1.5] sans saturer.
        return obs["image"].flatten().astype(np.float32) / 10.0


def make_minigrid_env(env_id="MiniGrid-MemoryS7-v0"):
    """Helper pour creer un MiniGrid avec wrapper d'aplatissement."""
    env = gym.make(env_id)
    env = FlatMiniGridWrapper(env)
    return env
