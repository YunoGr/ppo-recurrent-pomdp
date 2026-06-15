"""
PPO-LSTM - version corrigee 2.

Changements vs version precedente :
1. ReLU au lieu de Tanh dans l'encodeur (suit CleanRL ppo_atari_lstm.py).
   La saturation du Tanh limitait le signal vers le LSTM.
2. lstm_hidden_size par defaut a 128 (comme CleanRL pour des taches non-triviales).
3. learning_rate plus haut (2.5e-4) pour compenser la sparse reward.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentLSTM(nn.Module):
    """PPO-LSTM avec encodeur MLP + ReLU (au lieu de Tanh) puis LSTM."""

    def __init__(self, obs_dim, n_actions, hidden_size=128, lstm_hidden_size=128):
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size

        # IMPORTANT : ReLU (pas Tanh) pour ne pas saturer le signal vers LSTM
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(hidden_size, lstm_hidden_size)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.actor = layer_init(nn.Linear(lstm_hidden_size, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(lstm_hidden_size, 1), std=1)

    def initial_state(self, batch_size, device):
        h = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        return h, c

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))

        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None, seq_len=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state


@dataclass
class PPOLSTMConfig:
    exp_name: str = "ppo_lstm"
    seed: int = 1
    cuda: bool = True

    env_id: str = "CartPole-v1"
    use_masked_velocity: bool = True
    total_timesteps: int = 500_000

    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    num_envs: int = 8
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

    save_dir: str = "results"

    @property
    def batch_size(self): return self.num_envs * self.num_steps
    @property
    def envs_per_minibatch(self): return self.num_envs // self.num_minibatches
    @property
    def num_iterations(self): return self.total_timesteps // self.batch_size
