"""Stage-2 learned MLP controller for the terminal landing phase.

The policy is a small multilayer perceptron that maps the 13-D observation to a
3-D action ``[throttle, gimbal_x, gimbal_y]``. It is trained by behavior cloning
the classical expert (see ``scripts/train_mlp.py``) and refined to deliver a
precise, soft touchdown right on the "H" mark.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

OBS_DIM = 13
ACT_DIM = 3


class MLPPolicy(nn.Module):
    """Observation -> action. Output is squashed into valid action ranges."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden=(128, 128, 64)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, act_dim)]
        self.net = nn.Sequential(*layers)
        # observation normalization buffers (filled at train time)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.obs_mean) / (self.obs_std + 1e-6)
        raw = self.net(x)
        # throttle -> [0,1] via sigmoid; gimbals -> [-1,1] via tanh
        throttle = torch.sigmoid(raw[..., 0:1])
        gimbal = torch.tanh(raw[..., 1:3])
        return torch.cat([throttle, gimbal], dim=-1)

    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        self.obs_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(std, dtype=torch.float32))


class MLPController:
    """Wraps an ``MLPPolicy`` with the same ``act(env)`` interface."""

    def __init__(self, policy: Optional[MLPPolicy] = None,
                 checkpoint: Optional[str] = None, device: str = "cpu"):
        self.device = device
        if policy is not None:
            self.policy = policy
        elif checkpoint is not None:
            self.policy = self.load(checkpoint, device)
        else:
            self.policy = MLPPolicy()
        self.policy.to(device).eval()

    def reset(self):
        pass

    @torch.no_grad()
    def act(self, env) -> np.ndarray:
        obs = torch.as_tensor(env.get_obs(), dtype=torch.float32, device=self.device)
        a = self.policy(obs.unsqueeze(0)).squeeze(0).cpu().numpy()
        return a.astype(np.float32)

    @torch.no_grad()
    def act_obs(self, obs: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.policy(t).squeeze(0).cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------- persistence
    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"state_dict": self.policy.state_dict(),
                    "obs_dim": OBS_DIM, "act_dim": ACT_DIM}, path)

    @staticmethod
    def load(path: str, device: str = "cpu") -> MLPPolicy:
        ckpt = torch.load(path, map_location=device)
        policy = MLPPolicy(ckpt.get("obs_dim", OBS_DIM), ckpt.get("act_dim", ACT_DIM))
        policy.load_state_dict(ckpt["state_dict"])
        return policy
