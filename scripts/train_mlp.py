#!/usr/bin/env python3
"""Train the stage-2 terminal-landing MLP by behavior-cloning the classical expert.

Pipeline:
  1. Roll out the classical controller on many randomized descents, logging
     (obs, action) pairs only from the terminal phase (low altitude). A little
     exploration noise is injected on the *executed* action while the *recorded*
     target stays the expert action -- a DAgger-style trick that broadens the
     state coverage so the cloned policy is robust off the nominal path.
  2. Fit an MLP with MSE on the expert actions (with input normalization).
  3. Save the checkpoint and report a closed-loop evaluation of the full
     two-stage controller (classical hand-off -> MLP).

Usage:
    python scripts/train_mlp.py --episodes 400 --epochs 300
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rocket_landing.env import RocketEnv, EnvConfig
from rocket_landing.controllers import ClassicalController, MLPController
from rocket_landing.controllers.mlp import MLPPolicy
from rocket_landing.guidance import TwoStageController
from rocket_landing.rollout import evaluate

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def collect(args) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the expert and return terminal-phase (obs, action) arrays."""
    env = RocketEnv(EnvConfig(randomize=True))
    expert = ClassicalController()
    rng = np.random.default_rng(args.seed)
    obs_buf, act_buf = [], []

    for ep in range(args.episodes):
        env.cfg.seed = args.seed + ep
        obs = env.reset()
        expert.reset()
        done = False
        while not done:
            action = expert.act(env)
            # record expert target on terminal-phase states only
            if env.altitude <= args.switch_altitude:
                obs_buf.append(obs.copy())
                act_buf.append(action.copy())
                # inject exploration noise on the executed action
                exec_action = action.copy()
                exec_action[0] += rng.normal(0, args.noise * 0.5)
                exec_action[1:] += rng.normal(0, args.noise, size=2)
                exec_action = np.clip(exec_action, [0, -1, -1], [1, 1, 1])
            else:
                exec_action = action
            result = env.step(exec_action)
            obs, done = result.obs, result.done
        if (ep + 1) % 50 == 0:
            print(f"  collected {ep + 1}/{args.episodes} episodes, "
                  f"{len(obs_buf)} terminal samples")

    return np.asarray(obs_buf, dtype=np.float32), np.asarray(act_buf, dtype=np.float32)


def train(obs: np.ndarray, act: np.ndarray, args) -> MLPPolicy:
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    policy = MLPPolicy().to(device)
    policy.set_normalization(obs.mean(0), obs.std(0) + 1e-6)

    X = torch.as_tensor(obs, device=device)
    Y = torch.as_tensor(act, device=device)
    n = len(X)
    n_val = max(1, int(0.1 * n))
    perm = torch.randperm(n)
    tr, va = perm[n_val:], perm[:n_val]

    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.MSELoss()

    print(f"training on {n} samples ({device}) ...")
    for epoch in range(args.epochs):
        policy.train()
        idx = tr[torch.randperm(len(tr))]
        total = 0.0
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            opt.zero_grad()
            pred = policy(X[b])
            loss = loss_fn(pred, Y[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        if (epoch + 1) % 25 == 0 or epoch == 0:
            policy.eval()
            with torch.no_grad():
                vloss = loss_fn(policy(X[va]), Y[va]).item()
            print(f"  epoch {epoch + 1:4d}  train {total / len(tr):.5f}  val {vloss:.5f}")
    return policy.cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--noise", type=float, default=0.06,
                   help="exploration noise std on executed gimbal action")
    p.add_argument("--switch-altitude", dest="switch_altitude", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--out", default=os.path.join(MODEL_DIR, "mlp_policy.pt"))
    p.add_argument("--eval-episodes", type=int, default=50)
    args = p.parse_args()

    print("=== Stage-2 MLP behavior cloning ===")
    obs, act = collect(args)
    if len(obs) == 0:
        raise SystemExit("no terminal-phase samples collected")
    policy = train(obs, act, args)

    ctrl = MLPController(policy=policy)
    ctrl.save(args.out)
    print(f"saved policy -> {args.out}")

    print("=== closed-loop evaluation (two-stage controller) ===")
    env = RocketEnv(EnvConfig(randomize=True))
    two_stage = TwoStageController(mlp=ctrl, switch_altitude=args.switch_altitude)
    metrics = evaluate(env, two_stage, n=args.eval_episodes, base_seed=99999)
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
