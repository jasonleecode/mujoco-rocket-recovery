"""Episode rollout helpers used by scripts (eval, data collection)."""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from rocket_landing.env import RocketEnv


def run_episode(env: RocketEnv, controller, record: bool = False,
                seed: Optional[int] = None):
    """Run one episode. ``controller`` exposes ``reset()`` and ``act(env)``.

    Returns a dict with the final ``info`` plus optional logged trajectory.
    """
    if seed is not None:
        env.cfg.seed = seed
    obs = env.reset()
    if hasattr(controller, "reset"):
        controller.reset()

    obs_log: List[np.ndarray] = []
    act_log: List[np.ndarray] = []
    traj: List[np.ndarray] = []

    info = {"outcome": "running"}
    done = False
    while not done:
        action = controller.act(env)
        if record:
            obs_log.append(obs.copy())
            act_log.append(np.asarray(action, dtype=np.float32))
            traj.append(env.rocket_pos.copy())
        result = env.step(action)
        obs = result.obs
        done = result.done
        info = result.info

    out = dict(info)
    if record:
        out["obs"] = np.asarray(obs_log, dtype=np.float32)
        out["act"] = np.asarray(act_log, dtype=np.float32)
        out["traj"] = np.asarray(traj, dtype=np.float32)
    return out


def evaluate(env: RocketEnv, controller, n: int = 50, base_seed: int = 1000):
    """Run ``n`` randomized episodes; return aggregate metrics."""
    successes, horiz, speed, tilt = 0, [], [], []
    outcomes = {}
    for i in range(n):
        env.cfg.randomize = True
        out = run_episode(env, controller, seed=base_seed + i)
        successes += int(out.get("success", False))
        horiz.append(out.get("horiz_err", np.nan))
        speed.append(out.get("speed", np.nan))
        tilt.append(np.rad2deg(out.get("tilt", np.nan)))
        outcomes[out.get("outcome", "?")] = outcomes.get(out.get("outcome", "?"), 0) + 1
    return {
        "n": n,
        "success_rate": successes / n,
        "horiz_err_mean": float(np.nanmean(horiz)),
        "touchdown_speed_mean": float(np.nanmean(speed)),
        "tilt_deg_mean": float(np.nanmean(tilt)),
        "outcomes": outcomes,
    }
