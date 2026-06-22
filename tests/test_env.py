"""Smoke tests for the rocket recovery environment and controllers."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rocket_landing.env import RocketEnv, EnvConfig
from rocket_landing.controllers import ClassicalController, MLPController
from rocket_landing.rollout import run_episode


def test_model_loads_and_obs_shape():
    env = RocketEnv(EnvConfig(randomize=False))
    obs = env.reset()
    assert obs.shape == (13,)
    assert np.isfinite(obs).all()
    assert env.max_thrust > env.mass * env.gravity  # TWR > 1


def test_step_runs():
    env = RocketEnv(EnvConfig(randomize=False))
    env.reset()
    result = env.step(np.array([0.5, 0.0, 0.0]))
    assert result.obs.shape == (13,)
    assert np.isfinite(result.reward)


def test_classical_controller_soft_lands():
    env = RocketEnv(EnvConfig(seed=0, randomize=False))
    out = run_episode(env, ClassicalController(), seed=0)
    assert out["outcome"] == "soft-landing"
    assert out["success"] is True
    assert out["horiz_err"] < 1.0
    assert out["speed"] < 2.0


def test_classical_controller_randomized():
    env = RocketEnv(EnvConfig(randomize=True))
    ctrl = ClassicalController()
    successes = sum(run_episode(env, ctrl, seed=s).get("success", False)
                    for s in range(10))
    assert successes >= 9  # robust to randomized initial conditions


@pytest.mark.skipif(not os.path.exists("models/mlp_policy.pt"),
                    reason="trained MLP policy not present")
def test_two_stage_controller():
    from rocket_landing.guidance import TwoStageController
    env = RocketEnv(EnvConfig(randomize=True))
    mlp = MLPController(checkpoint="models/mlp_policy.pt")
    ctrl = TwoStageController(mlp=mlp, switch_altitude=12.0)
    out = run_episode(env, ctrl, seed=12345)
    assert out["outcome"] == "soft-landing"
    assert ctrl.stage == 2  # MLP took over for the terminal phase
