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


def test_vision_detector_locates_H():
    """The camera H-detector back-projects the mark to within ~0.2 m up high."""
    from rocket_landing.vision import HVisionSensor
    env = RocketEnv(EnvConfig(seed=3, randomize=False))
    sensor = HVisionSensor(env)
    env.reset()
    # fly down with the classical controller until the H is clearly visible
    ctrl = ClassicalController()
    while env.altitude > 7.0:
        env.step(ctrl.act(env))
    det = sensor.detect()
    assert det.found
    err = float(np.linalg.norm(det.marker_world[:2] - env.marker_pos[:2]))
    assert err < 0.25


def test_vision_in_the_loop_landing():
    """Closing the loop on the camera still lands softly on the H."""
    from rocket_landing.vision import HVisionSensor, VisionController
    env = RocketEnv(EnvConfig(randomize=True))
    sensor = HVisionSensor(env)
    ctrl = VisionController(ClassicalController(), sensor)
    out = run_episode(env, ctrl, seed=2001)
    assert out["outcome"] == "soft-landing"
    assert out["horiz_err"] < 1.0
    assert ctrl.n_detections > 0  # the camera actually drove the alignment
