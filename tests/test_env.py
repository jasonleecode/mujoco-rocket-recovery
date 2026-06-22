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


def test_imu_specific_force_upright():
    """A near-hovering upright rocket reads ~+g on the body-z accelerometer."""
    env = RocketEnv(EnvConfig(seed=0, randomize=False))
    env.reset()
    hover = env.mass * env.gravity / env.max_thrust
    for _ in range(40):
        env.step(np.array([hover, 0.0, 0.0]))
    accel = env.imu_accel
    assert abs(accel[2] - env.gravity) < 0.5
    assert np.linalg.norm(accel[:2]) < 0.5


def test_estimator_tracks_truth():
    """INS/GPS estimate converges to the true state during a descent."""
    from rocket_landing.estimator import StateEstimator
    from rocket_landing.utils import body_axis
    env = RocketEnv(EnvConfig(seed=5, randomize=True))
    env.reset()
    ctrl = ClassicalController()
    est = StateEstimator(env, seed=1)
    est.reset()
    pos_err, att_err = [], []
    done = False
    while not done:
        action = ctrl.act(env)          # control on truth: isolate estimator
        s = est.update()
        pos_err.append(np.linalg.norm(s.pos - env.pos_true))
        cos = np.clip(body_axis(env.quat_true, 2) @ body_axis(s.quat, 2), -1, 1)
        att_err.append(np.degrees(np.arccos(cos)))
        done = env.step(action).done
    half = len(pos_err) // 2
    assert np.mean(pos_err[half:]) < 1.0       # GPS-bounded position
    assert np.mean(att_err[half:]) < 4.0       # gyro-integrated attitude


@pytest.mark.skipif(not os.path.exists("models/mlp_policy.pt"),
                    reason="trained MLP policy not present")
def test_estimation_in_the_loop_landing():
    """Guidance on fused IMU+GPS state still lands softly."""
    from rocket_landing.estimator import StateEstimator, EstimationController
    from rocket_landing.guidance import TwoStageController
    env = RocketEnv(EnvConfig(randomize=True))
    base = TwoStageController(mlp=MLPController(checkpoint="models/mlp_policy.pt"))
    ctrl = EstimationController(base, StateEstimator(env, seed=7))
    successes = sum(run_episode(env, ctrl, seed=5000 + s).get("success", False)
                    for s in range(8))
    assert successes >= 7   # robust soft landings on estimated state


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
