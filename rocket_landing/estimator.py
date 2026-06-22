"""Onboard state estimation: a loosely-coupled INS/GNSS filter.

The rocket no longer reads its true pose from the simulator. Instead it carries
an **IMU** (gyro + accelerometer) and a **GPS** receiver, and fuses them:

  * Attitude  -- gyro integration with a Mahony-style complementary correction
    from the accelerometer (which senses the specific-force / "up" direction).
  * Position/velocity -- strap-down INS mechanization (rotate the specific force
    to world, add gravity, integrate) bounded by low-rate, noisy GPS fixes.

This closes the realistic loop: noisy sensors -> estimator -> guidance. Ground
truth is used only to synthesize the measurements and to score the landing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mujoco

from rocket_landing import utils


@dataclass
class EstState:
    pos: np.ndarray    # world position estimate (m)
    vel: np.ndarray    # world velocity estimate (m/s)
    quat: np.ndarray   # orientation estimate (w, x, y, z)
    avel: np.ndarray   # world-frame angular velocity estimate (rad/s)


@dataclass
class IMUConfig:
    # navigation-grade IMU (typical for a launch vehicle)
    gyro_noise: float = 0.001     # rad/s, white noise std
    gyro_bias: float = 0.0005     # rad/s, constant bias magnitude (random/run)
    accel_noise: float = 0.03     # m/s^2, white noise std
    accel_bias: float = 0.02      # m/s^2, constant bias magnitude (random/run)


@dataclass
class GPSConfig:
    # differential/SBAS-grade GPS with Doppler velocity
    rate_hz: float = 10.0         # fix rate
    pos_noise_xy: float = 0.25    # m, horizontal position std
    pos_noise_z: float = 0.5      # m, vertical position std (worse for GPS)
    vel_noise: float = 0.06       # m/s, Doppler velocity std


@dataclass
class EstimatorConfig:
    imu: IMUConfig = field(default_factory=IMUConfig)
    gps: GPSConfig = field(default_factory=GPSConfig)
    # Attitude is propagated by strap-down gyro integration. During powered
    # flight the specific force is ~thrust/m along the body axis and gravity
    # cancels, so the accelerometer carries no usable attitude reference -- any
    # accel correction biases the estimate toward "upright". The accelerometer
    # instead drives the INS translation mechanization. Set mahony_kp>0 only to
    # trim attitude while coasting at ~1 g specific force.
    mahony_kp: float = 0.0        # accel->attitude correction gain
    accel_gate: float = 1.0       # m/s^2; trust accel as "up" only near 1 g
    gps_kp: float = 0.3           # GPS position innovation -> position
    gps_kv_pos: float = 0.05      # GPS position innovation -> velocity
    gps_kv_vel: float = 0.5       # GPS Doppler velocity innovation -> velocity
    init_pos_sigma: float = 2.0   # initial estimate error
    init_vel_sigma: float = 1.0
    init_att_sigma_deg: float = 0.5   # pre-launch fine alignment (gyrocompass)


class StateEstimator:
    """Fuses IMU + GPS into an ego-state estimate (``EstState``)."""

    GRAVITY = np.array([0.0, 0.0, -9.81])

    def __init__(self, env, config: Optional[EstimatorConfig] = None,
                 seed: Optional[int] = None):
        self.env = env
        self.cfg = config or EstimatorConfig()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        c = self.cfg
        self.rng = np.random.default_rng(self.rng.integers(1 << 31))
        # constant biases drawn once per flight
        self.gyro_bias = self.rng.normal(0, c.imu.gyro_bias, 3)
        self.accel_bias = self.rng.normal(0, c.imu.accel_bias, 3)
        # initialize the estimate from a coarse, errorful prior around truth
        self.pos = self.env.pos_true + self.rng.normal(0, c.init_pos_sigma, 3)
        self.vel = self.env.vel_true + self.rng.normal(0, c.init_vel_sigma, 3)
        q = np.array(self.env.quat_true, dtype=np.float64)
        dang = np.deg2rad(c.init_att_sigma_deg) * self.rng.normal(0, 1, 3)
        mujoco.mju_quatIntegrate(q, dang, 1.0)
        self.quat = q / np.linalg.norm(q)
        self._gps_accum = 0.0
        self._gps_period = 1.0 / c.gps.rate_hz

    # ----------------------------------------------------------------- measure
    def _read_imu(self):
        gyro = self.env.imu_gyro + self.gyro_bias \
            + self.rng.normal(0, self.cfg.imu.gyro_noise, 3)
        accel = self.env.imu_accel + self.accel_bias \
            + self.rng.normal(0, self.cfg.imu.accel_noise, 3)
        return gyro, accel

    def _read_gps(self):
        c = self.cfg.gps
        pos = self.env.pos_true + self.rng.normal(
            0, [c.pos_noise_xy, c.pos_noise_xy, c.pos_noise_z])
        vel = self.env.vel_true + self.rng.normal(0, c.vel_noise, 3)
        return pos, vel

    # ------------------------------------------------------------------ update
    def update(self, dt: Optional[float] = None) -> EstState:
        dt = self.env.dt if dt is None else dt
        gyro, accel = self._read_imu()
        w = gyro - self.gyro_bias_est()

        # --- attitude: complementary filter (gyro + accel "up" reference) ---
        R = utils.quat2mat(self.quat)
        up_pred_body = R.T @ np.array([0.0, 0.0, 1.0])   # world up in body
        a_norm = np.linalg.norm(accel)
        if a_norm > 1e-6 and abs(a_norm - 9.81) < self.cfg.accel_gate:
            up_meas_body = accel / a_norm
            # body-rate correction that rotates predicted-up toward measured-up
            e = np.cross(up_meas_body, up_pred_body)        # tilt error in body
            w = w + self.cfg.mahony_kp * e
        q = np.array(self.quat, dtype=np.float64)
        mujoco.mju_quatIntegrate(q, w, dt)                 # body-rate integration
        self.quat = q / np.linalg.norm(q)
        R = utils.quat2mat(self.quat)

        # --- position/velocity: strap-down INS mechanization ---
        a_world = R @ accel + self.GRAVITY                 # specific force -> accel
        self.vel = self.vel + a_world * dt
        self.pos = self.pos + self.vel * dt

        # --- GPS correction at its own (slower) rate ---
        self._gps_accum += dt
        if self._gps_accum >= self._gps_period:
            self._gps_accum = 0.0
            p_gps, v_gps = self._read_gps()
            innov_p = p_gps - self.pos
            self.pos = self.pos + self.cfg.gps_kp * innov_p
            self.vel = self.vel + self.cfg.gps_kv_pos * innov_p \
                + self.cfg.gps_kv_vel * (v_gps - self.vel)

        avel_world = R @ (gyro - self.gyro_bias_est())
        return EstState(self.pos.copy(), self.vel.copy(),
                        self.quat.copy(), avel_world)

    def gyro_bias_est(self) -> np.ndarray:
        # biases are not estimated online in this simple filter
        return np.zeros(3)


class EstimationController:
    """Wrap any ``act(env)`` controller so guidance runs on estimated state."""

    def __init__(self, base, estimator: StateEstimator):
        self.base = base
        self.estimator = estimator

    @property
    def stage(self):
        return getattr(self.base, "stage", 1)

    def reset(self):
        self.estimator.env.use_estimate = False
        self.estimator.reset()
        if hasattr(self.base, "reset"):
            self.base.reset()

    def act(self, env) -> np.ndarray:
        env.estimate = self.estimator.update()
        env.use_estimate = True
        return self.base.act(env)
