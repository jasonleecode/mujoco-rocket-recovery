"""MuJoCo simulation environment for vector-thrust rocket recovery."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mujoco

from rocket_landing import utils

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "rocket.xml")

# Geometry constants tied to models/rocket.xml.
LEG_TIP_OFFSET = 6.17  # distance from rocket CoM (z=0) down to the leg tips
PAD_TOP_Z = 0.2        # world z of the pad top surface


@dataclass
class EnvConfig:
    model_path: str = _MODEL_PATH
    control_hz: float = 50.0          # controller update rate
    sim_hz: float = 500.0             # physics rate (matches timestep 0.002)
    max_seconds: float = 40.0
    # initial-condition randomization (used by reset)
    init_height: float = 25.0         # leg-tip height above the pad
    init_xy_range: float = 6.0        # horizontal offset magnitude
    init_vz: float = -8.0             # initial vertical speed
    init_vxy_range: float = 2.5
    init_tilt_deg: float = 8.0
    init_avel_range: float = 0.1
    seed: Optional[int] = None
    randomize: bool = True


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class RocketEnv:
    """Thin wrapper around MjModel/MjData with a normalized action interface.

    Action (np.ndarray shape (3,)), all in [-1, 1] except throttle in [0, 1]:
        a[0] = throttle  in [0, 1]   -> main engine thrust
        a[1] = gimbal_x  in [-1, 1]  -> nozzle deflection about body x
        a[2] = gimbal_y  in [-1, 1]  -> nozzle deflection about body y
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        self.cfg = config or EnvConfig()
        self.model = mujoco.MjModel.from_xml_path(self.cfg.model_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 1.0 / self.cfg.sim_hz
        self.n_substeps = int(round(self.cfg.sim_hz / self.cfg.control_hz))
        self.dt = self.n_substeps * self.model.opt.timestep
        self.rng = np.random.default_rng(self.cfg.seed)

        self._gimbal_limit = float(self.model.actuator_ctrlrange[0][1])  # rad
        self.max_thrust = float(self.model.actuator_ctrlrange[2][1])     # N
        self.mass = float(self.model.body_subtreemass[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rocket")])
        self.gravity = float(-self.model.opt.gravity[2])

        self._sid = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, n)
                     for n in ("rocket_pos", "rocket_quat", "rocket_vel",
                               "rocket_avel", "marker_pos", "imu_gyro",
                               "imu_accel")}
        self._sadr = {k: self.model.sensor_adr[v] for k, v in self._sid.items()}
        self.time = 0.0
        # optional vision-estimated marker position (set by a vision wrapper).
        # When present, observations/guidance use this instead of ground truth;
        # termination metrics always use the true marker (honest scoring).
        self.marker_estimate: Optional[np.ndarray] = None
        # optional ego-state estimate (set by a StateEstimator wrapper). When
        # use_estimate is True, guidance/observations read the estimate while
        # scoring still uses ground truth.
        self.estimate = None
        self.use_estimate = False

    # ------------------------------------------------------------------ sensors
    def _sensor(self, name: str, dim: int) -> np.ndarray:
        a = self._sadr[name]
        return np.array(self.data.sensordata[a:a + dim])

    # -- ground-truth state (used for scoring + to synthesize IMU/GPS) --
    @property
    def pos_true(self) -> np.ndarray:
        return self._sensor("rocket_pos", 3)

    @property
    def quat_true(self) -> np.ndarray:
        return self._sensor("rocket_quat", 4)

    @property
    def vel_true(self) -> np.ndarray:
        return self._sensor("rocket_vel", 3)

    @property
    def avel_true(self) -> np.ndarray:
        """True angular velocity in the world frame."""
        return self._sensor("rocket_avel", 3)

    @property
    def imu_gyro(self) -> np.ndarray:
        """True body-frame angular rate (the gyro measures this + noise/bias)."""
        return self._sensor("imu_gyro", 3)

    @property
    def imu_accel(self) -> np.ndarray:
        """True body-frame specific force (accelerometer measures this + noise)."""
        return self._sensor("imu_accel", 3)

    @property
    def altitude_true(self) -> float:
        return float(self.pos_true[2] - LEG_TIP_OFFSET - PAD_TOP_Z)

    # -- state seen by guidance: estimator output if present, else truth --
    @property
    def rocket_pos(self) -> np.ndarray:
        if self.use_estimate and self.estimate is not None:
            return self.estimate.pos
        return self.pos_true

    @property
    def rocket_quat(self) -> np.ndarray:
        if self.use_estimate and self.estimate is not None:
            return self.estimate.quat
        return self.quat_true

    @property
    def rocket_vel(self) -> np.ndarray:
        if self.use_estimate and self.estimate is not None:
            return self.estimate.vel
        return self.vel_true

    @property
    def rocket_avel(self) -> np.ndarray:
        if self.use_estimate and self.estimate is not None:
            return self.estimate.avel
        return self.avel_true

    @property
    def marker_pos(self) -> np.ndarray:
        """Ground-truth H-mark position (used for scoring)."""
        return self._sensor("marker_pos", 3)

    @property
    def marker_pos_meas(self) -> np.ndarray:
        """Marker position as 'measured' by guidance: the vision estimate if a
        vision wrapper has set one, otherwise ground truth."""
        return self.marker_estimate if self.marker_estimate is not None \
            else self.marker_pos

    @property
    def altitude(self) -> float:
        """Leg-tip height above the pad top (m), from the guidance state."""
        return float(self.rocket_pos[2] - LEG_TIP_OFFSET - PAD_TOP_Z)

    # ------------------------------------------------------------------- reset
    def reset(self, randomize: Optional[bool] = None) -> np.ndarray:
        if self.cfg.seed is not None:
            self.rng = np.random.default_rng(self.cfg.seed)
        randomize = self.cfg.randomize if randomize is None else randomize
        mujoco.mj_resetData(self.model, self.data)

        com_z = self.cfg.init_height + LEG_TIP_OFFSET + PAD_TOP_Z
        pos = np.array([0.0, 0.0, com_z])
        vel = np.array([0.0, 0.0, self.cfg.init_vz])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        avel = np.zeros(3)

        if randomize:
            r = self.cfg.init_xy_range
            pos[0] += self.rng.uniform(-r, r)
            pos[1] += self.rng.uniform(-r, r)
            vel[0] += self.rng.uniform(-self.cfg.init_vxy_range, self.cfg.init_vxy_range)
            vel[1] += self.rng.uniform(-self.cfg.init_vxy_range, self.cfg.init_vxy_range)
            vel[2] += self.rng.uniform(-1.5, 1.5)
            # small random tilt
            t = np.deg2rad(self.cfg.init_tilt_deg)
            axis = self.rng.uniform(-1, 1, 3); axis[2] = 0
            n = np.linalg.norm(axis)
            if n > 1e-6:
                axis /= n
                ang = self.rng.uniform(0, t)
                quat = np.array([np.cos(ang / 2),
                                 *(np.sin(ang / 2) * axis)])
            avel += self.rng.uniform(-self.cfg.init_avel_range,
                                     self.cfg.init_avel_range, 3)

        self.data.qpos[0:3] = pos
        self.data.qpos[3:7] = quat
        self.data.qvel[0:3] = vel
        self.data.qvel[3:6] = avel
        self.marker_estimate = None
        self.estimate = None
        self.use_estimate = False
        mujoco.mj_forward(self.model, self.data)
        self.time = 0.0
        return self.get_obs()

    # -------------------------------------------------------------------- step
    def _apply_action(self, action: np.ndarray) -> None:
        a = np.asarray(action, dtype=np.float64)
        throttle = float(np.clip(a[0], 0.0, 1.0))
        gx = float(np.clip(a[1], -1.0, 1.0)) * self._gimbal_limit
        gy = float(np.clip(a[2], -1.0, 1.0)) * self._gimbal_limit
        self.data.ctrl[0] = gx
        self.data.ctrl[1] = gy
        self.data.ctrl[2] = throttle * self.max_thrust

    def step(self, action: np.ndarray) -> StepResult:
        self._apply_action(action)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.time += self.dt
        obs = self.get_obs()
        done, info = self._termination()
        reward = self._reward(action, info)
        return StepResult(obs, reward, done, info)

    # ------------------------------------------------------------- observation
    def get_obs(self) -> np.ndarray:
        """Observation used by both controllers and the MLP policy.

        Layout (13,):
            [0:3]  rocket position relative to H mark  (dx, dy, dz)
            [3:6]  linear velocity                     (vx, vy, vz)
            [6:8]  body +Z tilt projected on world XY   (lean_x, lean_y)
            [8]    cos(tilt)                            (1 = perfectly upright)
            [9:12] angular velocity
            [12]   altitude (leg-tip height above pad)
        """
        rel = self.rocket_pos - self.marker_pos_meas
        vel = self.rocket_vel
        lean = utils.euler_from_axis(self.rocket_quat)
        bz = utils.body_axis(self.rocket_quat, 2)
        avel = self.rocket_avel
        return np.concatenate([rel, vel, lean, [bz[2]], avel, [self.altitude]]).astype(np.float32)

    # -------------------------------------------------------------- bookkeeping
    def _termination(self):
        # scoring always uses ground truth, never the estimator/vision output
        info = {}
        pos = self.pos_true
        vel = self.vel_true
        tilt = utils.tilt_angle(self.quat_true)
        alt = self.altitude_true
        horiz_err = float(np.linalg.norm((pos - self.marker_pos)[:2]))
        info.update(altitude=alt, horiz_err=horiz_err, tilt=tilt,
                    speed=float(np.linalg.norm(vel)), vz=float(vel[2]))

        done = False
        if alt <= 0.05:  # touched down
            done = True
            soft = (np.linalg.norm(vel) < 2.0) and (tilt < np.deg2rad(15)) \
                and (horiz_err < 1.0)
            info["landed"] = True
            info["success"] = bool(soft)
            info["outcome"] = "soft-landing" if soft else "crash"
        elif self.time >= self.cfg.max_seconds:
            done = True
            info["landed"] = False
            info["success"] = False
            info["outcome"] = "timeout"
        elif horiz_err > 50.0 or pos[2] > 80.0:
            done = True
            info["landed"] = False
            info["success"] = False
            info["outcome"] = "diverged"
        return done, info

    def _reward(self, action, info) -> float:
        """Shaped reward (used if training the MLP with RL; BC ignores it)."""
        pos = self.pos_true - self.marker_pos
        vel = self.vel_true
        tilt = utils.tilt_angle(self.quat_true)
        r = 0.0
        r -= 0.02 * np.linalg.norm(pos[:2])      # stay over the pad
        r -= 0.05 * abs(vel[2] + 1.0)            # track ~1 m/s descent
        r -= 0.02 * np.linalg.norm(vel[:2])
        r -= 0.5 * tilt
        if info.get("done") or info.get("landed"):
            if info.get("success"):
                r += 100.0
            elif info.get("outcome") == "crash":
                r -= 50.0
        return float(r)
