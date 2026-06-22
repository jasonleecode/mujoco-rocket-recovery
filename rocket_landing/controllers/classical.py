"""Stage-1 classical guidance & attitude controller.

Cascaded design:

  1. Guidance (outer loop): a PD law on horizontal position/velocity (toward the
     H mark) and a descent-rate profile gives a desired specific force ``f_des``
     in world frame (includes gravity compensation).
  2. Attitude (inner loop): the rocket should point its long axis along
     ``f_des``. The attitude error drives a desired body torque.
  3. Allocation: the gimbaled nozzle (lever arm below the CoM) turns the desired
     torque into nozzle deflection angles; throttle comes from ``|f_des|``.

The same controller is reused as the "expert" that generates demonstrations for
behavior-cloning the stage-2 MLP policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rocket_landing import utils


@dataclass
class ClassicalGains:
    # horizontal position/velocity guidance
    kp_pos: float = 0.45
    kd_pos: float = 1.1
    a_xy_max: float = 3.2          # max horizontal accel command (m/s^2)
    max_tilt_deg: float = 25.0     # cap the commanded thrust-vector tilt
    # vertical descent-rate profile  v_target = -(v0 + k*altitude)
    v_descent0: float = 0.5
    v_descent_k: float = 0.18
    v_descent_max: float = 6.0
    kp_vz: float = 0.9
    # attitude inner loop (angular-acceleration gains, scaled by inertia)
    kp_att: float = 7.0
    kd_att: float = 4.5
    # engine
    min_throttle: float = 0.03


class ClassicalController:
    """Model-based controller. ``act(env)`` returns a normalized action."""

    LEVER = 5.0  # nozzle distance below the CoM (m), from models/rocket.xml

    def __init__(self, gains: ClassicalGains | None = None):
        self.g = gains or ClassicalGains()

    def reset(self):
        pass

    def act(self, env) -> np.ndarray:
        g = self.g
        grav = env.gravity
        mass = env.mass

        quat = env.rocket_quat
        R = utils.quat2mat(quat)
        bz = R[:, 2]                       # body long axis in world

        rel = (env.rocket_pos - env.marker_pos)
        vel = env.rocket_vel
        avel_world = env.rocket_avel
        alt = env.altitude

        # ---- 1. guidance: desired specific force in world frame ----
        # horizontal: drive position & velocity to zero over the mark
        a_xy = -g.kp_pos * rel[:2] - g.kd_pos * vel[:2]
        n = np.linalg.norm(a_xy)
        if n > g.a_xy_max:
            a_xy *= g.a_xy_max / n

        # vertical: track a descent-rate profile that slows near the ground
        vz_target = -np.clip(g.v_descent0 + g.v_descent_k * max(alt, 0.0),
                             g.v_descent0, g.v_descent_max)
        a_z = g.kp_vz * (vz_target - vel[2])

        f_des = np.array([a_xy[0], a_xy[1], a_z + grav])  # specific force (m/s^2)

        # cap the tilt of the desired thrust direction
        max_tilt = np.deg2rad(g.max_tilt_deg)
        horiz = np.hypot(f_des[0], f_des[1])
        if f_des[2] > 1e-3:
            tilt = np.arctan2(horiz, f_des[2])
            if tilt > max_tilt and horiz > 1e-6:
                scale = (f_des[2] * np.tan(max_tilt)) / horiz
                f_des[0] *= scale
                f_des[1] *= scale

        f_mag = float(np.linalg.norm(f_des))
        thrust = mass * f_mag
        throttle = np.clip(thrust / env.max_thrust, g.min_throttle, 1.0)
        thrust = throttle * env.max_thrust

        # ---- 2. attitude: align body +Z with f_des ----
        zd = f_des / max(f_mag, 1e-6)
        e_world = np.cross(bz, zd)            # axis*sin(error angle), world frame
        e_body = R.T @ e_world
        omega_body = R.T @ avel_world

        inertia = 9000.0  # Ixx ~= Iyy of the rocket (models/rocket.xml)

        tau_x = inertia * (g.kp_att * e_body[0] - g.kd_att * omega_body[0])
        tau_y = inertia * (g.kp_att * e_body[1] - g.kd_att * omega_body[1])

        # ---- 3. allocation: torque -> gimbal deflection ----
        # torque_x ~ -L*T*delta_x  ;  torque_y ~ -L*T*delta_y
        denom = self.LEVER * max(thrust, 1e-3)
        delta_x = -tau_x / denom
        delta_y = -tau_y / denom

        lim = env._gimbal_limit
        gx = np.clip(delta_x, -lim, lim) / lim
        gy = np.clip(delta_y, -lim, lim) / lim

        return np.array([throttle, gx, gy], dtype=np.float32)
