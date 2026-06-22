"""Small math helpers (quaternions, rotations) built on numpy / mujoco."""
from __future__ import annotations

import numpy as np
import mujoco


def quat2mat(quat: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
    return mat.reshape(3, 3)


def body_axis(quat: np.ndarray, axis: int = 2) -> np.ndarray:
    """Return the world-frame direction of a body local axis (0=x,1=y,2=z)."""
    return quat2mat(quat)[:, axis]


def quat2euler(quat: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) -> roll, pitch, yaw (rad), XYZ intrinsic-ish.

    Good enough for small-angle attitude bookkeeping / logging.
    """
    w, x, y, z = quat
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


def tilt_angle(quat: np.ndarray) -> float:
    """Angle (rad) between the rocket long axis and world vertical."""
    bz = body_axis(quat, 2)
    return float(np.arccos(np.clip(bz[2], -1.0, 1.0)))


def euler_from_axis(quat: np.ndarray) -> np.ndarray:
    """Return (lean_x, lean_y): components of body +Z tilt in world XY.

    lean_x>0 means the nose tips toward +X, etc. Used by the attitude loop.
    """
    bz = body_axis(quat, 2)
    return np.array([bz[0], bz[1]])
