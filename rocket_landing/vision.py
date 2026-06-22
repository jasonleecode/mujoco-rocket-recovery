"""Camera-based H-mark detection for closed-loop visual landing.

A downward-facing camera on the rocket renders the pad. A lightweight classical
pipeline segments the bright "H" against the dark pad, takes its centroid, and
back-projects that pixel through the (known) camera pose onto the ground plane
to estimate the H-mark world position. That estimate replaces the previously
"privileged" ground-truth target, so guidance now aligns to the H purely from
what the camera sees -- exactly what an onboard landing system does (known ego
pose from the IMU + a vision detection of the target).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import mujoco

from rocket_landing.env import PAD_TOP_Z


@dataclass
class Detection:
    found: bool
    marker_world: Optional[np.ndarray]   # estimated H position in world (x, y, z)
    pixel: Optional[np.ndarray]          # blob centroid (col, row) in the image
    confidence: float                    # 0..1, from the bright-pixel area
    yaw: float                           # estimated H principal-axis angle (rad)
    image: Optional[np.ndarray] = None   # rendered RGB (uint8) if requested
    mask: Optional[np.ndarray] = None    # white-pixel mask (bool) if requested


class HVisionSensor:
    """Renders the rocket's down-camera and estimates the H-mark position."""

    def __init__(self, env, width: int = 200, height: int = 200,
                 bright_thresh: int = 200, min_pixels: int = 12,
                 camera: str = "downcam"):
        self.env = env
        self.width = width
        self.height = height
        self.bright_thresh = bright_thresh
        self.min_pixels = min_pixels
        self.renderer = mujoco.Renderer(env.model, height, width)
        self.cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        self.fovy = float(env.model.cam_fovy[self.cam_id])
        # precompute per-pixel ray tangents in the camera frame
        tan_v = np.tan(np.deg2rad(self.fovy) / 2.0)
        aspect = width / height
        tan_h = tan_v * aspect
        cols = (np.arange(width) - width / 2.0 + 0.5) / (width / 2.0)
        rows = (np.arange(height) - height / 2.0 + 0.5) / (height / 2.0)
        self._cols_t = cols * tan_h            # +x to the right
        self._rows_t = rows * tan_v            # image row -> -y (up is +y)

    def reset(self):
        pass

    # ------------------------------------------------------------------ render
    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.env.data, camera=self.cam_id)
        return self.renderer.render()

    # --------------------------------------------------------------- detection
    def detect(self, return_image: bool = False) -> Detection:
        img = self.render()
        gray = img.mean(axis=2)
        spread = img.max(axis=2).astype(np.int16) - img.min(axis=2).astype(np.int16)
        mask = (gray > self.bright_thresh) & (spread < 40)  # bright + low saturation
        ys, xs = np.nonzero(mask)
        n = len(xs)
        conf = min(1.0, n / 800.0)

        if n < self.min_pixels:
            det = Detection(False, None, None, conf, 0.0)
        else:
            cx, cy = xs.mean(), ys.mean()
            # principal-axis orientation of the blob (for reporting / alignment)
            dx, dy = xs - cx, ys - cy
            cov = np.array([[ (dx * dx).mean(), (dx * dy).mean()],
                            [ (dx * dy).mean(), (dy * dy).mean()]])
            w, v = np.linalg.eigh(cov)
            major = v[:, np.argmax(w)]
            yaw = float(np.arctan2(major[1], major[0]))
            marker = self._backproject(cx, cy)
            det = Detection(marker is not None, marker, np.array([cx, cy]),
                            conf, yaw)
        if return_image:
            det.image, det.mask = img, mask
        return det

    # --------------------------------------------------- pixel -> world ground
    def _backproject(self, col: float, row: float) -> Optional[np.ndarray]:
        """Intersect the camera ray through (col,row) with the pad-top plane."""
        ci = int(round(np.clip(col, 0, self.width - 1)))
        ri = int(round(np.clip(row, 0, self.height - 1)))
        # ray direction in camera frame (camera looks along -z, +y up)
        d_cam = np.array([self._cols_t[ci], -self._rows_t[ri], -1.0])
        cam_pos = np.array(self.env.data.cam_xpos[self.cam_id])
        cam_mat = np.array(self.env.data.cam_xmat[self.cam_id]).reshape(3, 3)
        d_world = cam_mat @ d_cam
        if d_world[2] >= -1e-6:                       # ray not pointing down
            return None
        t = (PAD_TOP_Z - cam_pos[2]) / d_world[2]
        if t <= 0:
            return None
        p = cam_pos + t * d_world
        p[2] = PAD_TOP_Z
        return p


class VisionController:
    """Wrap any ``act(env)`` controller so guidance uses the vision estimate.

    Each step it runs the detector; on a confident detection it writes the
    estimated H position into ``env.marker_estimate`` (consumed by the env's
    observation and by the controllers). On a miss it holds the last estimate.
    """

    def __init__(self, base, sensor: HVisionSensor,
                 min_confidence: float = 0.02, freeze_below: float = 7.0):
        self.base = base
        self.sensor = sensor
        self.min_confidence = min_confidence
        # below this altitude the engine plume, nozzle and legs occlude the H,
        # biasing the centroid -- so lock the last clean estimate instead.
        self.freeze_below = freeze_below
        self.last_marker: Optional[np.ndarray] = None
        self.n_detections = 0
        self.n_steps = 0

    @property
    def stage(self):
        return getattr(self.base, "stage", 1)

    def reset(self):
        self.last_marker = None
        self.n_detections = 0
        self.n_steps = 0
        self.sensor.env.marker_estimate = None
        self.sensor.reset()
        if hasattr(self.base, "reset"):
            self.base.reset()

    def act(self, env) -> np.ndarray:
        self.n_steps += 1
        if env.altitude > self.freeze_below:
            det = self.sensor.detect()
            if det.found and det.confidence >= self.min_confidence:
                self.last_marker = det.marker_world
                self.n_detections += 1
        # use the latest good estimate; before any detection, fall back to truth
        env.marker_estimate = self.last_marker
        return self.base.act(env)
