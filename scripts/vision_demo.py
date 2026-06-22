#!/usr/bin/env python3
"""Render a vision-in-the-loop landing: external view + onboard camera with the
H detection overlaid.

    python scripts/vision_demo.py --out vision_landing.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import mujoco
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rocket_landing.env import RocketEnv, EnvConfig
from rocket_landing.controllers import ClassicalController, MLPController
from rocket_landing.guidance import TwoStageController
from rocket_landing.vision import HVisionSensor, VisionController


def draw_cross(img, col, row, color, size=7):
    h, w = img.shape[:2]
    c, r = int(col), int(row)
    for d in range(-size, size + 1):
        if 0 <= r < h and 0 <= c + d < w:
            img[r, c + d] = color
        if 0 <= r + d < h and 0 <= c < w:
            img[r + d, c] = color


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="models/mlp_policy.pt")
    p.add_argument("--out", default="vision_landing.mp4")
    p.add_argument("--seed", type=int, default=2001)
    p.add_argument("--size", type=int, default=480, help="panel height (px)")
    p.add_argument("--fps", type=int, default=50)
    args = p.parse_args()

    env = RocketEnv(EnvConfig(seed=args.seed, randomize=True))
    if os.path.exists(args.policy):
        base = TwoStageController(mlp=MLPController(checkpoint=args.policy))
    else:
        base = ClassicalController()
    sensor = HVisionSensor(env, width=256, height=256)
    controller = VisionController(base, sensor)
    env.reset()
    controller.reset()

    ext = mujoco.Renderer(env.model, args.size, args.size)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = 130, -12
    flame_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "flame")

    frames = []
    done = False
    while not done:
        # detection (with image) for the overlay panel
        det = sensor.detect(return_image=True)
        if env.altitude > controller.freeze_below and det.found \
                and det.confidence >= controller.min_confidence:
            controller.last_marker = det.marker_world
            controller.n_detections += 1
        env.marker_estimate = controller.last_marker
        action = base.act(env)
        controller.n_steps += 1

        env.model.site_rgba[flame_sid][3] = 0.3 + 0.6 * float(action[0])

        # onboard panel: highlight detected H pixels + centroid cross
        panel = det.image.copy()
        if det.mask is not None:
            panel[det.mask] = [80, 220, 120]
        if det.found:
            draw_cross(panel, det.pixel[0], det.pixel[1], [255, 60, 60], 9)
        panel = np.repeat(np.repeat(panel, args.size // 256 + 1, 0),
                          args.size // 256 + 1, 1)[:args.size, :args.size]

        # external panel
        pos = env.rocket_pos
        cam.lookat[:] = [pos[0], pos[1], max(pos[2] - 3.0, 3.0)]
        cam.distance = 18.0 + 0.7 * max(env.altitude, 0.0)
        ext.update_scene(env.data, cam)
        ext_img = ext.render()

        sep = np.full((args.size, 4, 3), 30, np.uint8)
        frames.append(np.hstack([ext_img, sep, panel]))

        res = env.step(action)
        done = res.done

    info = res.info
    print(f"outcome={info['outcome']} success={info.get('success')} "
          f"horiz_err={info['horiz_err']:.2f}m speed={info['speed']:.2f}m/s "
          f"detections={controller.n_detections}/{controller.n_steps}")
    frames += [frames[-1]] * int(1.5 * args.fps)
    imageio.mimsave(args.out, frames, fps=args.fps, codec="libx264",
                    quality=8, macro_block_size=None)
    print(f"saved {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
