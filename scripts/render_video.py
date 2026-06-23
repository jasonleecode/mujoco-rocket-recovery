#!/usr/bin/env python3
"""Render a landing episode to an MP4/GIF with a rocket-tracking camera.

    python scripts/render_video.py --policy models/mlp_policy.pt --out landing.mp4
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


def build_controller(args):
    if args.policy and os.path.exists(args.policy):
        mlp = MLPController(checkpoint=args.policy)
        return TwoStageController(mlp=mlp, switch_altitude=args.switch_altitude)
    return ClassicalController()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="models/mlp_policy.pt")
    p.add_argument("--switch-altitude", dest="switch_altitude", type=float, default=12.0)
    p.add_argument("--out", default="landing.mp4")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--hold", type=float, default=1.5, help="freeze seconds after touchdown")
    args = p.parse_args()

    env = RocketEnv(EnvConfig(seed=args.seed, randomize=True))
    controller = build_controller(args)
    env.reset()
    if hasattr(controller, "reset"):
        controller.reset()

    renderer = mujoco.Renderer(env.model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.azimuth = 130
    cam.elevation = -12

    frames = []
    done = False
    while not done:
        action = controller.act(env)
        env.set_flame(action[0])
        result = env.step(action)
        done = result.done
        # track the rocket, easing the camera distance with altitude
        pos = env.rocket_pos
        cam.lookat[:] = [pos[0], pos[1], max(pos[2] - 3.0, 3.0)]
        cam.distance = 18.0 + 0.7 * max(env.altitude, 0.0)
        renderer.update_scene(env.data, cam)
        frames.append(renderer.render())

    info = result.info
    print(f"outcome={info['outcome']} success={info.get('success')} "
          f"horiz_err={info['horiz_err']:.2f}m speed={info['speed']:.2f}m/s "
          f"tilt={np.rad2deg(info['tilt']):.1f}deg")

    # hold the final frame
    frames += [frames[-1]] * int(args.hold * args.fps)

    if args.out.endswith(".gif"):
        imageio.mimsave(args.out, frames, fps=args.fps)
    else:
        imageio.mimsave(args.out, frames, fps=args.fps, codec="libx264",
                        quality=8, macro_block_size=None)
    print(f"saved {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
