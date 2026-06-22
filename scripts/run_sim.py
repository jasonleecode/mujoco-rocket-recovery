#!/usr/bin/env python3
"""Run a landing episode, optionally with the live MuJoCo viewer.

Examples:
    # classical controller only, live viewer
    python scripts/run_sim.py

    # full two-stage controller (classical -> trained MLP), live viewer
    python scripts/run_sim.py --policy models/mlp_policy.pt

    # headless, just print the outcome
    python scripts/run_sim.py --headless --policy models/mlp_policy.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rocket_landing.env import RocketEnv, EnvConfig
from rocket_landing.controllers import ClassicalController, MLPController
from rocket_landing.guidance import TwoStageController


def build_controller(args, env):
    if args.policy and os.path.exists(args.policy):
        mlp = MLPController(checkpoint=args.policy)
        print(f"loaded MLP policy from {args.policy}")
        ctrl = TwoStageController(mlp=mlp, switch_altitude=args.switch_altitude)
    else:
        print("no policy -> classical controller for the whole descent")
        ctrl = ClassicalController()
    if args.vision:
        from rocket_landing.vision import HVisionSensor, VisionController
        print("vision-in-the-loop: aligning to the H from the onboard camera")
        ctrl = VisionController(ctrl, HVisionSensor(env))
    return ctrl


def run_headless(env, controller):
    obs = env.reset()
    if hasattr(controller, "reset"):
        controller.reset()
    done = False
    while not done:
        result = env.step(controller.act(env))
        done = result.done
    info = result.info
    print(f"outcome={info['outcome']} success={info.get('success')} "
          f"horiz_err={info['horiz_err']:.2f}m speed={info['speed']:.2f}m/s "
          f"tilt={np.rad2deg(info['tilt']):.1f}deg")


def run_viewer(env, controller, realtime=True):
    import mujoco
    import mujoco.viewer

    obs = env.reset()
    if hasattr(controller, "reset"):
        controller.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        stage = getattr(controller, "stage", 1)
        while viewer.is_running():
            t0 = time.time()
            action = controller.act(env)
            # show the engine plume scaled with throttle
            flame_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "flame")
            env.model.site_rgba[flame_sid][3] = 0.3 + 0.6 * float(action[0])
            result = env.step(action)
            viewer.sync()
            if result.done:
                info = result.info
                print(f"outcome={info['outcome']} success={info.get('success')} "
                      f"horiz_err={info['horiz_err']:.2f}m speed={info['speed']:.2f}m/s "
                      f"tilt={np.rad2deg(info['tilt']):.1f}deg")
                time.sleep(2.0)
                env.reset()
                if hasattr(controller, "reset"):
                    controller.reset()
            if realtime:
                dt = env.dt - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default=None, help="path to trained MLP checkpoint")
    p.add_argument("--switch-altitude", dest="switch_altitude", type=float, default=12.0)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--vision", action="store_true",
                   help="close the loop on the onboard camera H-detector")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-randomize", dest="randomize", action="store_false")
    args = p.parse_args()

    env = RocketEnv(EnvConfig(seed=args.seed, randomize=args.randomize))
    controller = build_controller(args, env)

    if args.headless:
        run_headless(env, controller)
    else:
        try:
            run_viewer(env, controller)
        except Exception as e:  # noqa: BLE001 - viewer needs a display
            print(f"viewer unavailable ({e}); falling back to headless")
            run_headless(env, controller)


if __name__ == "__main__":
    main()
