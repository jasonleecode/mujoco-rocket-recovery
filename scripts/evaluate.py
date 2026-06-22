#!/usr/bin/env python3
"""Benchmark a controller over many randomized descents.

    python scripts/evaluate.py --controller classical
    python scripts/evaluate.py --controller two-stage --policy models/mlp_policy.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rocket_landing.env import RocketEnv, EnvConfig
from rocket_landing.controllers import ClassicalController, MLPController
from rocket_landing.guidance import TwoStageController
from rocket_landing.rollout import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller", choices=["classical", "two-stage", "mlp"],
                   default="classical")
    p.add_argument("--policy", default="models/mlp_policy.pt")
    p.add_argument("--switch-altitude", dest="switch_altitude", type=float, default=12.0)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--vision", action="store_true",
                   help="close the loop on the onboard camera H-detector")
    p.add_argument("--estimator", action="store_true",
                   help="run guidance on fused IMU+GPS state instead of truth")
    args = p.parse_args()

    env = RocketEnv(EnvConfig(randomize=True))

    if args.controller == "classical":
        ctrl = ClassicalController()
    else:
        mlp = MLPController(checkpoint=args.policy)
        if args.controller == "mlp":
            ctrl = mlp  # MLP for the whole descent (stress test)
        else:
            ctrl = TwoStageController(mlp=mlp, switch_altitude=args.switch_altitude)

    if args.vision:
        from rocket_landing.vision import HVisionSensor, VisionController
        ctrl = VisionController(ctrl, HVisionSensor(env))
    if args.estimator:
        from rocket_landing.estimator import StateEstimator, EstimationController
        ctrl = EstimationController(ctrl, StateEstimator(env))

    metrics = evaluate(env, ctrl, n=args.episodes, base_seed=args.seed)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
