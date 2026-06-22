"""Two-stage guidance: classical approach hand-off to the learned terminal MLP."""
from __future__ import annotations

from typing import Optional

import numpy as np

from rocket_landing.controllers import ClassicalController, MLPController


class TwoStageController:
    """Switch from the classical controller to the MLP near the pad.

    Stage 1 (altitude > ``switch_altitude``): classical guidance flies the
    rocket over the H mark and brakes the descent.
    Stage 2 (altitude <= ``switch_altitude``): the MLP policy takes over for the
    precise terminal landing. A small hysteresis avoids chattering at the
    boundary.
    """

    def __init__(self, classical: Optional[ClassicalController] = None,
                 mlp: Optional[MLPController] = None,
                 switch_altitude: float = 12.0, hysteresis: float = 1.0):
        self.classical = classical or ClassicalController()
        self.mlp = mlp
        self.switch_altitude = switch_altitude
        self.hysteresis = hysteresis
        self.stage = 1
        self.history = []

    def reset(self):
        self.stage = 1
        self.history = []
        self.classical.reset()
        if self.mlp is not None:
            self.mlp.reset()

    def act(self, env) -> np.ndarray:
        alt = env.altitude
        # stage transition with hysteresis
        if self.stage == 1 and alt <= self.switch_altitude and self.mlp is not None:
            self.stage = 2
        elif self.stage == 2 and alt > self.switch_altitude + self.hysteresis:
            self.stage = 1

        if self.stage == 2 and self.mlp is not None:
            action = self.mlp.act(env)
        else:
            action = self.classical.act(env)

        self.history.append((self.stage, alt))
        return np.asarray(action, dtype=np.float32)
