"""Vector-thrust rocket recovery simulation in MuJoCo.

Two-stage descent control:
  * Stage 1 (approach / deceleration): a classical cascaded PD guidance &
    attitude controller (``rocket_landing.controllers.classical``).
  * Stage 2 (terminal landing): a learned MLP policy that fine-tunes the
    touchdown onto the "H" mark (``rocket_landing.controllers.mlp``).
"""

from rocket_landing.env import RocketEnv, EnvConfig

__all__ = ["RocketEnv", "EnvConfig"]
__version__ = "0.1.0"
