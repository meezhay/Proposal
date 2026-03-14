"""
rule_based_agent.py — Threshold-based dispatch policy (deterministic baseline).

Policy logic (priority order):
────────────────────────────────────────────────────────────────────
Condition                                 → Action
────────────────────────────────────────────────────────────────────
SOC < CRIT_SOC (0.15)                    → shed_load               (4)
SOC < LOW_SOC  (0.30)                    → charge_from_diesel       (0)
PV > load AND SOC < HIGH_SOC (0.85)     → solar_only (stores PV)  (3)
PV > load AND SOC >= HIGH_SOC           → solar_only (curtail OK)  (3)
PV <= load AND SOC >= MED_SOC  (0.50)   → discharge_battery        (1)
PV <= load AND SOC < MED_SOC            → run_diesel_only          (2)
────────────────────────────────────────────────────────────────────

Observation indices (matches MiniGridEnv._make_obs):
  0 — SOC
  1 — PV_kW  (deterministic part, same RNG state peek)
  2 — load_kW
  3 — diesel_on  (0/1)
  4 — hour_sin
  5 — hour_cos
  6 — health_index
"""

from __future__ import annotations
import numpy as np
from .base_agent import BaseAgent
from minigrid_mdp.env.minigrid_env import (
    ACTION_CHARGE_FROM_DIESEL,
    ACTION_DISCHARGE_BATTERY,
    ACTION_RUN_DIESEL_ONLY,
    ACTION_SOLAR_ONLY,
    ACTION_SHED_LOAD,
)


class RuleBasedAgent(BaseAgent):
    """
    Hand-crafted threshold policy.  Serves as the rule-book baseline against
    which learned policies are benchmarked.

    Thresholds are tuned for a 100 kWh LFP battery with 80 % DoD limit
    serving a 35 kW peak Nigerian rural load.  They reproduce common HOMER-
    style 'low-SOC diesel start' setpoints (HOMER Energy, 2021).
    """

    agent_type = "rule_based"

    # SOC thresholds
    CRIT_SOC  = 0.15   # emergency load-shed
    LOW_SOC   = 0.30   # start diesel to charge battery
    MED_SOC   = 0.50   # prefer battery discharge over diesel
    HIGH_SOC  = 0.85   # battery effectively full; stop charging

    # PV observation index
    IDX_SOC    = 0
    IDX_PV     = 1
    IDX_LOAD   = 2

    def act(self, obs: np.ndarray) -> int:
        soc  = float(obs[self.IDX_SOC])
        pv   = float(obs[self.IDX_PV])
        load = float(obs[self.IDX_LOAD])

        # ── Critical: battery almost empty ──────────────────────────────
        if soc < self.CRIT_SOC:
            return ACTION_SHED_LOAD

        # ── Battery low: must charge via diesel ─────────────────────────
        if soc < self.LOW_SOC:
            return ACTION_CHARGE_FROM_DIESEL

        # ── PV surplus available ─────────────────────────────────────────
        if pv >= load:
            # Solar alone can serve load; store the surplus
            return ACTION_SOLAR_ONLY

        # ── PV deficit, decide battery vs diesel ────────────────────────
        if soc >= self.MED_SOC:
            return ACTION_DISCHARGE_BATTERY
        else:
            return ACTION_RUN_DIESEL_ONLY
