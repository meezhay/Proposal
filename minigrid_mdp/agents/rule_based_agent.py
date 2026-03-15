"""
rule_based_agent.py — Threshold-based dispatch policy (deterministic baseline).

Policy logic (priority order):
────────────────────────────────────────────────────────────────────────
Condition                                           → Action
────────────────────────────────────────────────────────────────────────
SOC < CRIT_SOC (0.15)                              → shed_load          (4)
PV >= pv_negligible_kw (10% of rated)              → solar_only         (3)
  [PV active: always use solar, charge if surplus]
SOC < LOW_SOC (0.30) AND PV negligible             → charge_from_diesel (0)
  [dark + low battery: only now allow diesel]
SOC >= MED_SOC (0.50) AND PV negligible            → discharge_battery  (1)
SOC in [LOW_SOC, MED_SOC) AND PV negligible        → run_diesel_only    (2)
────────────────────────────────────────────────────────────────────────

Diesel activation requires BOTH conditions simultaneously:
  (1) PV output < 10% of rated PV capacity (negligible solar)
  (2) SOC < LOW_SOC (0.30)

pv_negligible_kw is derived from pv.rated_power_kw in system_params.yaml
at __init__ time — never hardcoded.

Observation indices (matches MiniGridEnv._make_obs):
  0 — SOC
  1 — PV_kW
  2 — load_kW
  3 — diesel_on  (0/1)
  4 — hour_sin
  5 — hour_cos
  6 — health_index
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import yaml

from .base_agent import BaseAgent
from minigrid_mdp.env.minigrid_env import (
    ACTION_CHARGE_FROM_DIESEL,
    ACTION_DISCHARGE_BATTERY,
    ACTION_RUN_DIESEL_ONLY,
    ACTION_SOLAR_ONLY,
    ACTION_SHED_LOAD,
)


def _load_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).parent.parent / "config" / "system_params.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


class RuleBasedAgent(BaseAgent):
    """
    Hand-crafted threshold policy.  Serves as the rule-book baseline against
    which learned policies are benchmarked.

    Diesel activation requires BOTH conditions simultaneously:
      (1) PV output < 10% of rated PV capacity (negligible solar)
      (2) SOC < LOW_SOC (0.30)
    When PV is actively generating above that threshold, the agent always
    prioritises solar, regardless of SOC level.

    The PV negligibility threshold is derived from pv.rated_power_kw in
    system_params.yaml — never hardcoded.

    Thresholds are tuned for a 100 kWh LFP battery with 80% DoD limit
    serving a 35 kW peak Nigerian rural load.  They reproduce common HOMER-
    style 'low-SOC diesel start' setpoints (HOMER Energy, 2021).
    """

    agent_type = "rule_based"

    # SOC thresholds
    CRIT_SOC = 0.15   # emergency load-shed
    LOW_SOC  = 0.30   # diesel start floor (only when PV is negligible)
    MED_SOC  = 0.50   # prefer battery discharge over diesel
    HIGH_SOC = 0.85   # battery effectively full

    # Fraction of rated PV below which output is considered negligible
    PV_NEGLIGIBLE_FRAC = 0.10

    IDX_SOC  = 0
    IDX_PV   = 1
    IDX_LOAD = 2

    def __init__(self, action_space, config_path: str | Path | None = None):
        super().__init__(action_space)
        cfg = _load_config(config_path)
        # e.g. 50.0 kW * 0.10 = 5.0 kW — rescales automatically with config
        self.pv_negligible_kw: float = cfg["pv"]["rated_power_kw"] * self.PV_NEGLIGIBLE_FRAC

    def act(self, obs: np.ndarray) -> int:
        soc  = float(obs[self.IDX_SOC])
        pv   = float(obs[self.IDX_PV])

        pv_is_negligible = pv < self.pv_negligible_kw

        # ── Critical: battery almost empty ──────────────────────────────
        if soc < self.CRIT_SOC:
            return ACTION_SHED_LOAD

        # ── PV is actively generating: always use solar ──────────────────
        # solar_only dispatch will absorb any surplus into the battery
        if not pv_is_negligible:
            return ACTION_SOLAR_ONLY

        # ── Below here: PV is negligible (dark / heavy overcast) ────────

        # ── Low SOC + no PV: start diesel to charge battery ─────────────
        if soc < self.LOW_SOC:
            return ACTION_CHARGE_FROM_DIESEL

        # ── Enough stored energy: discharge battery ──────────────────────
        if soc >= self.MED_SOC:
            return ACTION_DISCHARGE_BATTERY

        # ── Mid SOC, no PV: run diesel to serve load ─────────────────────
        return ACTION_RUN_DIESEL_ONLY
