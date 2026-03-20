"""
minigrid_env.py — Gymnasium environment for a 50 kW solar-battery-diesel mini-grid.

State vector  : [SOC, PV_output_kW, load_demand_kW, diesel_status, time_of_day, battery_health_index]
Action space  : Discrete(5)
  0 — charge_from_diesel   : run diesel, direct excess to battery
  1 — discharge_battery    : serve load from battery; no diesel
  2 — run_diesel_only      : serve load from diesel; battery idle
  3 — solar_only           : serve load from PV; battery absorbs surplus
  4 — shed_load            : deliberately curtail non-critical load
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import yaml


# ---------------------------------------------------------------------------
# Action index constants (keeps agents readable)
# ---------------------------------------------------------------------------
ACTION_CHARGE_FROM_DIESEL  = 0
ACTION_DISCHARGE_BATTERY   = 1
ACTION_RUN_DIESEL_ONLY     = 2
ACTION_SOLAR_ONLY          = 3
ACTION_SHED_LOAD           = 4
N_ACTIONS                  = 5


def _load_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).parent.parent / "config" / "system_params.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


class MiniGridEnv(gym.Env):
    """Gymnasium environment for a solar-battery-diesel mini-grid."""

    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, config_path: str | Path | None = None, render_mode: str | None = None):
        super().__init__()

        self.cfg       = _load_config(config_path)
        self.sim_cfg   = self.cfg["simulation"]
        self.pv_cfg    = self.cfg["pv"]
        self.bat_cfg   = self.cfg["battery"]
        self.die_cfg   = self.cfg["diesel"]
        self.load_cfg  = self.cfg["load"]
        self.rew_cfg   = self.cfg["reward"]

        self.dt          = self.sim_cfg["timestep_hours"]    # hours per step
        self.ep_len      = self.sim_cfg["episode_length"]    # steps per episode
        self.render_mode = render_mode

        # Seeded RNG — reset() can re-seed per episode
        self._np_rng = np.random.default_rng(self.sim_cfg["seed"])

        # --- Action space -----------------------------------------------
        self.action_space = spaces.Discrete(N_ACTIONS)

        # --- Observation space ------------------------------------------
        # [SOC, PV_kW, load_kW, diesel_on, hour_sin, hour_cos, health]
        # We encode time-of-day as (sin, cos) so the boundary 0/23 is smooth.
        # SOC              ∈ [0, 1]
        # PV_kW            ∈ [0, pv.rated_power_kw]
        # load_kW          ∈ [0, load.peak_demand_kw * 1.5]   (slack)
        # diesel_on        ∈ {0, 1}   (we make it continuous for NN compat.)
        # hour_sin         ∈ [-1, 1]
        # hour_cos         ∈ [-1, 1]
        # health_index     ∈ [0, 1]
        low  = np.array([0.0,  0.0,  0.0,  0.0, -1.0, -1.0, 0.0], dtype=np.float32)
        high = np.array([1.0,
                         self.pv_cfg["rated_power_kw"],
                         self.load_cfg["peak_demand_kw"] * 1.5,
                         1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Internal state (initialised properly in reset())
        self._soc          = self.bat_cfg["initial_soc"]
        self._health       = 1.0          # battery health index ∈ [0,1]
        self._diesel_on    = False
        self._step_count   = 0

        # Cached PV/load drawn by _make_obs(); consumed by the following step()
        # so both the observation and the dispatch use the same noise sample.
        self._cached_pv:   float = 0.0
        self._cached_load: float = 0.0

        # Episode log list (list of dicts, converted to DataFrame on demand)
        self._log: list[dict] = []

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None
              ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        self._soc        = self.bat_cfg["initial_soc"]
        self._health     = 1.0
        self._diesel_on  = False
        self._step_count = 0
        self._log        = []

        obs = self._make_obs()
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        hour = self._step_count % 24

        # 1. Consume the PV/load samples that were drawn by _make_obs() when
        #    this timestep's observation was built — no second RNG draw.
        pv_avail = self._cached_pv
        load_dem = self._cached_load

        # 2. Dispatch logic — returns energy flows (kWh over dt)
        flows = self._dispatch(action, pv_avail, load_dem, hour)

        # 3. Update battery SOC & health
        self._update_battery(flows)

        # 4. Compute reward components
        reward, rew_info = self._compute_reward(flows, load_dem)

        # 5. Advance time
        self._step_count += 1
        terminated = self._step_count >= self.ep_len
        truncated  = False

        obs = self._make_obs()

        # 6. Log the step
        self._log.append({
            "step":             self._step_count - 1,
            "hour":             hour,
            "action":           action,
            "soc":              round(flows["soc_before"], 4),
            "soc_after":        round(self._soc, 4),
            "pv_kw":            round(pv_avail, 3),
            "load_kw":          round(load_dem, 3),
            "diesel_on":        int(flows["diesel_on"]),
            "diesel_kw":        round(flows["diesel_kw"], 3),
            "battery_kw":       round(flows.get("battery_kw", 0.0), 3),
            "unmet_load_kw":    round(flows["unmet_load_kw"], 3),
            "curtailed_kw":     round(flows.get("curtailed_kw", 0.0), 3),
            "health":           round(self._health, 5),
            "reward":           round(reward, 4),
            **{f"rew_{k}": round(v, 4) for k, v in rew_info.items()},
        })

        info = {"flows": flows, "reward_components": rew_info}
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode == "human":
            step = self._step_count
            print(f"[t={step:02d}] SOC={self._soc:.2f}  health={self._health:.4f}  "
                  f"diesel={'ON' if self._diesel_on else 'off'}")

    # ------------------------------------------------------------------
    # Stochastic physics
    # ------------------------------------------------------------------
    def _pv_output(self, hour: int) -> float:
        """Gaussian irradiance curve with multiplicative noise."""
        p    = self.pv_cfg
        peak = p["rated_power_kw"] * p["panel_efficiency"] * p["performance_ratio"]
        # Daylight bell centred at peak_hour, std ≈ 3 h so sunrise~7, sunset~19
        det  = peak * math.exp(-0.5 * ((hour - p["irradiance_peak_hour"]) / 3.0) ** 2)
        noise_mult = 1.0 + self._np_rng.normal(0, p["irradiance_sigma_frac"])
        return float(np.clip(det * noise_mult, 0.0, peak))

    def _load_demand(self, hour: int) -> float:
        """Base load + evening Gaussian peak + multiplicative noise."""
        l    = self.load_cfg
        base = l["base_demand_kw"]
        peak_add = (l["peak_demand_kw"] - base) * math.exp(
            -0.5 * ((hour - l["peak_hour"]) / l["peak_width_hours"]) ** 2
        )
        det  = base + peak_add
        noise_mult = 1.0 + self._np_rng.normal(0, l["demand_sigma_frac"])
        return float(np.clip(det * noise_mult, base * 0.5, l["peak_demand_kw"] * 1.3))

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def _dispatch(self, action: int, pv_kw: float, load_kw: float, hour: int) -> dict:
        """
        Translate a discrete action into physical energy flows.

        All power values are in kW; energy = power * dt (kWh).
        Returns a dict of flow magnitudes and flags for reward / logging.
        """
        bat  = self.bat_cfg
        die  = self.die_cfg

        soc_before    = self._soc
        capacity_kwh  = bat["capacity_kwh"] * self._health  # effective capacity

        # Available battery energy above min_soc
        bat_avail_kwh = (self._soc - bat["min_soc"]) * capacity_kwh
        bat_avail_kw  = min(bat_avail_kwh / self.dt, bat["max_discharge_rate_kw"])

        # Headroom in battery below max_soc
        bat_room_kwh  = (bat["max_soc"] - self._soc) * capacity_kwh
        bat_room_kw   = min(bat_room_kwh / self.dt, bat["max_charge_rate_kw"])

        diesel_kw      = 0.0
        battery_kw     = 0.0   # positive = discharge (supply load), negative = charge
        unmet_load_kw  = 0.0
        curtailed_kw   = 0.0
        diesel_on      = False

        if action == ACTION_CHARGE_FROM_DIESEL:
            # Diesel runs at rated power; load served first, surplus charges battery
            diesel_on  = True
            diesel_kw  = min(die["rated_power_kw"],
                             die["rated_power_kw"] * die["min_load_fraction"] + load_kw)
            supply_kw  = pv_kw + diesel_kw
            surplus_kw = supply_kw - load_kw
            charge_kw  = min(surplus_kw, bat_room_kw)
            battery_kw = -charge_kw  # negative = charging
            unmet_load_kw = max(0.0, load_kw - supply_kw + charge_kw)

        elif action == ACTION_DISCHARGE_BATTERY:
            # PV + battery serve load; no diesel
            diesel_on  = False
            supply_kw  = pv_kw + bat_avail_kw
            if supply_kw >= load_kw:
                battery_kw = load_kw - pv_kw  # only what's needed
                battery_kw = max(0.0, min(battery_kw, bat_avail_kw))
            else:
                battery_kw    = bat_avail_kw
                unmet_load_kw = load_kw - pv_kw - bat_avail_kw

        elif action == ACTION_RUN_DIESEL_ONLY:
            # Diesel alone serves load; PV curtailed if battery full
            diesel_on  = True
            diesel_kw  = min(max(load_kw, die["rated_power_kw"] * die["min_load_fraction"]),
                             die["rated_power_kw"])
            supply_kw  = pv_kw + diesel_kw
            surplus_kw = supply_kw - load_kw
            # try to store surplus in battery
            charge_kw  = min(surplus_kw, bat_room_kw)
            battery_kw = -charge_kw
            curtailed_kw = max(0.0, surplus_kw - charge_kw)

        elif action == ACTION_SOLAR_ONLY:
            # Only PV; battery absorbs surplus or supplements deficit
            diesel_on   = False
            if pv_kw >= load_kw:
                surplus_kw   = pv_kw - load_kw
                charge_kw    = min(surplus_kw, bat_room_kw)
                battery_kw   = -charge_kw
                curtailed_kw = surplus_kw - charge_kw
            else:
                deficit_kw   = load_kw - pv_kw
                discharge_kw = min(deficit_kw, bat_avail_kw)
                battery_kw   = discharge_kw
                unmet_load_kw = deficit_kw - discharge_kw

        elif action == ACTION_SHED_LOAD:
            # Deliberately curtail load to minimum (base load only)
            diesel_on     = False
            effective_load = self.load_cfg["base_demand_kw"]
            supply_kw      = pv_kw + bat_avail_kw
            if supply_kw >= effective_load:
                battery_kw = max(0.0, effective_load - pv_kw)
            else:
                battery_kw    = bat_avail_kw
                unmet_load_kw = effective_load - pv_kw - bat_avail_kw
            # shedding the rest of the load is the "action cost"
            curtailed_kw  = max(0.0, load_kw - effective_load)

        self._diesel_on = diesel_on

        return {
            "action":         action,
            "soc_before":     soc_before,
            "pv_kw":          pv_kw,
            "load_kw":        load_kw,
            "diesel_kw":      diesel_kw,
            "diesel_on":      diesel_on,
            "battery_kw":     battery_kw,   # + = discharge, - = charge
            "unmet_load_kw":  max(0.0, unmet_load_kw),
            "curtailed_kw":   curtailed_kw,
        }

    # ------------------------------------------------------------------
    # Battery state update
    # ------------------------------------------------------------------
    def _update_battery(self, flows: dict) -> None:
        bat          = self.bat_cfg
        capacity_kwh = bat["capacity_kwh"] * self._health

        battery_kw = flows["battery_kw"]   # + discharge, – charge
        eta        = bat["round_trip_efficiency"]

        if battery_kw >= 0:  # discharge
            delta_soc = -(battery_kw * self.dt) / capacity_kwh
        else:                 # charge — efficiency applied on charge side
            delta_soc = (-battery_kw * self.dt * eta) / capacity_kwh

        new_soc = float(np.clip(self._soc + delta_soc,
                                bat["min_soc"], bat["max_soc"]))

        # Degradation — proportional to DoD of this step  [Wöhrle et al. 2021]
        dod_this_step = abs(new_soc - self._soc)
        self._health  = max(0.0, self._health
                            - bat["degradation_per_cycle"] * dod_this_step * 2)

        self._soc = new_soc

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(self, flows: dict, load_kw: float) -> tuple[float, dict]:
        r    = self.rew_cfg
        die  = self.die_cfg
        bat  = self.bat_cfg

        # 1. Unmet load penalty (VoLL) — also applied to deliberately curtailed
        #    demand (shed_load action) so that shedding is always strictly
        #    costlier than dispatching diesel.  PV curtailment from other
        #    actions (surplus solar that cannot be stored) is not penalised.
        unmet_kwh            = flows["unmet_load_kw"] * self.dt
        curtailed_demand_kwh = (flows["curtailed_kw"] * self.dt
                                if flows["action"] == ACTION_SHED_LOAD else 0.0)
        r_unmet              = -r["unmet_load_cost_per_kwh"] * (unmet_kwh + curtailed_demand_kwh)

        # 2. Fuel cost
        diesel_kwh      = flows["diesel_kw"] * self.dt
        fuel_litres     = diesel_kwh * die["fuel_consumption_l_per_kwh"]
        fuel_cost       = fuel_litres * die["fuel_price_usd_per_litre"]
        startup_cost    = die["startup_cost_usd"] if flows["diesel_on"] else 0.0
        fixed_cost      = die["hourly_fixed_cost_usd"] * self.dt if flows["diesel_on"] else 0.0
        r_fuel          = -(fuel_cost + startup_cost + fixed_cost) * r["fuel_cost_weight"]

        # 3. Battery stress — penalise DoD beyond threshold
        dod             = 1.0 - self._soc
        excess_dod      = max(0.0, dod - bat["stress_threshold_dod"])
        r_bat_stress    = -r["battery_stress_cost_per_dod"] * excess_dod

        # 4. Solar-hour diesel penalty — fire whenever PV exceeds the negligible
        #    threshold (10 % of rated capacity), not only when there is a net
        #    PV surplus.  Shadow-prices the available solar energy at the same
        #    fuel rate, doubling the effective cost of co-running diesel during
        #    daylight and discouraging charge-from-diesel while PV is available.
        pv_threshold = 0.10 * self.pv_cfg["rated_power_kw"]
        if flows["diesel_on"] and flows["pv_kw"] > pv_threshold:
            shadow_kwh     = flows["pv_kw"] * self.dt
            r_diesel_solar = -(shadow_kwh
                               * die["fuel_consumption_l_per_kwh"]
                               * die["fuel_price_usd_per_litre"]
                               * r["fuel_cost_weight"])
        else:
            r_diesel_solar = 0.0

        total = r_unmet + r_fuel + r_bat_stress + r_diesel_solar

        rew_components = {
            "unmet_load":     r_unmet,
            "fuel":           r_fuel,
            "battery_stress": r_bat_stress,
            "diesel_solar":   r_diesel_solar,
        }
        return total, rew_components

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------
    def _make_obs(self) -> np.ndarray:
        hour = self._step_count % 24

        # Sample once and cache so that step() uses the identical values for
        # dispatch — eliminates the double-sampling bug where the observation
        # and the actual dispatch drew independent noise realisations.
        self._cached_pv   = self._pv_output(hour)
        self._cached_load = self._load_demand(hour)

        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)

        return np.array([
            self._soc,
            self._cached_pv,
            self._cached_load,
            float(self._diesel_on),
            hour_sin,
            hour_cos,
            self._health,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def log(self) -> list[dict]:
        """Return the per-step log list."""
        return self._log

    def log_as_dataframe(self):
        """Return episode log as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self._log)
