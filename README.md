# Mini-Grid Dispatch RL

A Gymnasium simulation environment and reinforcement-learning benchmark for
autonomous energy dispatch in a **50 kW solar-battery-diesel rural mini-grid**,
modelled on Nigerian off-grid communities.

The project explores whether a learned policy (PPO) can outperform a hand-crafted
rule-based policy by minimising fuel cost, battery stress, and unserved load.

---

## System overview

| Component | Specification |
|-----------|--------------|
| PV array | 50 kW rated, η = 18 %, PR = 0.80, solar noon at 13:00 |
| Battery | 100 kWh LFP, 80 % DoD limit, round-trip η = 92 % |
| Diesel generator | 30 kW rated, min load 30 %, fuel 0.60 USD/L |
| Load profile | 35 kW peak at 20:00 (Gaussian), 8 kW base, ±12 % stochastic |
| Timestep | 1 hour; episode = 24 steps (one day) |

---

## MDP formulation

**State** (7-dim continuous):

```
[SOC, PV_kW, load_kW, diesel_on, hour_sin, hour_cos, battery_health_index]
```

**Actions** (discrete, 5):

| Index | Action | Description |
|-------|--------|-------------|
| 0 | `charge_from_diesel` | Run diesel; route surplus to battery |
| 1 | `discharge_battery` | Serve load from battery; diesel off |
| 2 | `run_diesel_only` | Serve load from diesel; battery idle |
| 3 | `solar_only` | Serve load from PV; battery absorbs surplus |
| 4 | `shed_load` | Curtail non-critical load |

**Reward** (per step):

```
r = − fuel_cost − VoLL × unmet_kWh − battery_stress − diesel_solar_penalty
```

- **VoLL** (Value of Lost Load) = 2.00 USD/kWh applied to shed demand
- **Diesel solar penalty** = 0.10 USD/hr when diesel runs while PV > 10 % rated

---

## Agents

| Agent | Description |
|-------|-------------|
| `RuleBasedAgent` | Threshold policy: prioritise solar → discharge battery → diesel start at SOC < 0.30 |
| `RandomAgent` | Uniform random action each step (lower bound) |
| `DRLAgent` | PPO (Stable-Baselines3), trained for 200 k timesteps |

---

## Locked baselines

Evaluated over 20 episodes (seeds 0–19) under the corrected reward function
(VoLL + solar-hour diesel penalty).  These numbers are the reference target for
DRL experiments.

| Agent | Mean reward | Diesel hours | Unmet load |
|-------|------------|--------------|------------|
| Rule-based | **−61.87 ± 1.86** | 62.1 % ± 2.9 | 0.08 % ± 0.31 |
| Random | −242.85 ± 83.96 | 40.2 % ± 10.3 | 19.02 % ± 11.1 |

Results stored in `minigrid_mdp/logs/baseline_results.json`.

---

## Project structure

```
minigrid_mdp/
├── env/
│   └── minigrid_env.py      # Gymnasium environment
├── agents/
│   ├── base_agent.py
│   ├── rule_based_agent.py
│   ├── random_agent.py
│   └── drl_placeholder.py
├── config/
│   └── system_params.yaml   # All physical + reward parameters
├── logs/
│   └── baseline_results.json
├── runner.py                # run_episode() utility
└── tests/

train_ppo.py                 # Train PPO and produce comparison table
models/                      # Saved SB3 model checkpoints
```

---

## Quickstart

```bash
pip install -e ".[dev,drl]"

# Run baseline evaluation
python -m minigrid_mdp.runner

# Train PPO and compare against baselines
python train_ppo.py
```

---

## References

1. IRENA (2016). *Scaling Up Mini-grids for the Last Billion.*
2. Balogun et al. (2020). Techno-economic analysis of solar-diesel hybrid systems for rural electrification in Nigeria. *Energy Reports 6.*
3. Dufo-López & Bernal-Agustín (2015). Multi-objective design of PV–wind–diesel–hydrogen–battery systems. *Renewable Energy 82.*
4. Wöhrle et al. (2021). Battery degradation models for mini-grid sizing. *Applied Energy 290.*
