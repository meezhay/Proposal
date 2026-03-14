"""
01_rule_based_episode.py
========================
Run a single 24-hour episode with the RuleBasedAgent and visualise the results.
Execute as a plain Python script or open in Jupyter as a percent-script notebook.

  python minigrid_mdp/notebooks/01_rule_based_episode.py
"""

# %% [markdown]
# # Mini-Grid Dispatch — Rule-Based Agent Walkthrough
#
# This notebook walks through a single 24-hour episode of the `MiniGridEnv`
# driven by the `RuleBasedAgent`, then plots the key dispatch signals.

# %% Imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — swap to "TkAgg" / "Qt5Agg" for interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from minigrid_mdp.env.minigrid_env import MiniGridEnv
from minigrid_mdp.agents import RuleBasedAgent, RandomAgent

# %% [markdown]
# ## 1 — Run one episode per agent

# %%
ACTION_LABELS = {
    0: "charge_diesel",
    1: "discharge_bat",
    2: "diesel_only",
    3: "solar_only",
    4: "shed_load",
}

def run_single_episode(AgentClass, seed: int = 42) -> pd.DataFrame:
    env   = MiniGridEnv()
    obs, _ = env.reset(seed=seed)
    agent  = AgentClass(env.action_space)
    done   = False
    while not done:
        action = agent.act(obs)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    df = env.log_as_dataframe()
    df["agent_type"] = agent.agent_type
    df["action_label"] = df["action"].map(ACTION_LABELS)
    return df

df_rule   = run_single_episode(RuleBasedAgent, seed=42)
df_random = run_single_episode(RandomAgent,    seed=42)

print("Rule-based episode summary:")
print(df_rule[["hour","soc","pv_kw","load_kw","diesel_on","unmet_load_kw",
               "reward","action_label"]].to_string(index=False))

# %% [markdown]
# ## 2 — Summary statistics

# %%
def episode_summary(df: pd.DataFrame) -> dict:
    return {
        "agent"            : df["agent_type"].iloc[0],
        "total_reward"     : df["reward"].sum().round(3),
        "unmet_load_kwh"   : df["unmet_load_kw"].sum().round(3),
        "fuel_cost_usd"    : (-df["rew_fuel"]).sum().round(3),
        "bat_stress_cost"  : (-df["rew_battery_stress"]).sum().round(4),
        "diesel_hours"     : int(df["diesel_on"].sum()),
        "final_soc"        : df["soc_after"].iloc[-1].round(4),
        "final_health"     : df["health"].iloc[-1].round(5),
    }

summary = pd.DataFrame([episode_summary(df_rule), episode_summary(df_random)])
print("\nComparison:")
print(summary.to_string(index=False))

# %% [markdown]
# ## 3 — Dispatch visualisation

# %%
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(4, 1, hspace=0.45)

hours = df_rule["hour"]

# ── Panel 1: Power flows ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.fill_between(hours, df_rule["pv_kw"],  alpha=0.4, color="#f7c241", label="PV output (kW)")
ax1.fill_between(hours, df_rule["load_kw"], alpha=0.3, color="#e74c3c", label="Load demand (kW)")
ax1.bar(hours, df_rule["diesel_kw"], color="#7f8c8d", alpha=0.6, label="Diesel (kW)", width=0.7)
ax1.bar(hours, df_rule["unmet_load_kw"], color="#c0392b", alpha=0.9, label="Unmet load (kW)", width=0.7)
ax1.set_ylabel("Power (kW)")
ax1.set_title("Rule-Based Agent — 24-Hour Dispatch")
ax1.legend(loc="upper left", fontsize=8)
ax1.set_xlim(-0.5, 23.5)
ax1.set_xticks(range(0, 24, 2))

# ── Panel 2: Battery SOC ──────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.plot(hours, df_rule["soc"],       color="#2980b9", lw=2, label="SOC (start of step)")
ax2.plot(hours, df_rule["soc_after"], color="#2980b9", lw=1, ls="--", alpha=0.6, label="SOC (end of step)")
ax2.axhline(0.30, color="orange",  ls=":", lw=1, label="LOW_SOC (0.30)")
ax2.axhline(0.15, color="red",     ls=":", lw=1, label="CRIT_SOC (0.15)")
ax2.axhline(0.85, color="green",   ls=":", lw=1, label="HIGH_SOC (0.85)")
ax2.set_ylabel("State of Charge")
ax2.set_ylim(0, 1.05)
ax2.legend(loc="upper right", fontsize=8)
ax2.set_xlim(-0.5, 23.5)
ax2.set_xticks(range(0, 24, 2))

# ── Panel 3: Reward components ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.bar(hours, df_rule["rew_unmet_load"],     label="Unmet load",     color="#e74c3c", alpha=0.8)
ax3.bar(hours, df_rule["rew_fuel"],           label="Fuel cost",      color="#7f8c8d", alpha=0.8)
ax3.bar(hours, df_rule["rew_battery_stress"], label="Battery stress",  color="#8e44ad", alpha=0.8)
ax3.bar(hours, df_rule["rew_diesel_idle"],    label="Diesel idle",    color="#d35400", alpha=0.8)
ax3.set_ylabel("Reward ($)")
ax3.set_title("Reward Components per Step")
ax3.legend(loc="lower left", fontsize=8)
ax3.set_xlim(-0.5, 23.5)
ax3.set_xticks(range(0, 24, 2))

# ── Panel 4: Action taken ─────────────────────────────────────────────────────
action_colors = {
    "charge_diesel":  "#e67e22",
    "discharge_bat":  "#2980b9",
    "diesel_only":    "#7f8c8d",
    "solar_only":     "#f7c241",
    "shed_load":      "#c0392b",
}
ax4 = fig.add_subplot(gs[3])
for h, lbl in zip(hours, df_rule["action_label"]):
    ax4.bar(h, 1, color=action_colors.get(lbl, "grey"), alpha=0.85, width=0.9)

# Legend
from matplotlib.patches import Patch
legend_patches = [Patch(color=c, label=l) for l, c in action_colors.items()]
ax4.legend(handles=legend_patches, loc="upper right", fontsize=8, ncol=3)
ax4.set_yticks([])
ax4.set_xlabel("Hour of Day")
ax4.set_title("Action Taken per Hour")
ax4.set_xlim(-0.5, 23.5)
ax4.set_xticks(range(0, 24, 2))

out_path = Path(__file__).parent / "rule_based_episode.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nPlot saved → {out_path}")
plt.close()

# %% [markdown]
# ## 4 — Export logs

# %%
from pathlib import Path
log_dir = Path(__file__).resolve().parents[1] / "logs"
log_dir.mkdir(exist_ok=True)

combined = pd.concat([df_rule, df_random], ignore_index=True)
combined.to_csv(log_dir / "notebook_episode_logs.csv", index=False)
print(f"Logs saved → {log_dir / 'notebook_episode_logs.csv'}")
print("\nDone.")
