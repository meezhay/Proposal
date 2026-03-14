"""
tests/test_env.py — Basic sanity checks for MiniGridEnv.

Run with:  pytest minigrid_mdp/tests/ -v
"""

import pytest
import numpy as np
import gymnasium as gym

from minigrid_mdp.env.minigrid_env import MiniGridEnv, N_ACTIONS
from minigrid_mdp.agents import RandomAgent, RuleBasedAgent


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    e = MiniGridEnv()
    e.reset(seed=0)
    return e


# ─────────────────────────────────────────────────────────────────────────────
# 1. Observation space
# ─────────────────────────────────────────────────────────────────────────────

def test_obs_shape(env):
    obs, _ = env.reset()
    assert obs.shape == (7,), f"Expected (7,), got {obs.shape}"


def test_obs_within_bounds(env):
    obs, _ = env.reset()
    assert env.observation_space.contains(obs), \
        f"Initial obs out of bounds: {obs}"


def test_obs_bounds_after_steps(env):
    obs, _ = env.reset(seed=42)
    for _ in range(24):
        action = env.action_space.sample()
        obs, *_ = env.step(action)
        assert env.observation_space.contains(obs), \
            f"Obs out of bounds after step: {obs}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Action space
# ─────────────────────────────────────────────────────────────────────────────

def test_action_space_size(env):
    assert env.action_space.n == N_ACTIONS


def test_all_actions_valid(env):
    for action in range(N_ACTIONS):
        e = MiniGridEnv()
        e.reset(seed=0)
        obs, reward, term, trunc, info = e.step(action)
        assert obs is not None
        assert isinstance(reward, float)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Episode termination
# ─────────────────────────────────────────────────────────────────────────────

def test_episode_terminates_after_24_steps(env):
    env.reset(seed=1)
    done = False
    steps = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
    assert steps == 24, f"Episode should last 24 steps, lasted {steps}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. SOC physics
# ─────────────────────────────────────────────────────────────────────────────

def test_soc_stays_in_bounds():
    e = MiniGridEnv()
    e.reset(seed=5)
    bat = e.bat_cfg
    for _ in range(24):
        e.step(e.action_space.sample())
        assert bat["min_soc"] <= e._soc <= bat["max_soc"], \
            f"SOC out of bounds: {e._soc}"


def test_battery_health_non_increasing():
    e = MiniGridEnv()
    e.reset(seed=7)
    prev_health = e._health
    for _ in range(24):
        e.step(e.action_space.sample())
        assert e._health <= prev_health + 1e-9, \
            "Battery health should be non-increasing"
        prev_health = e._health


# ─────────────────────────────────────────────────────────────────────────────
# 5. Logging
# ─────────────────────────────────────────────────────────────────────────────

def test_log_length_matches_steps():
    e = MiniGridEnv()
    e.reset(seed=3)
    for _ in range(24):
        e.step(e.action_space.sample())
    assert len(e.log) == 24


def test_log_dataframe_columns():
    import pandas as pd
    e = MiniGridEnv()
    e.reset(seed=3)
    for _ in range(24):
        e.step(e.action_space.sample())
    df = e.log_as_dataframe()
    required_cols = {"step", "hour", "soc", "pv_kw", "load_kw",
                     "diesel_on", "unmet_load_kw", "reward", "health"}
    assert required_cols.issubset(df.columns), \
        f"Missing columns: {required_cols - set(df.columns)}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Agents
# ─────────────────────────────────────────────────────────────────────────────

def test_random_agent_full_episode():
    e = MiniGridEnv()
    obs, _ = e.reset(seed=0)
    agent = RandomAgent(e.action_space)
    done = False
    while not done:
        action = agent.act(obs)
        obs, _, term, trunc, _ = e.step(action)
        done = term or trunc
    assert len(e.log) == 24


def test_rule_based_agent_full_episode():
    e = MiniGridEnv()
    obs, _ = e.reset(seed=0)
    agent = RuleBasedAgent(e.action_space)
    done = False
    while not done:
        action = agent.act(obs)
        obs, _, term, trunc, _ = e.step(action)
        done = term or trunc
    assert len(e.log) == 24


def test_rule_based_sheds_at_critical_soc():
    """Force SOC to critical level and verify agent sheds load."""
    from minigrid_mdp.env.minigrid_env import ACTION_SHED_LOAD
    e = MiniGridEnv()
    obs, _ = e.reset(seed=0)
    agent = RuleBasedAgent(e.action_space)

    # Manually set SOC below critical threshold
    e._soc = 0.10
    obs = e._make_obs()

    action = agent.act(obs)
    assert action == ACTION_SHED_LOAD, \
        f"Expected ACTION_SHED_LOAD at critical SOC, got {action}"


def test_rule_based_charges_at_low_soc():
    """Force SOC to low level and verify agent charges from diesel."""
    from minigrid_mdp.env.minigrid_env import ACTION_CHARGE_FROM_DIESEL
    e = MiniGridEnv()
    obs, _ = e.reset(seed=0)
    agent = RuleBasedAgent(e.action_space)

    e._soc = 0.25   # above CRIT (0.15), below LOW (0.30)
    obs = e._make_obs()

    action = agent.act(obs)
    assert action == ACTION_CHARGE_FROM_DIESEL, \
        f"Expected ACTION_CHARGE_FROM_DIESEL at low SOC, got {action}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Reward sanity
# ─────────────────────────────────────────────────────────────────────────────

def test_reward_is_non_positive():
    """All reward components are penalties; total reward ≤ 0."""
    e = MiniGridEnv()
    e.reset(seed=0)
    for _ in range(24):
        _, reward, _, _, _ = e.step(e.action_space.sample())
        assert reward <= 1e-6, f"Reward should be ≤ 0, got {reward}"


def test_reward_worse_with_shed_load_at_full_soc():
    """Shed-load at high SOC should be worse than solar-only."""
    from minigrid_mdp.env.minigrid_env import ACTION_SOLAR_ONLY, ACTION_SHED_LOAD

    rewards = {}
    for action_idx, action in enumerate([ACTION_SOLAR_ONLY, ACTION_SHED_LOAD]):
        e = MiniGridEnv()
        e.reset(seed=0)
        e._soc = 0.80   # high SOC — no need to shed
        _, r, _, _, _ = e.step(action)
        rewards[action] = r

    assert rewards[ACTION_SOLAR_ONLY] >= rewards[ACTION_SHED_LOAD], \
        "Solar-only should outperform shed-load at high SOC"
