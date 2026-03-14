"""
runner.py — Run one or more agents through the MiniGridEnv for N episodes.

Usage:
    python -m minigrid_mdp.runner --agent rule_based --episodes 5
    python -m minigrid_mdp.runner --agent random     --episodes 5
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from minigrid_mdp.env.minigrid_env import MiniGridEnv
from minigrid_mdp.agents import RandomAgent, RuleBasedAgent


LOG_DIR = Path(__file__).parent / "logs"


def run_episode(env: MiniGridEnv, agent, episode_idx: int = 0) -> pd.DataFrame:
    """Run one episode; return a timestep log DataFrame."""
    obs, _ = env.reset(seed=episode_idx)
    agent.reset()

    done = False
    while not done:
        action   = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    df = env.log_as_dataframe()
    df.insert(0, "episode", episode_idx)
    df.insert(1, "agent_type", agent.agent_type)
    return df


def run_agent(agent_name: str, n_episodes: int = 1, save_csv: bool = True) -> pd.DataFrame:
    """Run an agent for N episodes and return the combined log."""
    env = MiniGridEnv()

    if agent_name == "rule_based":
        agent = RuleBasedAgent(env.action_space)
    elif agent_name == "random":
        agent = RandomAgent(env.action_space)
    else:
        raise ValueError(f"Unknown agent: {agent_name!r}. Choose 'rule_based' or 'random'.")

    frames = []
    for ep in range(n_episodes):
        df = run_episode(env, agent, episode_idx=ep)
        frames.append(df)
        ep_reward = df["reward"].sum()
        print(f"  Episode {ep:3d} | total_reward={ep_reward:8.3f} | "
              f"unmet_kwh={df['unmet_load_kw'].sum():.2f} | "
              f"fuel_cost=${-df['rew_fuel'].sum():.2f}")

    all_logs = pd.concat(frames, ignore_index=True)

    if save_csv:
        LOG_DIR.mkdir(exist_ok=True)
        out_path = LOG_DIR / f"{agent_name}_episodes.csv"
        all_logs.to_csv(out_path, index=False)
        print(f"\n  Saved → {out_path}")

    return all_logs


def main():
    parser = argparse.ArgumentParser(description="Mini-grid dispatch runner")
    parser.add_argument("--agent",    default="rule_based",
                        choices=["rule_based", "random"])
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    print(f"\n{'─'*55}")
    print(f"  Agent: {args.agent}   Episodes: {args.episodes}")
    print(f"{'─'*55}")
    df = run_agent(args.agent, n_episodes=args.episodes)

    print(f"\n  Summary across {args.episodes} episodes:")
    print(f"    Mean episode reward : {df.groupby('episode')['reward'].sum().mean():.3f}")
    print(f"    Unmet load rate     : {(df['unmet_load_kw'] > 0).mean()*100:.1f}%")
    print(f"    Diesel usage rate   : {df['diesel_on'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
