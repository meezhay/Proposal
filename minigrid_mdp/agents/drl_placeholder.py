"""
drl_placeholder.py — Stub for a future Stable-Baselines3 DRL agent.

Phase 2 will replace this with a trained PPO / SAC policy.
Import guarded so the module loads even without stable-baselines3 installed.
"""

from __future__ import annotations
import numpy as np
from .base_agent import BaseAgent


class DRLAgent(BaseAgent):
    """
    Wraps a stable-baselines3 policy loaded from disk.

    Usage (Phase 2):
        agent = DRLAgent(env.action_space, model_path="logs/ppo_minigrid.zip")
    """

    agent_type = "drl"

    def __init__(self, action_space, model_path: str | None = None):
        super().__init__(action_space)
        self._model = None

        if model_path is not None:
            try:
                from stable_baselines3 import PPO
                self._model = PPO.load(model_path)
            except ImportError:
                raise ImportError(
                    "stable-baselines3 is required for DRLAgent. "
                    "Install it with: pip install stable-baselines3"
                )

    def act(self, obs: np.ndarray) -> int:
        if self._model is None:
            raise RuntimeError(
                "No model loaded.  Provide model_path= when constructing DRLAgent, "
                "or train a model first (Phase 2)."
            )
        action, _ = self._model.predict(obs, deterministic=True)
        return int(action)
