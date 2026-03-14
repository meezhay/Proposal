"""random_agent.py — Uniform random action selection (sanity-check baseline)."""

from __future__ import annotations
import numpy as np
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Selects uniformly at random from the discrete action space.
    Purpose: establishes a lower-bound performance floor for comparison.
    """

    agent_type = "random"

    def act(self, obs: np.ndarray) -> int:
        return int(self.action_space.sample())
