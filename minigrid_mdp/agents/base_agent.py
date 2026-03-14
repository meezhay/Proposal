"""base_agent.py — Abstract base for all dispatch agents."""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Common interface every agent must implement."""

    agent_type: str = "base"

    def __init__(self, action_space):
        self.action_space = action_space

    @abstractmethod
    def act(self, obs: np.ndarray) -> int:
        """Choose an action given the current observation vector."""

    def reset(self) -> None:
        """Called at episode start; override if agent has internal state."""
