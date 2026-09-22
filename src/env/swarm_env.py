import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class SwarmEnv(ParallelEnv):
    metadata = {
        "name": "swarmrl_v0"
    }

    _ACTION_LOW = np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32)
    _ACTION_HIGH = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    def __init__(self, num_drones=4):
        super().__init__()

        self.num_drones = num_drones

        self.possible_agents = [
            f"drone_{i}"
            for i in range(num_drones)
        ]

        self.agents = self.possible_agents.copy()

        # Each action is [velocity, pitch, yaw, roll].
        self._action_spaces = {
            agent: spaces.Box(
                low=self._ACTION_LOW,
                high=self._ACTION_HIGH,
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

    def action_space(self, agent):
        """Return the continuous action space for a drone."""
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        self.agents = self.possible_agents.copy()

        observations = {
            agent: np.zeros(1, dtype=np.float32)
            for agent in self.agents
        }

        infos = {
            agent: {}
            for agent in self.agents
        }

        return observations, infos

    def step(self, actions):
        """Execute actions for all active drones."""
        pass