import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class SwarmEnv(ParallelEnv):
    """
    Custom PettingZoo Parallel environment for a drone swarm.

    Each drone receives:
        - position: x, y, z
        - velocity: vx, vy, vz
        - orientation: roll, pitch, yaw

    Each drone controls:
        - velocity
        - pitch
        - yaw
        - roll
    """

    metadata = {
        "name": "swarm_env_v0",
        "render_modes": [],
    }

    def __init__(self, num_drones=3, max_cycles=100):
        super().__init__()

        self.num_drones = num_drones
        self.max_cycles = max_cycles

        self.possible_agents = [
            f"drone_{i}" for i in range(num_drones)
        ]

        self.agents = self.possible_agents.copy()

        # Observation:
        # x, y, z
        # vx, vy, vz
        # roll, pitch, yaw
        self.observation_spaces = {
            agent: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(9,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        # Action:
        # velocity, pitch, yaw, roll
        self.action_spaces = {
            agent: spaces.Box(
                low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        self.state = None
        self.step_count = 0

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents.copy()
        self.step_count = 0

        self.state = {
            agent: np.zeros(9, dtype=np.float32)
            for agent in self.possible_agents
        }

        observations = {
            agent: self.state[agent].copy()
            for agent in self.agents
        }

        infos = {
            agent: {}
            for agent in self.agents
        }

        return observations, infos

    def step(self, actions):
        self.step_count += 1

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent in self.agents:
            action = np.asarray(
                actions[agent],
                dtype=np.float32,
            )

            # Clip action to valid range.
            action = np.clip(action, -1.0, 1.0)

            velocity = action[0]
            pitch = action[1]
            yaw = action[2]
            roll = action[3]

            # Simple state update.
            self.state[agent][0] += velocity * 0.1
            self.state[agent][1] += pitch * 0.1
            self.state[agent][2] += roll * 0.1

            self.state[agent][3] = velocity
            self.state[agent][4] = pitch
            self.state[agent][5] = yaw

            self.state[agent][6] = roll
            self.state[agent][7] = pitch
            self.state[agent][8] = yaw

            # Simple cooperative reward.
            # Encourage actions close to zero/stable flight.
            reward = float(1.0 - np.mean(np.square(action)))

            observations[agent] = self.state[agent].copy()
            rewards[agent] = reward

            terminations[agent] = False
            truncations[agent] = self.step_count >= self.max_cycles

            infos[agent] = {
                "step": self.step_count,
            }

        if self.step_count >= self.max_cycles:
            self.agents = []

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )

    def render(self):
        pass

    def close(self):
        pass