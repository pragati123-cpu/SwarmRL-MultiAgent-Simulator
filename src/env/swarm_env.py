import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .dynamic_obstacles import DynamicObstacleManager


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
        "render_modes": ["state"],
    }

    def __init__(
        self,
        num_drones=3,
        max_cycles=100,
        obstacle_count=0,
        obstacle_bounds=10.0,
        obstacle_radius=1.0,
        obstacle_max_speed=1.0,
        obstacle_dt=1.0,
        drone_radius=0.5,
        obstacle_collision_penalty=10.0,
    ):
        super().__init__()

        self.num_drones = num_drones
        self.max_cycles = max_cycles
        if obstacle_dt <= 0.0:
            raise ValueError("obstacle_dt must be positive")
        if drone_radius <= 0.0:
            raise ValueError("drone_radius must be positive")
        if obstacle_collision_penalty < 0.0:
            raise ValueError("obstacle_collision_penalty must be non-negative")
        self.obstacle_dt = float(obstacle_dt)
        self.drone_radius = float(drone_radius)
        self.obstacle_collision_penalty = float(obstacle_collision_penalty)
        self.obstacle_manager = DynamicObstacleManager(
            count=obstacle_count,
            bounds=obstacle_bounds,
            radius=obstacle_radius,
            max_speed=obstacle_max_speed,
        )
        self.obstacle_states = np.empty((0, 6), dtype=np.float32)

        self.possible_agents = [
            f"drone_{i}" for i in range(num_drones)
        ]

        self.agents = self.possible_agents.copy()

        # Observation:
        # x, y, z, vx, vy, vz, roll, pitch, yaw,
        # normalized distance to each configured obstacle.
        self._observation_dim = 9 + self.obstacle_manager.count
        self.observation_spaces = {
            agent: spaces.Box(
                low=np.concatenate(
                    (
                        np.full(9, -np.inf, dtype=np.float32),
                        np.zeros(
                            self.obstacle_manager.count,
                            dtype=np.float32,
                        ),
                    )
                ),
                high=np.concatenate(
                    (
                        np.full(9, np.inf, dtype=np.float32),
                        np.ones(
                            self.obstacle_manager.count,
                            dtype=np.float32,
                        ),
                    )
                ),
                shape=(self._observation_dim,),
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

    def _get_observation(self, agent):
        """Return the drone state with normalized obstacle distances."""

        state = self.state[agent]
        if not self.obstacle_manager.obstacles:
            return state.copy()

        position = state[:3]
        obstacle_positions = self.obstacle_states[:, :3]
        distances = np.linalg.norm(
            obstacle_positions - position,
            axis=1,
        )
        normalized_distances = np.clip(
            distances / self.obstacle_manager.bounds,
            0.0,
            1.0,
        ).astype(np.float32)
        return np.concatenate((state, normalized_distances)).astype(
            np.float32,
            copy=False,
        )

    def _detect_obstacle_collisions(self):
        """Return agents whose drone sphere intersects an obstacle sphere."""

        if not self.obstacle_manager.obstacles:
            return set()

        collisions = set()
        obstacle_positions = self.obstacle_states[:, :3]
        collision_distance = (
            self.drone_radius + self.obstacle_manager.radius
        )
        for agent in self.agents:
            distances = np.linalg.norm(
                obstacle_positions - self.state[agent][:3],
                axis=1,
            )
            if np.any(distances <= collision_distance):
                collisions.add(agent)
        return collisions

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents.copy()
        self.step_count = 0
        self.obstacle_states = self.obstacle_manager.reset(seed=seed)

        self.state = {
            agent: np.zeros(9, dtype=np.float32)
            for agent in self.possible_agents
        }

        observations = {
            agent: self._get_observation(agent)
            for agent in self.agents
        }

        infos = {
            agent: {
                "obstacle_states": self.obstacle_states.copy(),
            }
            for agent in self.agents
        }

        return observations, infos

    def step(self, actions):
        self.step_count += 1
        self.obstacle_states = self.obstacle_manager.step(self.obstacle_dt)

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        action_rewards = {}

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
            action_rewards[agent] = float(1.0 - np.mean(np.square(action)))

        collided_agents = self._detect_obstacle_collisions()
        for agent in self.agents:
            # Encourage actions close to zero/stable flight and penalize
            # intersections with moving obstacles.
            action_reward = action_rewards[agent]
            collision_penalty = (
                self.obstacle_collision_penalty
                if agent in collided_agents
                else 0.0
            )
            reward = float(action_reward - collision_penalty)

            observations[agent] = self._get_observation(agent)
            rewards[agent] = reward

            terminations[agent] = False
            truncations[agent] = self.step_count >= self.max_cycles

            infos[agent] = {
                "step": self.step_count,
                "obstacle_states": self.obstacle_states.copy(),
                "obstacle_collision": agent in collided_agents,
                "reward_components": {
                    "action": float(action_reward),
                    "obstacle_collision": float(-collision_penalty),
                },
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
        """Return a serializable snapshot for visualization clients."""

        return {
            "step": self.step_count,
            "drones": {
                agent: self.state[agent][:3].copy()
                for agent in self.possible_agents
            }
            if self.state is not None
            else {},
            "obstacles": self.obstacle_states[:, :3].copy(),
            "obstacle_radii": np.full(
                self.obstacle_manager.count,
                self.obstacle_manager.radius,
                dtype=np.float32,
            ),
        }

    def close(self):
        pass