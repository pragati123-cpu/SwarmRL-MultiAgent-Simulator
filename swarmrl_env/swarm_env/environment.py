from __future__ import annotations

import functools
from typing import Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class SwarmEnv(ParallelEnv):

    metadata = {
        "name": "swarmrl_v0",
        "is_parallelizable": True,
        "render_modes": ["human", None],
    }

    def __init__(
        self,
        n_agents: int = 50,
        n_neighbors: int = 5,
        world_size: float = 100.0,
        max_episode_steps: int = 500,
        collision_radius: float = 1.5,
        max_speed: float = 2.0,
        grid_cell_size: float = 5.0,
        render_mode: Optional[str] = None,
    ):

        super().__init__()

        if n_agents < 2:
            raise ValueError("n_agents must be >= 2 so neighbors can be computed")

        self.possible_agents = [f"drone_{i}" for i in range(n_agents)]
        self.n_neighbors = int(min(n_neighbors, n_agents - 1))
        self.world_size = float(world_size)
        self.half_size = self.world_size / 2.0
        self.max_episode_steps = int(max_episode_steps)
        self.collision_radius = float(collision_radius)
        self.max_speed = float(max_speed)
        self.grid_cell_size = float(grid_cell_size)
        self.render_mode = render_mode

        # own pos(3) + own vel(3) + k neighbor relative positions(3k) + 6 wall distances
        self._obs_dim = 3 + 3 + 3 * self.n_neighbors + 6

        self._action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
            )
            for agent in self.possible_agents
        }

        # Runtime state (populated in reset())
        self.agents: list[str] = []
        self.positions: dict[str, np.ndarray] = {}
        self.velocities: dict[str, np.ndarray] = {}
        self.timestep: int = 0
        self.visited_cells: set[tuple[int, int, int]] = set()
        self.np_random: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------ #
    # PettingZoo required space accessors (cached per agent id)
    # ------------------------------------------------------------------ #
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    # ------------------------------------------------------------------ #
    # Core loop: reset()
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]
        self.timestep = 0
        self.visited_cells = set()

        options = options or {}
        spawn_fraction = float(options.get("spawn_fraction", 0.5))
        spawn_half = self.half_size * spawn_fraction

        # Spawn agents randomly within a fraction of the world volume so
        # they start clustered near the center rather than at the walls.
        self.positions = {
            agent: self.np_random.uniform(
                low=-spawn_half, high=spawn_half, size=3
            ).astype(np.float32)
            for agent in self.agents
        }
        self.velocities = {
            agent: np.zeros(3, dtype=np.float32) for agent in self.agents
        }

        # Mark each agent's spawn cell as already-visited so the very
        # first step doesn't hand out a free exploration reward for
        # simply existing.
        for agent in self.agents:
            self.visited_cells.add(self._position_to_cell(self.positions[agent]))

        observations = {
            agent: self._get_observation(agent) for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    # ------------------------------------------------------------------ #
    # Core loop: step()
    # ------------------------------------------------------------------ #
    def step(self, actions: dict):
    
        if not self.agents:
            # step() called after episode already ended
            return {}, {}, {}, {}, {}

        # 1) Apply actions -> update velocity & position for every agent.
        for agent in self.agents:
            if agent not in actions:
                # Missing action (e.g. a crashed/disconnected node):
                # treat as hover -> zero velocity this step.
                self.velocities[agent] = np.zeros(3, dtype=np.float32)
                continue

            action = np.clip(np.asarray(actions[agent], dtype=np.float32), -1.0, 1.0)
            throttle_cmd, pitch_cmd, yaw_cmd = action

            speed = (throttle_cmd + 1.0) / 2.0 * self.max_speed  # [0, max_speed]
            pitch = pitch_cmd * (np.pi / 2.0)  # [-pi/2, pi/2]
            yaw = yaw_cmd * np.pi  # [-pi, pi]

            vx = speed * np.cos(pitch) * np.cos(yaw)
            vy = speed * np.cos(pitch) * np.sin(yaw)
            vz = speed * np.sin(pitch)
            velocity = np.array([vx, vy, vz], dtype=np.float32)

            self.velocities[agent] = velocity
            new_pos = self.positions[agent] + velocity
            # Clip to world bounds (walls act as a hard, non-destructive limit).
            new_pos = np.clip(new_pos, -self.half_size, self.half_size)
            self.positions[agent] = new_pos.astype(np.float32)

        # 2) Collision detection (pairwise distance < collision_radius).
        collided_agents = self._detect_collisions()

        # 3) Rewards: exploration bonus + collision penalty.
        rewards = {}
        for agent in self.agents:
            reward = 0.0
            cell = self._position_to_cell(self.positions[agent])
            if cell not in self.visited_cells:
                self.visited_cells.add(cell)
                reward += 1.0
            if agent in collided_agents:
                reward -= 100.0
            rewards[agent] = float(reward)

        # 4) Termination / truncation bookkeeping.
        self.timestep += 1
        time_up = self.timestep >= self.max_episode_steps
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: time_up for agent in self.agents}

        # 5) Observations + infos.
        observations = {
            agent: self._get_observation(agent) for agent in self.agents
        }
        infos = {
            agent: {
                "collided": agent in collided_agents,
                "coverage_cells": len(self.visited_cells),
            }
            for agent in self.agents
        }

        # PettingZoo convention: once every agent is done, clear self.agents.
        if time_up or all(terminations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _get_observation(self, agent: str) -> np.ndarray:
        own_pos = self.positions[agent]
        own_vel = self.velocities[agent]

        neighbor_rel_positions = self._nearest_neighbor_offsets(agent)

        # Distance to each of the 6 bounding walls, normalized to [0, 1].
        wall_distances = np.array(
            [
                (own_pos[0] - (-self.half_size)),  # to x-min wall
                (self.half_size - own_pos[0]),      # to x-max wall
                (own_pos[1] - (-self.half_size)),  # to y-min wall
                (self.half_size - own_pos[1]),      # to y-max wall
                (own_pos[2] - (-self.half_size)),  # to z-min wall
                (self.half_size - own_pos[2]),      # to z-max wall
            ],
            dtype=np.float32,
        ) / self.world_size

        obs = np.concatenate(
            [
                own_pos / self.half_size,
                own_vel / max(self.max_speed, 1e-6),
                neighbor_rel_positions / self.world_size,
                wall_distances,
            ]
        ).astype(np.float32)

        assert obs.shape[0] == self._obs_dim
        return obs

    def _nearest_neighbor_offsets(self, agent: str) -> np.ndarray:

        own_pos = self.positions[agent]
        others = [a for a in self.agents if a != agent]

        if not others:
            return np.zeros(3 * self.n_neighbors, dtype=np.float32)

        other_positions = np.stack([self.positions[a] for a in others])
        deltas = other_positions - own_pos  # (num_others, 3)
        dists = np.linalg.norm(deltas, axis=1)

        k = min(self.n_neighbors, len(others))
        nearest_idx = np.argsort(dists)[:k]
        nearest_deltas = deltas[nearest_idx]

        if k < self.n_neighbors:
            pad = np.zeros((self.n_neighbors - k, 3), dtype=np.float32)
            nearest_deltas = np.concatenate([nearest_deltas, pad], axis=0)

        return nearest_deltas.flatten().astype(np.float32)

    def _detect_collisions(self) -> set[str]:
        agents = self.agents
        n = len(agents)
        if n < 2:
            return set()

        positions = np.stack([self.positions[a] for a in agents])  # (n, 3)
        # Pairwise distance matrix.
        diff = positions[:, None, :] - positions[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist_matrix, np.inf)

        collided_mask = np.any(dist_matrix < self.collision_radius, axis=1)
        return {agents[i] for i in range(n) if collided_mask[i]}

    def _position_to_cell(self, position: np.ndarray) -> tuple[int, int, int]:
        cell = np.floor(position / self.grid_cell_size).astype(int)
        return int(cell[0]), int(cell[1]), int(cell[2])

    # ------------------------------------------------------------------ #
    # Misc PettingZoo API
    # ------------------------------------------------------------------ #
    def render(self):
        # Actual 3D rendering is handled by the React/Three.js dashboard
        # (see Week 1/2 "Simulation & Rendering" track). This is a no-op
        # placeholder kept for API compatibility with tools that expect it.
        if self.render_mode == "human":
            coverage = len(self.visited_cells)
            print(f"[t={self.timestep}] active_agents={len(self.agents)} coverage_cells={coverage}")

    def close(self):
        pass
