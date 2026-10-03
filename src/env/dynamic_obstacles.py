"""Dynamic obstacle primitives used by the swarm environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DynamicObstacle:
    """A spherical obstacle with constant velocity and reflecting bounds."""

    position: np.ndarray
    velocity: np.ndarray
    radius: float = 1.0

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float32).copy()
        self.velocity = np.asarray(self.velocity, dtype=np.float32).copy()
        if self.position.shape != (3,) or self.velocity.shape != (3,):
            raise ValueError("position and velocity must each have shape (3,)")
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")

    def update(self, dt: float, bounds: float) -> None:
        """Advance the obstacle and reflect it at the cubic world boundary."""

        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        if bounds <= self.radius:
            raise ValueError("bounds must be greater than obstacle radius")

        self.position += self.velocity * np.float32(dt)
        lower = -float(bounds) + self.radius
        upper = float(bounds) - self.radius

        for axis in range(3):
            while self.position[axis] < lower or self.position[axis] > upper:
                if self.position[axis] < lower:
                    self.position[axis] = lower + (lower - self.position[axis])
                    self.velocity[axis] = abs(self.velocity[axis])
                elif self.position[axis] > upper:
                    self.position[axis] = upper - (self.position[axis] - upper)
                    self.velocity[axis] = -abs(self.velocity[axis])

    def as_array(self) -> np.ndarray:
        """Return the obstacle position and velocity as one observation value."""

        return np.concatenate((self.position, self.velocity)).astype(
            np.float32,
            copy=True,
        )


class DynamicObstacleManager:
    """Create and advance a deterministic collection of dynamic obstacles."""

    def __init__(
        self,
        count: int,
        bounds: float,
        radius: float = 1.0,
        max_speed: float = 1.0,
    ) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        if bounds <= radius:
            raise ValueError("bounds must be greater than radius")
        if max_speed < 0.0:
            raise ValueError("max_speed must be non-negative")

        self.count = int(count)
        self.bounds = float(bounds)
        self.radius = float(radius)
        self.max_speed = float(max_speed)
        self.obstacles: list[DynamicObstacle] = []

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Sample obstacle positions and velocities for a new episode."""

        rng = np.random.default_rng(seed)
        limit = self.bounds - self.radius
        self.obstacles = [
            DynamicObstacle(
                position=rng.uniform(-limit, limit, size=3),
                velocity=rng.uniform(
                    -self.max_speed,
                    self.max_speed,
                    size=3,
                ),
                radius=self.radius,
            )
            for _ in range(self.count)
        ]
        return self.as_array()

    def step(self, dt: float) -> np.ndarray:
        """Advance every obstacle and return the current state."""

        for obstacle in self.obstacles:
            obstacle.update(dt, self.bounds)
        return self.as_array()

    def as_array(self) -> np.ndarray:
        """Return shape ``(count, 6)`` obstacle state, or an empty array."""

        if not self.obstacles:
            return np.empty((0, 6), dtype=np.float32)
        return np.stack([obstacle.as_array() for obstacle in self.obstacles])
