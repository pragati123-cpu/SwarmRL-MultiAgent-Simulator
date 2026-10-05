from __future__ import annotations

import torch
from torch import Tensor, nn


class CentralizedCritic(nn.Module):
    """Estimate a shared value from the observations of the whole swarm."""

    def __init__(
        self,
        observation_dim: int,
        n_agents: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.observation_dim = observation_dim
        self.n_agents = n_agents
        self.network = nn.Sequential(
            nn.Linear(observation_dim * n_agents, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observations: Tensor) -> Tensor:
        """Return one value estimate for each leading batch dimension."""
        expected_shape = (self.n_agents, self.observation_dim)
        if observations.shape[-2:] != expected_shape:
            raise ValueError(
                "observations must end with "
                f"(n_agents, observation_dim) = {expected_shape}, "
                f"got {tuple(observations.shape[-2:])}"
            )

        flattened = observations.reshape(*observations.shape[:-2], -1)
        return self.network(flattened).squeeze(-1)