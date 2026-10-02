from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Normal


@dataclass(frozen=True)
class FlightControls:
    """Physical controls represented by the environment's action contract."""

    throttle: Tensor
    pitch: Tensor
    yaw: Tensor


class ActorPolicy(nn.Module):
    """A shared stochastic actor for continuous per-drone actions.

    The policy consumes one local observation per row and returns normalized
    actions in ``[-1, 1]`` for throttle, pitch, and yaw. The returned log
    probability is corrected for the tanh squashing transformation so it can
    be used directly by PPO.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def distribution(self, observations: Tensor) -> Normal:
        features = self.network(observations)
        mean = self.mean(features)
        std = self.log_std.clamp(-20.0, 2.0).exp().expand_as(mean)
        return Normal(mean, std)

    def forward(
        self,
        observations: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor]:
        distribution = self.distribution(observations)
        latent_action = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(latent_action)
        log_probability = self._log_probability(distribution, latent_action, action)
        return action, log_probability

    def evaluate_actions(
        self,
        observations: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return PPO log probabilities and entropy for sampled actions."""
        clipped_actions = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        latent_actions = torch.atanh(clipped_actions)
        distribution = self.distribution(observations)
        log_probability = self._log_probability(
            distribution, latent_actions, clipped_actions
        )
        entropy = distribution.entropy().sum(dim=-1)
        return log_probability, entropy

    def map_action_to_controls(
        self,
        action: Tensor,
        max_speed: float,
    ) -> FlightControls:
        """Map normalized policy output to physical flight controls."""
        if action.shape[-1] != 3:
            raise ValueError("action must have three values: throttle, pitch, yaw")
        if max_speed < 0.0:
            raise ValueError("max_speed must be non-negative")

        bounded_action = action.clamp(-1.0, 1.0)
        return FlightControls(
            throttle=(bounded_action[..., 0] + 1.0) * 0.5 * max_speed,
            pitch=bounded_action[..., 1] * (torch.pi / 2.0),
            yaw=bounded_action[..., 2] * torch.pi,
        )

    @staticmethod
    def _log_probability(
        distribution: Normal,
        latent_action: Tensor,
        action: Tensor,
    ) -> Tensor:
        correction = torch.log(1.0 - action.square() + 1e-6)
        return (distribution.log_prob(latent_action) - correction).sum(dim=-1)