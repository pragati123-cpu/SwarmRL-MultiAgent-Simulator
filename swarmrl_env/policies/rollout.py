from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RolloutBatch:
    local_observations: Tensor
    global_observations: Tensor
    actions: Tensor
    old_log_probabilities: Tensor
    advantages: Tensor
    returns: Tensor


class RolloutBuffer:
    """Fixed-size on-policy storage for one parallel swarm rollout."""

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        observation_dim: int,
        action_dim: int,
    ) -> None:
        if capacity <= 0 or n_agents <= 0 or observation_dim <= 0 or action_dim <= 0:
            raise ValueError("rollout dimensions must be positive")

        self.capacity = capacity
        self.n_agents = n_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self._local_observations = torch.zeros(
            (capacity, n_agents, observation_dim)
        )
        self._global_observations = torch.zeros_like(self._local_observations)
        self._actions = torch.zeros((capacity, n_agents, action_dim))
        self._log_probabilities = torch.zeros((capacity, n_agents))
        self._rewards = torch.zeros((capacity, n_agents))
        self._values = torch.zeros(capacity)
        self._terminated = torch.zeros(capacity, dtype=torch.bool)
        self._advantages: Tensor | None = None
        self._returns: Tensor | None = None
        self.position = 0

    @property
    def full(self) -> bool:
        return self.position == self.capacity

    def add(
        self,
        local_observations: Tensor,
        global_observations: Tensor,
        actions: Tensor,
        log_probabilities: Tensor,
        rewards: Tensor,
        values: Tensor,
        terminated: bool,
    ) -> None:
        if self.full:
            raise RuntimeError("rollout buffer is full")
        self._check_shapes(
            local_observations,
            global_observations,
            actions,
            log_probabilities,
            rewards,
            values,
        )
        index = self.position
        self._local_observations[index].copy_(local_observations.detach())
        self._global_observations[index].copy_(global_observations.detach())
        self._actions[index].copy_(actions.detach())
        self._log_probabilities[index].copy_(log_probabilities.detach())
        self._rewards[index].copy_(rewards.detach())
        self._values[index] = values.detach().reshape(())
        self._terminated[index] = terminated
        self.position += 1
        self._advantages = None
        self._returns = None

    def compute_returns_and_advantages(
        self,
        last_value: Tensor,
        last_terminated: bool,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[Tensor, Tensor]:
        if self.position == 0:
            raise RuntimeError("cannot compute advantages for an empty rollout")
        if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be between 0 and 1")

        steps = self.position
        advantages = torch.zeros((steps, self.n_agents))
        next_advantage = torch.zeros(self.n_agents)
        next_value = last_value.detach().reshape(())
        for index in range(steps - 1, -1, -1):
            nonterminal = 1.0 - float(
                self._terminated[index].item() or (index == steps - 1 and last_terminated)
            )
            delta = self._rewards[index] + gamma * next_value * nonterminal - self._values[index]
            next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
            advantages[index] = next_advantage
            next_value = self._values[index]

        self._advantages = advantages
        self._returns = advantages + self._values[:steps, None]
        return self._returns, self._advantages

    def batch(self, normalize_advantages: bool = True) -> RolloutBatch:
        if self.position == 0 or self._advantages is None or self._returns is None:
            raise RuntimeError("compute returns and advantages before creating a batch")

        advantages = self._advantages
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        steps = self.position
        return RolloutBatch(
            local_observations=self._local_observations[:steps].reshape(
                steps * self.n_agents, self.observation_dim
            ),
            global_observations=self._global_observations[:steps].repeat_interleave(
                self.n_agents, dim=0
            ),
            actions=self._actions[:steps].reshape(steps * self.n_agents, self.action_dim),
            old_log_probabilities=self._log_probabilities[:steps].reshape(
                steps * self.n_agents
            ),
            advantages=advantages.reshape(steps * self.n_agents),
            returns=self._returns[:steps].reshape(steps * self.n_agents),
        )

    def _check_shapes(
        self,
        local_observations: Tensor,
        global_observations: Tensor,
        actions: Tensor,
        log_probabilities: Tensor,
        rewards: Tensor,
        values: Tensor,
    ) -> None:
        expected_local = (self.n_agents, self.observation_dim)
        if tuple(local_observations.shape) != expected_local:
            raise ValueError(f"local_observations must have shape {expected_local}")
        if tuple(global_observations.shape) != expected_local:
            raise ValueError(f"global_observations must have shape {expected_local}")
        if tuple(actions.shape) != (self.n_agents, self.action_dim):
            raise ValueError("actions have an unexpected shape")
        if tuple(log_probabilities.shape) != (self.n_agents,):
            raise ValueError("log_probabilities have an unexpected shape")
        if tuple(rewards.shape) != (self.n_agents,):
            raise ValueError("rewards have an unexpected shape")
        if values.numel() != 1:
            raise ValueError("values must contain one centralized value")