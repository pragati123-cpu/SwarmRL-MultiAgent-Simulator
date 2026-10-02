from __future__ import annotations

import torch
from torch import Tensor

from swarmrl_env.policies.actor import ActorPolicy
from swarmrl_env.policies.critic import CentralizedCritic
from swarmrl_env.policies.rollout import RolloutBatch


class MAPPOUpdater:
    """Perform clipped PPO updates for a shared actor and centralized critic."""

    def __init__(
        self,
        actor: ActorPolicy,
        critic: CentralizedCritic,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 1e-3,
        clip_ratio: float = 0.2,
        value_loss_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
    ) -> None:
        if actor_learning_rate <= 0.0 or critic_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if clip_ratio <= 0.0:
            raise ValueError("clip_ratio must be positive")
        if value_loss_coefficient < 0.0 or entropy_coefficient < 0.0:
            raise ValueError("loss coefficients must be non-negative")

        self.actor = actor
        self.critic = critic
        self.clip_ratio = clip_ratio
        self.value_loss_coefficient = value_loss_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.actor_optimizer = torch.optim.Adam(
            actor.parameters(), lr=actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=critic_learning_rate
        )

    def update(self, batch: RolloutBatch, epochs: int = 1) -> dict[str, float]:
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        metrics: dict[str, float] = {}
        for _ in range(epochs):
            log_probabilities, entropy = self.actor.evaluate_actions(
                batch.local_observations, batch.actions
            )
            ratio = (log_probabilities - batch.old_log_probabilities).exp()
            unclipped = ratio * batch.advantages
            clipped = ratio.clamp(
                1.0 - self.clip_ratio, 1.0 + self.clip_ratio
            ) * batch.advantages
            actor_loss = -torch.minimum(unclipped, clipped).mean()

            values = self.critic(batch.global_observations).reshape(-1)
            critic_loss = 0.5 * (batch.returns - values).square().mean()
            total_loss = (
                actor_loss
                + self.value_loss_coefficient * critic_loss
                - self.entropy_coefficient * entropy.mean()
            )

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            metrics = {
                "actor_loss": self._scalar(actor_loss),
                "critic_loss": self._scalar(critic_loss),
                "entropy": self._scalar(entropy.mean()),
                "approx_kl": self._scalar(
                    (batch.old_log_probabilities - log_probabilities).mean()
                ),
                "clip_fraction": self._scalar(
                    (torch.abs(ratio - 1.0) > self.clip_ratio).float().mean()
                ),
            }
        return metrics

    @staticmethod
    def _scalar(value: Tensor) -> float:
        return float(value.detach().cpu().item())