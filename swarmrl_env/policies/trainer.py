from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from swarmrl_env.policies.actor import ActorPolicy
from swarmrl_env.policies.critic import CentralizedCritic
from swarmrl_env.policies.rollout import RolloutBuffer
from swarmrl_env.policies.updater import MAPPOUpdater


class MAPPOTrainer:
    """Collect parallel swarm rollouts and apply MAPPO updates."""

    def __init__(
        self,
        env: Any,
        rollout_length: int = 128,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        if rollout_length <= 0:
            raise ValueError("rollout_length must be positive")
        if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be between 0 and 1")

        self.env = env
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.agents = list(env.possible_agents)
        observation_dim = env.observation_space(self.agents[0]).shape[0]
        action_dim = env.action_space(self.agents[0]).shape[0]
        self.actor = ActorPolicy(observation_dim, action_dim, hidden_dim)
        self.critic = CentralizedCritic(observation_dim, len(self.agents), hidden_dim)
        self.updater = MAPPOUpdater(self.actor, self.critic)

    def train_iteration(self, seed: int | None = None) -> dict[str, float]:
        observations, _ = self.env.reset(seed=seed)
        buffer = RolloutBuffer(
            self.rollout_length,
            len(self.agents),
            self.actor.network[0].in_features,
            self.actor.action_dim,
        )
        episode_finished = False

        for _ in range(self.rollout_length):
            local_observations = self._observation_tensor(observations)
            actions, log_probabilities = self.actor(local_observations)
            value = self.critic(local_observations)
            action_dict = {
                agent: actions[index].detach().numpy()
                for index, agent in enumerate(self.agents)
            }
            next_observations, rewards, terminations, truncations, _ = self.env.step(
                action_dict
            )
            episode_finished = bool(
                terminations and all(terminations.values())
            ) or bool(truncations and all(truncations.values()))
            buffer.add(
                local_observations,
                local_observations,
                actions,
                log_probabilities,
                torch.tensor(
                    [rewards[agent] for agent in self.agents], dtype=torch.float32
                ),
                value,
                terminated=episode_finished,
            )
            observations = next_observations
            if episode_finished:
                break

        if episode_finished:
            last_value = torch.zeros(())
        else:
            last_value = self.critic(self._observation_tensor(observations))
        buffer.compute_returns_and_advantages(
            last_value,
            last_terminated=episode_finished,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        metrics = self.updater.update(buffer.batch())
        metrics["rollout_steps"] = float(buffer.position)
        return metrics

    def _observation_tensor(self, observations: dict[str, np.ndarray]) -> Tensor:
        return torch.tensor(
            np.stack([observations[agent] for agent in self.agents]),
            dtype=torch.float32,
        )