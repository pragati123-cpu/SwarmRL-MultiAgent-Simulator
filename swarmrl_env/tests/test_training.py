import pytest

torch = pytest.importorskip("torch")

from swarmrl_env.policies import (
    ActorPolicy,
    CentralizedCritic,
    MAPPOUpdater,
    RolloutBuffer,
)


def make_rollout():
    buffer = RolloutBuffer(
        capacity=3, n_agents=2, observation_dim=5, action_dim=3
    )
    actor = ActorPolicy(observation_dim=5, hidden_dim=16)
    critic = CentralizedCritic(observation_dim=5, n_agents=2, hidden_dim=16)

    for _ in range(3):
        local_observations = torch.randn(2, 5)
        actions, log_probabilities = actor(local_observations)
        buffer.add(
            local_observations,
            local_observations,
            actions,
            log_probabilities,
            torch.ones(2),
            critic(local_observations),
            terminated=False,
        )
    buffer.compute_returns_and_advantages(torch.zeros(()), last_terminated=False)
    return buffer, actor, critic


def test_rollout_buffer_computes_flattened_training_batch():
    buffer, _, _ = make_rollout()

    batch = buffer.batch()

    assert batch.local_observations.shape == (6, 5)
    assert batch.global_observations.shape == (6, 2, 5)
    assert batch.actions.shape == (6, 3)
    assert batch.advantages.shape == (6,)
    assert torch.isfinite(batch.returns).all()


def test_mappo_update_changes_policy_parameters_and_reports_metrics():
    buffer, actor, critic = make_rollout()
    batch = buffer.batch()
    before = [parameter.detach().clone() for parameter in actor.parameters()]

    metrics = MAPPOUpdater(actor, critic).update(batch)

    assert set(metrics) == {
        "actor_loss",
        "critic_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, actor.parameters())
    )