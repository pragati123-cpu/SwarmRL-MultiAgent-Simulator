import pytest

torch = pytest.importorskip("torch")

from swarmrl_env.policies import CentralizedCritic


def test_centralized_critic_returns_one_value_per_batch_item():
    critic = CentralizedCritic(observation_dim=17, n_agents=4)
    observations = torch.randn((6, 4, 17), requires_grad=True)

    values = critic(observations)
    values.sum().backward()

    assert values.shape == (6,)
    assert torch.isfinite(values).all()
    assert observations.grad is not None
    assert torch.isfinite(observations.grad).all()


def test_centralized_critic_rejects_wrong_swarm_shape():
    critic = CentralizedCritic(observation_dim=17, n_agents=4)

    with pytest.raises(ValueError, match="observations must end"):
        critic(torch.zeros((6, 3, 17)))