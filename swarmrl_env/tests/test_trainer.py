import pytest

torch = pytest.importorskip("torch")

from swarmrl_env.swarm_env import SwarmEnv
from swarmrl_env.policies import MAPPOTrainer


def test_mappo_trainer_rejects_invalid_update_epochs():
    env = SwarmEnv(n_agents=3, n_neighbors=2)

    with pytest.raises(ValueError, match="update_epochs must be positive"):
        MAPPOTrainer(env, update_epochs=0)


def test_mappo_trainer_collects_and_updates_a_real_environment():
    env = SwarmEnv(
        n_agents=3,
        n_neighbors=2,
        world_size=20.0,
        max_episode_steps=10,
        max_speed=2.0,
    )
    trainer = MAPPOTrainer(
        env,
        rollout_length=4,
        hidden_dim=16,
        update_epochs=2,
    )

    metrics = trainer.train_iteration(seed=7)

    assert metrics["rollout_steps"] == 4.0
    assert torch.isfinite(torch.tensor(metrics["actor_loss"]))
    assert torch.isfinite(torch.tensor(metrics["critic_loss"]))