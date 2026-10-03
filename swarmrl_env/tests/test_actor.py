import numpy as np
import pytest

torch = pytest.importorskip("torch")

from swarmrl_env.policies import ActorPolicy


def test_actor_samples_bounded_actions_and_maps_controls():
    policy = ActorPolicy(observation_dim=17)
    observations = torch.zeros((4, 17))

    actions, log_probability = policy(observations)
    controls = policy.map_action_to_controls(actions, max_speed=2.0)

    assert actions.shape == (4, 3)
    assert log_probability.shape == (4,)
    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)
    assert torch.all(controls.throttle >= 0.0)
    assert torch.all(controls.throttle <= 2.0)
    assert torch.all(controls.pitch.abs() <= np.pi / 2.0)
    assert torch.all(controls.yaw.abs() <= np.pi)


def test_actor_evaluates_sampled_actions_for_ppo():
    policy = ActorPolicy(observation_dim=17)
    observations = torch.randn((4, 17))
    actions, old_log_probability = policy(observations)

    log_probability, entropy = policy.evaluate_actions(observations, actions)

    assert log_probability.shape == old_log_probability.shape
    assert entropy.shape == old_log_probability.shape
    assert torch.isfinite(log_probability).all()
    assert torch.isfinite(entropy).all()