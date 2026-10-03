import numpy as np

from src.env.swarm_env import SwarmEnv


def test_environment_reset():
    env = SwarmEnv(num_drones=3)

    observations, infos = env.reset(seed=42)

    assert len(observations) == 3
    assert len(infos) == 3

    for agent in env.possible_agents:
        assert observations[agent].shape == (9,)


def test_environment_step():
    env = SwarmEnv(num_drones=3)

    observations, infos = env.reset(seed=42)

    actions = {
        agent: env.action_space(agent).sample()
        for agent in env.agents
    }

    (
        observations,
        rewards,
        terminations,
        truncations,
        infos,
    ) = env.step(actions)

    assert len(rewards) == 3

    for agent in env.possible_agents:
        assert isinstance(rewards[agent], float)


def test_action_space():
    env = SwarmEnv(num_drones=3)

    for agent in env.possible_agents:
        action = env.action_space(agent).sample()

        assert action.shape == (4,)
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)