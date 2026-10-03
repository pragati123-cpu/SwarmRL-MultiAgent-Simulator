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


def test_dynamic_obstacles_reset_and_move_without_changing_observation_shape():
    env = SwarmEnv(
        num_drones=2,
        obstacle_count=2,
        obstacle_bounds=10.0,
        obstacle_max_speed=2.0,
        obstacle_dt=0.5,
    )

    observations, infos = env.reset(seed=42)
    initial_states = infos["drone_0"]["obstacle_states"]

    assert initial_states.shape == (2, 6)
    assert observations["drone_0"].shape == (9,)

    actions = {
        agent: np.zeros(4, dtype=np.float32)
        for agent in env.agents
    }
    _, _, _, _, step_infos = env.step(actions)

    updated_states = step_infos["drone_0"]["obstacle_states"]
    assert updated_states.shape == (2, 6)
    assert not np.array_equal(initial_states, updated_states)


def test_dynamic_obstacle_reset_is_seeded():
    env_a = SwarmEnv(num_drones=2, obstacle_count=1)
    env_b = SwarmEnv(num_drones=2, obstacle_count=1)

    _, infos_a = env_a.reset(seed=7)
    _, infos_b = env_b.reset(seed=7)

    np.testing.assert_array_equal(
        infos_a["drone_0"]["obstacle_states"],
        infos_b["drone_0"]["obstacle_states"],
    )