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
    assert observations["drone_0"].shape == (11,)

    actions = {
        agent: np.zeros(4, dtype=np.float32)
        for agent in env.agents
    }
    _, _, _, _, step_infos = env.step(actions)

    updated_states = step_infos["drone_0"]["obstacle_states"]
    assert updated_states.shape == (2, 6)
    assert not np.array_equal(initial_states, updated_states)


def test_obstacle_distance_is_in_observation_and_collision_is_penalized():
    env = SwarmEnv(
        num_drones=2,
        obstacle_count=1,
        obstacle_bounds=10.0,
        obstacle_radius=1.0,
        obstacle_max_speed=0.0,
        obstacle_collision_penalty=7.0,
    )
    env.reset(seed=3)
    agent = env.agents[0]
    env.state[agent][:3] = env.obstacle_states[0, :3]

    actions = {
        current_agent: np.zeros(4, dtype=np.float32)
        for current_agent in env.agents
    }
    observations, rewards, _, _, infos = env.step(actions)

    assert observations[agent].shape == (10,)
    assert observations[agent][-1] <= 0.2
    assert infos[agent]["obstacle_collision"] is True
    assert infos[agent]["reward_components"]["obstacle_collision"] == -7.0
    assert rewards[agent] <= -6.0


def test_render_returns_drone_and_obstacle_snapshot():
    env = SwarmEnv(num_drones=2, obstacle_count=2)
    env.reset(seed=4)

    snapshot = env.render()

    assert snapshot["step"] == 0
    assert set(snapshot["drones"]) == set(env.possible_agents)
    assert snapshot["obstacles"].shape == (2, 3)
    assert snapshot["obstacle_radii"].shape == (2,)
    snapshot["obstacles"][0, 0] = 999.0
    assert env.obstacle_states[0, 0] != 999.0


def test_dynamic_obstacle_reset_is_seeded():
    env_a = SwarmEnv(num_drones=2, obstacle_count=1)
    env_b = SwarmEnv(num_drones=2, obstacle_count=1)

    _, infos_a = env_a.reset(seed=7)
    _, infos_b = env_b.reset(seed=7)

    np.testing.assert_array_equal(
        infos_a["drone_0"]["obstacle_states"],
        infos_b["drone_0"]["obstacle_states"],
    )


def test_wind_pushes_drones_and_air_resistance_opposes_motion():
    env = SwarmEnv(
        num_drones=1,
        wind_velocity=(2.0, 0.0, 0.0),
        air_resistance=0.5,
        turbulence_strength=0.0,
        physics_dt=0.1,
    )
    env.reset(seed=8)

    actions = {"drone_0": np.zeros(4, dtype=np.float32)}
    observations, _, _, _, infos = env.step(actions)

    np.testing.assert_allclose(infos["drone_0"]["wind_acceleration"], [1, 0, 0])
    assert observations["drone_0"][3] > 0.0

    env.state["drone_0"][3:6] = [1.0, 0.0, 0.0]
    env.air_resistance = 1.0
    env.wind_velocity[:] = 0.0
    observations, _, _, _, infos = env.step(actions)

    np.testing.assert_allclose(infos["drone_0"]["wind_acceleration"], [-1, 0, 0])
    assert observations["drone_0"][3] < 0.0


def test_turbulence_is_repeatable_for_a_seed():
    env_a = SwarmEnv(num_drones=1, turbulence_strength=0.4)
    env_b = SwarmEnv(num_drones=1, turbulence_strength=0.4)
    env_a.reset(seed=29)
    env_b.reset(seed=29)
    actions = {"drone_0": np.zeros(4, dtype=np.float32)}

    observation_a, _, _, _, info_a = env_a.step(actions)
    observation_b, _, _, _, info_b = env_b.step(actions)

    np.testing.assert_array_equal(observation_a["drone_0"], observation_b["drone_0"])
    np.testing.assert_array_equal(
        info_a["drone_0"]["turbulence_acceleration"],
        info_b["drone_0"]["turbulence_acceleration"],
    )
    assert np.any(info_a["drone_0"]["turbulence_acceleration"] != 0.0)