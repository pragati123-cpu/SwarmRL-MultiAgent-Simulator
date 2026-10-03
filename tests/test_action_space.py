import numpy as np

from src.env.swarm_env import SwarmEnv


def test_action_space_shape_and_bounds():
    env = SwarmEnv(num_drones=4)

    for agent in env.possible_agents:
        space = env.action_space(agent)

        # Action must contain:
        # [velocity, pitch, yaw, roll]
        assert space.shape == (4,)

        # Lower bounds
        np.testing.assert_array_equal(
            space.low,
            np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32)
        )

        # Upper bounds
        np.testing.assert_array_equal(
            space.high,
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        )


def test_random_actions_are_valid():
    env = SwarmEnv(num_drones=4)

    for agent in env.possible_agents:
        space = env.action_space(agent)

        for _ in range(10):
            action = space.sample()

            assert action.shape == (4,)
            assert action.dtype == np.float32
            assert space.contains(action)


def test_valid_drone_action():
    env = SwarmEnv(num_drones=4)

    action = np.array(
        [0.8, -0.2, 0.4, 0.1],
        dtype=np.float32
    )

    for agent in env.possible_agents:
        assert env.action_space(agent).contains(action)


def test_invalid_drone_actions():
    env = SwarmEnv(num_drones=4)

    space = env.action_space("drone_0")

    # Velocity > 1
    invalid_velocity = np.array(
        [1.5, 0.0, 0.0, 0.0],
        dtype=np.float32
    )

    # Pitch < -1
    invalid_pitch = np.array(
        [0.5, -1.5, 0.0, 0.0],
        dtype=np.float32
    )

    # Wrong number of action values
    wrong_shape = np.array(
        [0.5, 0.0, 0.0],
        dtype=np.float32
    )

    assert not space.contains(invalid_velocity)
    assert not space.contains(invalid_pitch)
    assert not space.contains(wrong_shape) 

def test_number_of_drones():
    env = SwarmEnv(num_drones=6)

    assert len(env.possible_agents) == 6
    assert env.possible_agents == [
        "drone_0",
        "drone_1",
        "drone_2",
        "drone_3",
        "drone_4",
        "drone_5",
    ]

def test_action_boundary_values():
    env = SwarmEnv(num_drones=4)

    space = env.action_space("drone_0")

    minimum_action = np.array(
        [0.0, -1.0, -1.0, -1.0],
        dtype=np.float32
    )

    maximum_action = np.array(
        [1.0, 1.0, 1.0, 1.0],
        dtype=np.float32
    )

    assert space.contains(minimum_action)
    assert space.contains(maximum_action)