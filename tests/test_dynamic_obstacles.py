import numpy as np
import pytest

from src.env.dynamic_obstacles import DynamicObstacle, DynamicObstacleManager


def test_obstacle_update_moves_position_and_preserves_state_shape():
    obstacle = DynamicObstacle(
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([1.0, -2.0, 0.5]),
        radius=0.5,
    )

    obstacle.update(dt=2.0, bounds=10.0)

    np.testing.assert_allclose(obstacle.position, [2.0, -4.0, 1.0])
    assert obstacle.as_array().shape == (6,)


def test_obstacle_reflects_at_world_bounds():
    obstacle = DynamicObstacle(
        position=np.array([8.0, 0.0, 0.0]),
        velocity=np.array([3.0, 0.0, 0.0]),
        radius=1.0,
    )

    obstacle.update(dt=1.0, bounds=10.0)

    np.testing.assert_allclose(obstacle.position, [7.0, 0.0, 0.0])
    np.testing.assert_allclose(obstacle.velocity, [-3.0, 0.0, 0.0])


def test_manager_reset_is_deterministic_and_step_changes_positions():
    manager_a = DynamicObstacleManager(
        count=3,
        bounds=10.0,
        radius=1.0,
        max_speed=2.0,
    )
    manager_b = DynamicObstacleManager(
        count=3,
        bounds=10.0,
        radius=1.0,
        max_speed=2.0,
    )

    initial_a = manager_a.reset(seed=42)
    initial_b = manager_b.reset(seed=42)

    np.testing.assert_array_equal(initial_a, initial_b)
    updated = manager_a.step(dt=0.5)

    assert updated.shape == (3, 6)
    assert not np.array_equal(initial_a, updated)
    assert np.all(np.abs(updated[:, :3]) <= 9.0)


def test_manager_validates_configuration():
    with pytest.raises(ValueError, match="count"):
        DynamicObstacleManager(count=-1, bounds=10.0)

    with pytest.raises(ValueError, match="radius"):
        DynamicObstacleManager(count=1, bounds=1.0, radius=1.0)
