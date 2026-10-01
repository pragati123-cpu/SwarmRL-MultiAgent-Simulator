from week2.task2.src.collision_reward import (
    calculate_collision_penalty,
    calculate_distance,
    is_outside_boundary,
)


def test_distance_calculation():
    assert calculate_distance(
        (0, 0, 0),
        (3, 4, 0)
    ) == 5


def test_drone_collision_penalty():
    position = (50, 50, 50)

    other_positions = [
        (51, 50, 50)
    ]

    assert calculate_collision_penalty(
        position,
        other_positions
    ) == -100


def test_boundary_collision_penalty():
    position = (101, 50, 50)

    assert calculate_collision_penalty(
        position,
        []
    ) == -100


def test_no_collision():
    position = (50, 50, 50)

    other_positions = [
        (20, 20, 20)
    ]

    assert calculate_collision_penalty(
        position,
        other_positions
    ) == 0


def test_boundary_inside():
    position = (50, 50, 50)

    assert is_outside_boundary(position) is False
    
def test_boundary_edge_position_is_valid():
    position = (100, 100, 100)

    assert is_outside_boundary(position) is False