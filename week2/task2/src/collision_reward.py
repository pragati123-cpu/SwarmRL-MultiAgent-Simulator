from math import sqrt


COLLISION_PENALTY = -100

SPACE_BOUNDS = {
    "x": (0.0, 100.0),
    "y": (0.0, 100.0),
    "z": (0.0, 100.0),
}

COLLISION_DISTANCE = 2.0


def calculate_distance(position_a, position_b):
    """Calculate Euclidean distance between two 3D positions."""

    dx = position_a[0] - position_b[0]
    dy = position_a[1] - position_b[1]
    dz = position_a[2] - position_b[2]

    return sqrt(dx**2 + dy**2 + dz**2)

def is_outside_boundary(position):
    """Check whether a drone position is outside the 3D environment."""

    x, y, z = position

    return not (
        SPACE_BOUNDS["x"][0] <= x <= SPACE_BOUNDS["x"][1]
        and SPACE_BOUNDS["y"][0] <= y <= SPACE_BOUNDS["y"][1]
        and SPACE_BOUNDS["z"][0] <= z <= SPACE_BOUNDS["z"][1]
    )


def has_drone_collision(position, other_positions):
    """Check whether a drone collides with another drone."""

    for other_position in other_positions:
        distance = calculate_distance(position, other_position)

        if distance <= COLLISION_DISTANCE:
            return True

    return False


def calculate_collision_penalty(position, other_positions):
    """
    Return -100 when a drone hits another drone
    or goes outside the environment boundary.
    """

    if is_outside_boundary(position):
        return COLLISION_PENALTY

    if has_drone_collision(position, other_positions):
        return COLLISION_PENALTY

    return 0