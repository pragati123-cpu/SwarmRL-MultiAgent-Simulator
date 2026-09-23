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