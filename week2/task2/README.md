# Week 2 - Task 2: Collision Penalty and Exploration Reward

## Objective

Implement the collision penalty and coverage-based exploration reward used by
the current SwarmRL multi-agent environment.

## Exploration Reward

The PettingZoo environment tracks visited 3D grid cells in
`SwarmEnv.visited_cells`. A drone receives `+1` when its current cell has not
been visited by any drone before; the cell is then added to the shared set.
Revisiting a known cell returns `0` exploration reward. Spawn cells are marked
visited during `reset()`.

The grid resolution is configured with `grid_cell_size`, and the running
unique-cell count is available as `infos[agent]["coverage_cells"]`.

## Collision Conditions

A drone receives a penalty of **-100** when:

1. It collides with another drone.
2. It moves outside the 3D environment boundary.

## Configuration

- Collision penalty: `-100`
- Collision distance threshold: `2.0`
- X boundary: `0 to 100`
- Y boundary: `0 to 100`
- Z boundary: `0 to 100`

## Implementation

The `collision_reward.py` module provides:

- 3D Euclidean distance calculation
- Drone-to-drone collision detection
- Environment boundary detection
- Collision penalty calculation

## Reward Logic

```text
If drone is outside boundary:
    Reward = -100

Else if drone collides with another drone:
    Reward = -100

Else:
    Reward = 0
```