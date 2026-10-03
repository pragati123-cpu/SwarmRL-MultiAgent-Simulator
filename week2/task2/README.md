# Week 2 - Task 2: Collision Penalty and Exploration Reward

## Objective

This folder contains a standalone collision/boundary penalty helper. It is not
called by the integrated PettingZoo environment, whose combined coverage,
separation, and collision reward is implemented in
`swarmrl_env/swarm_env/environment.py`.

The integrated environment awards `+1.0` for a newly visited cell, up to
`+0.5` for separation from the nearest drone, and applies `-10.0` for a
collision. Its reward weights and target separation are configurable; see
`swarmrl_env/READEME.md` for the full formula and per-agent reward breakdown.

## Exploration Reward

The PettingZoo environment tracks visited 3D grid cells in
`SwarmEnv.visited_cells`. A drone receives `+1` when its current cell has not
been visited by any drone before; the cell is then added to the shared set.
Revisiting a known cell returns `0` exploration reward. Spawn cells are marked
visited during `reset()`.

The grid resolution is configured with `grid_cell_size`, and the running
unique-cell count is available as `infos[agent]["coverage_cells"]`.

## Collision Conditions

A caller of the standalone helper receives a penalty of **-100** when:

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