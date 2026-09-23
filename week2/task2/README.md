# Week 2 - Task 2: Collision Penalty

## Objective

Implement a severe collision penalty for drones in the SwarmRL multi-agent simulator.

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