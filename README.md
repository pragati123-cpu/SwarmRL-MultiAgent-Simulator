# SwarmRL Week 3 - MAPPO Training Pipeline

## Task

Integrate the custom PettingZoo multi-agent drone environment
with Ray RLlib for MAPPO-style multi-agent training.

## Architecture

```text
Custom PettingZoo Parallel Environment
                |
                v
          PettingZooEnv
                |
                v
          Ray RLlib PPO
                |
                v
       Shared Drone Policy
                |
                v
       Parallel EnvRunners
                |
                v
        Multi-Agent Training

## Dynamic obstacles

The PettingZoo environment supports optional moving spherical obstacles. Set
`obstacle_count` above zero to enable them; each obstacle has a seeded initial
position, a velocity, and reflecting world boundaries. When enabled, each
drone observation includes one normalized distance per obstacle.

Obstacle configuration is also available through `build_mappo_config`:

```python
config = build_mappo_config(
    num_drones=3,
    obstacle_count=4,
    obstacle_bounds=10.0,
    obstacle_radius=1.0,
    obstacle_max_speed=0.5,
    obstacle_dt=1.0,
    drone_radius=0.5,
    obstacle_collision_penalty=10.0,
)
```

Drone-obstacle intersections are reported in `info["obstacle_collision"]` and
subtract `obstacle_collision_penalty` from the action reward. `env.render()`
returns a copy-safe state snapshot containing drone positions, obstacle
positions, and obstacle radii for visualization clients.