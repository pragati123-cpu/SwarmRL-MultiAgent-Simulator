# SwarmRL MAPPO Training Pipeline

SwarmRL integrates a custom PettingZoo parallel drone environment with Ray
RLlib PPO for shared-policy, MAPPO-style multi-agent training.

## Architecture

```text
Custom PettingZoo Parallel Environment
                |
                v
          PettingZooEnv
                |
                v
          Ray RLlib PPO (Torch)
                |
                v
       Shared Drone Policy
                |
                v
       Parallel EnvRunners
                |
                v
        Multi-Agent Training
```

## Setup

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the focused environment tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_mappo_pipeline.py tests\test_dynamic_obstacles.py -q
```

## Dynamic obstacles

Moving spherical obstacles are optional. The default `obstacle_count=0`
preserves the original nine-value drone observation. Set `obstacle_count` to a
positive value to enable dynamic obstacles.

```python
from src.env.swarm_env import SwarmEnv

env = SwarmEnv(
    num_drones=3,
    max_cycles=100,
    obstacle_count=4,
    obstacle_bounds=10.0,
    obstacle_radius=1.0,
    obstacle_max_speed=0.5,
    obstacle_dt=1.0,
    drone_radius=0.5,
    obstacle_collision_penalty=10.0,
)
observations, infos = env.reset(seed=42)
```

Each obstacle has a three-dimensional position, velocity, and radius. Initial
state is deterministic for a supplied reset seed. Obstacles reflect from the
world boundary rather than leaving the simulation volume.

### Observation contract

The base observation contains nine values:

| Segment | Size | Contents |
| --- | ---: | --- |
| Position | 3 | Drone position |
| Velocity | 3 | Commanded velocity |
| Orientation | 3 | Roll, pitch, and yaw |

When obstacles are enabled, one additional normalized distance is appended for
each configured obstacle. Therefore, the observation shape is
`(9 + obstacle_count,)`. The corresponding observation space is updated
automatically.

### Rewards and info

The base action reward is:

```text
1.0 - mean(action ** 2)
```

If the drone sphere intersects an obstacle sphere, the configured
`obstacle_collision_penalty` is subtracted. Each agent receives:

- `info["obstacle_collision"]`: whether it collided this step
- `info["obstacle_states"]`: a copy of the `(obstacle_count, 6)` position and
  velocity array
- `info["reward_components"]`: action and obstacle-collision contributions

### Render snapshot

`env.render()` returns a copy-safe state snapshot for visualization clients:

```python
snapshot = env.render()
# snapshot["step"]
# snapshot["drones"]          # agent -> position
# snapshot["obstacles"]       # shape (obstacle_count, 3)
# snapshot["obstacle_radii"]  # shape (obstacle_count,)
```

## MAPPO configuration

Obstacle settings can be passed through `build_mappo_config`; they are
forwarded to every RLlib environment runner:

```python
from src.training.mappo_config import build_mappo_config

config = build_mappo_config(
    num_drones=3,
    num_env_runners=2,
    obstacle_count=4,
    obstacle_bounds=10.0,
    obstacle_radius=1.0,
    obstacle_max_speed=0.5,
    obstacle_dt=1.0,
    drone_radius=0.5,
    obstacle_collision_penalty=10.0,
)
```

The RLlib policy remains shared across drones. Enabling obstacles changes the
policy input space, so checkpoints trained with different obstacle counts
should not be mixed without retraining or an explicit observation adapter.

## Environment modules

- `src/env/swarm_env.py`: root RLlib/PettingZoo environment used by the MAPPO
  pipeline and dynamic-obstacle implementation.
- `src/env/dynamic_obstacles.py`: reusable obstacle and obstacle-manager
  primitives.
- `swarmrl_env/swarm_env/environment.py`: separate legacy PyTorch training
  environment with its own observation and reward contract, documented in
  [`swarmrl_env/READEME.md`](swarmrl_env/READEME.md).
