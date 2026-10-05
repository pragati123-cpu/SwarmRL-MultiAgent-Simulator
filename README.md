# SwarmRL - MAPPO Drone Simulation

PettingZoo parallel environment for multi-drone simulation, with Ray RLlib PPO
training using a shared PyTorch policy.

## Environment Physics

Each drone observes position, velocity, orientation, and optionally normalized
distances to configured moving obstacles. Actions control forward velocity,
pitch, yaw, and roll. Wind and turbulence perturb the commanded velocity during
each physics step:

```text
wind_acceleration = air_resistance * (wind_velocity - current_velocity)
turbulence_acceleration ~ Normal(0, turbulence_strength^2 * I)
actual_velocity = commanded_velocity
                          + (wind_acceleration + turbulence_acceleration) * physics_dt
position += actual_velocity * physics_dt
```

The defaults enable a mild wind from the positive X direction:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `wind_velocity` | `(0.15, 0.0, 0.0)` | Ambient wind velocity on the X, Y, and Z axes |
| `air_resistance` | `0.2` | Drag response to velocity relative to the wind |
| `turbulence_strength` | `0.05` | Standard deviation of per-axis random acceleration |
| `physics_dt` | `0.1` | Physics integration timestep |

Set `turbulence_strength=0.0` for deterministic motion without random gusts.
Calling `reset(seed=...)` makes the turbulence and obstacle sampling
repeatable. Each step's `info` includes `wind_acceleration` and
`turbulence_acceleration` for inspecting environmental effects; the observation
shape is unchanged.

## Configure Training

The environmental parameters can be passed to the MAPPO configuration builder
and are forwarded to each RLlib environment runner:

```python
from src.training.mappo_config import build_mappo_config

config = build_mappo_config(
      num_drones=3,
      wind_velocity=(0.3, 0.1, 0.0),
      air_resistance=0.25,
      turbulence_strength=0.08,
      physics_dt=0.1,
)
```

Run training with `python train_mappo.py`. Install dependencies first with
`pip install -r requirements.txt`.

## Architecture

```text
      PettingZoo Parallel Environment
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
