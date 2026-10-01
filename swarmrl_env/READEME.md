# SwarmRL Backend: Environment and MAPPO

This package contains the PettingZoo parallel environment and the PyTorch
MAPPO policy stack used to train it. The environment models multi-drone
exploration, separation, collision avoidance, and bounded 3D flight.

## Setup

From the repository root:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r swarmrl_env/requirements.txt
```

Run the random environment demo with:

```bash
PYTHONPATH=swarmrl_env .venv/Scripts/python.exe swarmrl_env/demo.py
```

Run environment tests with:

```bash
PYTHONPATH=swarmrl_env .venv/Scripts/python.exe -m pytest -q swarmrl_env/tests/test_env.py
```

Run the policy and integration tests with:

```bash
.venv/Scripts/python.exe -m pytest -q swarmrl_env/tests/test_actor.py swarmrl_env/tests/test_critic.py swarmrl_env/tests/test_training.py swarmrl_env/tests/test_trainer.py
```

## Package layout

```text
swarmrl_env/
├── requirements.txt
├── demo.py
├── policies/
│   ├── actor.py       # local Gaussian actor and action mapping
│   ├── critic.py      # centralized value function
│   ├── rollout.py     # trajectory storage and GAE
│   ├── updater.py     # clipped PPO update
│   └── trainer.py     # environment rollout/update integration
├── swarm_env/
│   ├── __init__.py
│   └── environment.py
└── tests/
    ├── test_env.py
    ├── test_actor.py
    ├── test_critic.py
    ├── test_training.py
    └── test_trainer.py
```

## Environment contract

`SwarmEnv` implements the PettingZoo `ParallelEnv` API:

```python
from swarmrl_env.swarm_env import SwarmEnv

env = SwarmEnv(
    n_agents=50,
    n_neighbors=5,
    world_size=100.0,
    max_episode_steps=500,
    collision_radius=1.5,
    max_speed=2.0,
    grid_cell_size=5.0,
)

observations, infos = env.reset(seed=0)
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
observations, rewards, terminations, truncations, infos = env.step(actions)
```

For `n_neighbors=k`, each local observation has shape `(12 + k,)`:

| Segment | Size | Contents |
| --- | ---: | --- |
| Position | 3 | Position normalized by half the world size |
| Velocity | 3 | Velocity normalized by `max_speed` |
| Neighbors | `k` | Sorted nearest-neighbor distances |
| Walls | 6 | Normalized distances to the six world boundaries |

Each action is a `Box(-1, 1, shape=(3,))`:

| Index | Command | Physical mapping |
| ---: | --- | --- |
| 0 | Throttle | `[0, max_speed]` |
| 1 | Pitch | `[-pi/2, pi/2]` |
| 2 | Yaw | `[-pi, pi]` |

The environment converts the commands to a 3D velocity, integrates position
for one step, and clips the position to the cubic world bounds. Rewards report
coverage novelty, separation, and collision penalty components in `infos`.

## MAPPO policy stack

`ActorPolicy` is shared across drones. It consumes one local observation per
row and returns a tanh-bounded action plus its PPO log probability. Its
`map_action_to_controls()` helper exposes throttle, pitch, and yaw in physical
units.

`CentralizedCritic` consumes the full swarm observation tensor with shape
`(..., n_agents, observation_dim)` and returns one shared value estimate per
leading batch item.

`RolloutBuffer` stores synchronized per-agent transitions, computes generalized
advantage estimates, and produces flattened `RolloutBatch` objects. The
`MAPPOUpdater` applies the clipped PPO objective to the shared actor and value
critic. `MAPPOTrainer` connects these components to `SwarmEnv`:

```python
from swarmrl_env.policies import MAPPOTrainer

trainer = MAPPOTrainer(env, rollout_length=128)
metrics = trainer.train_iteration(seed=0)
```

`train_iteration()` collects one rollout, computes returns and advantages, runs
one optimizer update, and returns actor loss, critic loss, entropy, approximate
KL, clip fraction, and rollout step metrics.

## Reward and episode behavior

- Coverage gives a positive reward the first time an agent enters a grid cell.
- Separation increases with distance to the nearest other agent up to
  `target_separation`.
- Collisions apply `collision_penalty` to each involved agent.
- Episodes truncate at `max_episode_steps`; positions are clipped rather than
  treated as fatal out-of-bounds terminations.
- After an episode ends, `step({})` returns empty dictionaries as required by
  the PettingZoo parallel API.