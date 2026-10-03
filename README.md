# SwarmRL Multi-Agent Simulator

SwarmRL is a PettingZoo parallel environment for multi-drone exploration and
collision avoidance. It includes a PyTorch MAPPO implementation that maps
local drone observations to continuous flight controls and uses a centralized
critic for swarm-level value estimation.

## Backend setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
.venv\Scripts\python.exe -m pip install -r swarmrl_env/requirements.txt
```

Run the backend tests from the repository root:

```bash
PYTHONPATH=swarmrl_env .venv/Scripts/python.exe -m pytest -q swarmrl_env/tests
```

The policy and action-space checks can also be run together:

```bash
.venv/Scripts/python.exe -m pytest -q swarmrl_env/tests/test_actor.py swarmrl_env/tests/test_critic.py swarmrl_env/tests/test_training.py swarmrl_env/tests/test_trainer.py tests/test_action_space.py
```

## MAPPO components

The policy package is under `swarmrl_env/policies/`:

- `ActorPolicy`: shared tanh-squashed Gaussian actor with PPO log probabilities.
- `CentralizedCritic`: value function over all agents' observations.
- `RolloutBuffer`: parallel trajectory storage and GAE return calculation.
- `MAPPOUpdater`: clipped PPO actor/critic update.
- `MAPPOTrainer`: real-environment rollout collection and one update iteration.

```python
from swarmrl_env.swarm_env import SwarmEnv
from swarmrl_env.policies import MAPPOTrainer

env = SwarmEnv(n_agents=6, n_neighbors=3)
trainer = MAPPOTrainer(env, rollout_length=128)
metrics = trainer.train_iteration(seed=0)
```

## Environment contract

For `n_neighbors=k`, each local observation has shape `(12 + k,)` and contains
normalized position, normalized velocity, sorted neighbor distances, and six
normalized wall distances.

Each action is a three-value `Box(-1, 1)`:

| Index | Command | Physical range |
| --- | --- | --- |
| 0 | throttle | `[0, max_speed]` |
| 1 | pitch | `[-pi/2, pi/2]` |
| 2 | yaw | `[-pi, pi]` |

The environment converts these commands into a 3D velocity, clips positions to
the world bounds, and reports coverage, separation, and collision rewards.

## Frontend

The React/Three.js dashboard is developed independently:

```bash
cd frontend
npm install
npm run dev
```

Each drone displays a semi-transparent, emissive sensor cone and a ground
coverage ring. Their low-opacity materials keep the scene beneath them visible,
while emissive color follows the drone's altitude color. The current scene uses
a ground grid; the same materials also allow terrain meshes to remain visible
through the coverage visualization.