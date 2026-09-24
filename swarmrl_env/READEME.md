# SwarmRL - Environment Reset, Step, and Reward Logic

**Project:** SwarmRL — Multi-Agent Deep Reinforcement Learning Simulator
**Track:** Reinforcement Learning (Ray RLlib, PyTorch)
**This deliverable's scope:** `Environment Reset & Step Logic (Python/PettingZoo)` —
standard `reset()` and `step()` handlers for the multi-agent environment loop,
including observation, collision, and exploration-reward behavior.

This repo implements the piece of Week 1 ("Environment Physics") assigned to
this task: a working `reset()`/`step()` loop for the custom PettingZoo
environment, including the continuous action space (throttle / pitch / yaw),
the sorted distances to nearest neighbors observation, boundary clipping, collision
detection, and the coverage-based reward. It does **not** cover the
Three.js/WebSocket rendering track (that's the parallel "Simulation &
Rendering" column for Week 1) or the RLlib training loop (Week 3) — those are
separate teammates' deliverables that will import this environment.

---

## Folder structure

```
swarmrl_env/
├── READEME.md
├── requirements.txt
├── demo.py                    # random-action rollout smoke test
├── swarm_env/
│   ├── __init__.py            # exposes SwarmEnv + parallel_env()/env() factory
│   └── environment.py         # SwarmEnv: reset(), step(), obs/reward/collision logic
└── tests/
    └── test_env.py            # pytest suite for reset()/step() correctness
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## Try it

```bash
python demo.py              # random rollout, prints per-step stats
python -m pytest tests/ -v  # correctness tests (10 tests)
```

The test suite covers reset determinism, bounded positions, observation-space
validity, collision penalties, exploration rewards, episode truncation, and
safe stepping after an episode ends.

---

## What `SwarmEnv` implements

`swarm_env/environment.py` defines `SwarmEnv(ParallelEnv)`, a PettingZoo
**Parallel API** environment (all agents act simultaneously each step — this
is the API RLlib's multi-agent trainers expect, and what the Ray RLlib track
will consume in Week 3).

### Action space - per agent, `Box(low=-1, high=1, shape=(3,))`

| Index | Meaning       | Mapped range              |
|-------|---------------|----------------------------|
| 0     | throttle      | `[0, max_speed]`           |
| 1     | pitch command | `[-π/2, π/2]`               |
| 2     | yaw command   | `[-π, π]`                   |

`step()` converts `(throttle, pitch, yaw)` into a 3D velocity vector via the
standard spherical→cartesian conversion, then integrates position by one
step and clips it to the world's bounding box.

### Observation space - per agent, `Box(shape=(6 + k + 6,))`

For `n_neighbors=k` (default 5):

| Segment                    | Size | Contents                                         |
|-----------------------------|------|---------------------------------------------------|
| Own position (normalized)  | 3    | position / half world-size                        |
| Own velocity (normalized)  | 3    | velocity / max_speed                               |
| Nearest-neighbor distances | k    | sorted distances to the k closest other agents    |
| Wall distances             | 6    | normalized distance to each of the 6 bounding walls|

Nearest neighbors are recomputed each step by brute-force pairwise distance
(fine for the target swarm size of ~50 agents; easy to swap for a KD-tree
later if the swarm size grows).

### Reward

- `+1` the **first** time an agent enters a previously-unvisited cell of a
  coarse coverage grid (`grid_cell_size`, default 5.0 world units) —
  implements the brief's "reward agents for exploring unvisited
  coordinates (+1)".
- `-100` for every step an agent is within `collision_radius` of another
  agent — implements "heavily penalize them for colliding with each other
  (-100)".

### Episode termination

- `truncations[agent] = True` for all agents once `max_episode_steps` is
  reached (time-limit truncation, not a "failure").
- `terminations` is currently always `False` — positions are hard-clipped
  to the world bounds rather than treated as a fatal out-of-bounds event.
  This is intentionally left as a hook: a "fly out of bounds = terminate"
  rule can be added here later (e.g. for the Week 4 curriculum-learning
  step) without touching the rest of the loop.
- Per PettingZoo convention, `self.agents` is cleared once the episode
  ends, and `step()` is a safe no-op (returns empty dicts) if called again
  after that.

### Reset behavior (`reset(seed=None, options=None)`)

- Re-seeds the environment's RNG if `seed` is given (fully deterministic
  given the same seed — covered by a test).
- Spawns all agents at random positions within a configurable central
  fraction of the world (`options={"spawn_fraction": 0.5}` by default) so
  the swarm starts clustered rather than at the walls.
- Zeroes all velocities.
- Marks each agent's spawn cell as already "visited" so the first step
  doesn't hand out a free exploration reward just for existing.
- Returns `(observations, infos)` exactly matching the PettingZoo Parallel
  API contract.

---

## Interface contract for teammates

This is the surface other tracks should build against:

```python
from swarm_env import SwarmEnv

env = SwarmEnv(
    n_agents=50,          # swarm size
    n_neighbors=5,         # observed nearest neighbors
    world_size=100.0,      # cubic world side length
    max_episode_steps=500,
    collision_radius=1.5,
    max_speed=2.0,
    grid_cell_size=5.0,
)

observations, infos = env.reset(seed=0)
# observations: dict[agent_id -> np.ndarray shape=(obs_dim,)]

actions = {agent: env.action_space(agent).sample() for agent in env.agents}
observations, rewards, terminations, truncations, infos = env.step(actions)
# rewards: dict[agent_id -> float]
# infos[agent_id]["collided"]: bool
# infos[agent_id]["coverage_cells"]: int (running total unique cells visited)
```

- **RLlib / MAPPO track (Week 3):** wrap with
  `ray.rllib.env.wrappers.pettingzoo_env.ParallelPettingZooEnv(SwarmEnv(...))`
  or register via `pettingzoo.utils.conversions` as needed — the
  `reset`/`step` signatures here already match what those wrappers expect.
- **WebSocket / Three.js dashboard track:** the per-step source of truth
  for drone positions is `env.positions` (`dict[agent_id -> np.ndarray(3,)]`),
  and `infos[agent]["coverage_cells"]` / `["collided"]` are the two live
  metrics called out in the Week 1 brief ("push the exact X, Y, Z
  coordinates of all agents in the environment every step").

## Notes / things intentionally left for later weeks

- No dynamic obstacles or wind resistance yet (Week 4: curriculum learning).
- No centralized-critic / MAPPO logic here — this environment only defines
  the world; the learning algorithm is a separate track.
- Nearest-neighbor lookup is brute-force `O(n²)`; acceptable at `n≈50` but
  flagged here in case the swarm size grows significantly.
