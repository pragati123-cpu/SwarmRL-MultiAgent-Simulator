# SwarmRL Multi-Agent Simulator

SwarmRL is a PettingZoo-based multi-drone environment with a Python simulation
backend and a React/Three.js frontend.

The repository contains both backend simulation code and a separate frontend
application, so each can be developed and tested independently.

## Backend setup

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r swarmrl_env/requirements.txt
```

Run the backend tests:

```bash
PYTHONPATH=swarmrl_env python -m pytest -q swarmrl_env/tests
```

Run the random-action backend demo:

```bash
cd swarmrl_env
python demo.py
```

## Task 2: Observation Space and Physics

Each drone receives a normalized observation vector containing:

- Spatial position: normalized x, y, and z coordinates.
- Current velocity: normalized x, y, and z velocity.
- Distances to the nearest drones, sorted from closest to farthest.
- Normalized distances to the six world boundaries.

The number of nearest-drone distances is controlled by `n_neighbors`. The
observation space is a Gymnasium `Box` and is exposed per agent through the
PettingZoo `ParallelEnv` API.

The reward balances coverage, safe spacing, and collision avoidance. A drone
receives `+1` the first time it enters a previously unvisited 3D grid cell
(spawn cells are already marked visited), plus a separation score from `0` to
`0.5` based on its distance to the nearest other drone. That score reaches its
maximum at `target_separation` (default: twice `collision_radius`). A collision
adds `-10`, which outweighs the maximum positive reward of `+1.5`. The weights
and collision penalty can be configured on `SwarmEnv`. Per-agent info includes
the `coverage`, `separation`, and `collision` reward components, and
`infos[agent]["coverage_cells"]` reports the running number of unique cells.

Run the backend tests from the repository root:

```bash
PYTHONPATH=swarmrl_env python -m pytest -q swarmrl_env/tests
```

## Frontend: React & 3D Canvas

This task sets up the React frontend and provides the basic 3D viewport using Three.js and React Three Fiber.

## Technologies

- React
- Vite
- Three.js
- @react-three/fiber

## Implemented

- React frontend setup using Vite
- Three.js integration
- React Three Fiber Canvas setup
- Basic 3D camera configuration
- Lighting setup
- Ready for future 3D terrain and drone visualization

## Run Locally

Install dependencies:

```bash
cd frontend
npm install
npm run dev
```