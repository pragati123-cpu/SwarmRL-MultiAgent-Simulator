# SwarmRL Multi-Agent Simulator

SwarmRL is a PettingZoo-based multi-drone environment with a Python simulation
backend and a React/Three.js frontend.

## Task 2: Observation Space and Physics

Each drone receives a normalized observation vector containing:

- Spatial position: normalized x, y, and z coordinates.
- Current velocity: normalized x, y, and z velocity.
- Distances to the nearest drones, sorted from closest to farthest.
- Normalized distances to the six world boundaries.

The number of nearest-drone distances is controlled by `n_neighbors`. The
observation space is a Gymnasium `Box` and is exposed per agent through the
PettingZoo `ParallelEnv` API.

Run the backend tests from the repository root:

```bash
PYTHONPATH=swarmrl_env python -m pytest -q swarmrl_env/tests
```

## Task 3: React & 3D Canvas Boilerplate

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
npm install