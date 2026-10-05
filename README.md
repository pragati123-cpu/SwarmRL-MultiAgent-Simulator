# Week 4 – Task 4: Ground Search & Coverage Texture Painting

## Overview

Implemented a ground search and coverage visualization system for the **SwarmRL Multi-Agent Simulator** using React and Three.js.

The system detects the area covered by drone sensor cones, tracks the visited ground cells, and visually highlights the covered areas in real time.

## Features

* Ground coverage grid visualization
* Sensor-based ground coverage detection
* Covered ground cell tracking
* Green coverage visualization
* Real-time coverage updates
* Coverage percentage calculation
* Coverage progress indicator
* Optimized state updates using JavaScript `Set`
* Polished coverage cell visualization

## Technology Used

* React
* Three.js
* React Three Fiber
* React Three Drei
* JavaScript
* WebSocket

## Implementation

### 1. Ground Coverage Grid

A configurable ground grid is created using the Drei `Grid` component.

Configuration:

* Ground Size: `100`
* Grid Divisions: `50`
* Total Cells: `2500`
* Cell Size: `2`

### 2. Sensor Coverage Detection

The `getGroundCoverage()` function calculates which ground cells are inside the drone sensor radius.

The sensor coverage radius is:

```text
SENSOR_RADIUS = 2
```

The function checks the distance between the sensor position and each ground-cell center.

Only cells within the sensor radius are considered covered.

### 3. Covered Cell Tracking

Covered cells are stored using a JavaScript `Set`.

Each cell is represented using:

```text
x,z
```

Example:

```text
2,5
3,5
2,6
```

Using a `Set` prevents duplicate cells from being stored.

### 4. Real-Time Coverage Update

Whenever a drone changes its position, the sensor coverage is recalculated.

Newly covered cells are added to the existing coverage state.

Already covered cells are ignored.

This allows the ground coverage to grow dynamically as drones move.

### 5. Coverage Visualization

Covered cells are rendered as green semi-transparent surfaces above the ground grid.

A small border effect is also added to improve visibility and make the covered region easier to identify.

### 6. Coverage Percentage

The application calculates the percentage of the ground that has been covered.

Formula:

```text
Coverage Percentage =
(Covered Cells / Total Cells) × 100
```

The value is displayed in the simulator HUD.

Example:

```text
Covered Cells: 125
Ground Coverage: 5.00%
```

A progress bar is also displayed for quick visual feedback.

## Project Files

Important files added/updated for this task:

```text
frontend/
└── src/
    ├── App.jsx
    ├── GroundCoverage.jsx
    └── sensorCoverage.js
```

### `App.jsx`

Responsible for:

* Receiving drone telemetry
* Managing covered-cell state
* Calculating coverage percentage
* Updating coverage in real time
* Displaying the simulator HUD

### `GroundCoverage.jsx`

Responsible for:

* Rendering the ground grid
* Rendering covered ground cells
* Displaying coverage visualization

### `sensorCoverage.js`

Responsible for:

* Calculating sensor-covered ground cells
* Checking distance between sensor and ground cells

## Git Commits

The implementation was completed through the following commits:

1. `Add ground coverage grid`
2. `Detect sensor ground coverage`
3. `Track covered ground cells`
4. `Paint covered ground areas green`
5. `Update coverage in real time`
6. `Optimize ground coverage updates`
7. `Polish ground coverage visualization`

## Testing

Frontend production build was tested using:

```bash
npm run build
```

The application can be started using:

```bash
npm run dev
```

The WebSocket backend runs on:

```text
ws://localhost:8000/ws/drones
```

## Result

The simulator now provides a real-time visual representation of the ground area searched by the drone swarm.

As drones move, their sensor coverage is detected and the corresponding ground cells are highlighted. The HUD displays the number of covered cells and the overall ground coverage percentage.

## Task Status

**Week 4 – Task 4: Completed ✅**
