## Task 4: Drone Sensor Cone

Each drone includes a 3D sensor/LiDAR vision cone built using Three.js `ConeGeometry`.

The sensor cone is attached to each drone so that it follows the drone's position in the 3D scene.

### Implementation

- React Three Fiber
- Three.js `ConeGeometry`
- Reusable `SensorCone` component
- Configurable radius, height, and segments
- Drone-relative sensor positioning
- Downward-facing sensor cone
- Sensor cone rendered for every drone

### Frontend Files

- `frontend/src/App.jsx` - Renders drones and attaches the sensor cone.
- `frontend/src/SensorCone.jsx` - Reusable sensor cone component.

### Run Locally

```bash
cd frontend
npm install
npm run dev