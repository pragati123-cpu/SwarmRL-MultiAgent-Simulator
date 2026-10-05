# SwarmRL Visualizer

The React Three Fiber dashboard renders live drone telemetry from the backend
WebSocket at `ws://localhost:8000/ws/drones`.

## Run the dashboard

```bash
npm install
npm run dev
```

Build and lint checks are available through `npm run build` and `npm run lint`.

## Sensor coverage rendering

`src/SensorCone.jsx` draws an open, double-sided cone from each drone to the
ground and a ring at the projected coverage boundary. Both use low-opacity,
emissive standard materials with depth writing disabled. This keeps the grid or
terrain visible beneath overlapping sensor volumes while matching the drone's
color. Emissive material provides the glow without a post-processing bloom pass.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and Oxlint's TypeScript related rules in your project.
