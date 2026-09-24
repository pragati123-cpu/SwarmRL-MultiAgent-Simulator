import { useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";

const WS_URL = "ws://localhost:8000/ws/drones";

function Drone({ drone }) {
  return (
    <mesh position={[drone.x, drone.y, drone.z]}>
      <sphereGeometry args={[0.2, 16, 16]} />
      <meshStandardMaterial color="cyan" />
    </mesh>
  );
}

function App() {
  const [drones, setDrones] = useState([]);

  useEffect(() => {
    const socket = new WebSocket(WS_URL);

    socket.onopen = () => {
      console.log("WebSocket connected");
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        console.log("Received drone data:", data);

        if (Array.isArray(data)) {
  setDrones(data);
} else if (Array.isArray(data.drones)) {
  const formattedDrones = data.drones.map((drone) => ({
    id: drone.id,
    x: drone.position[0],
    y: drone.position[1],
    z: drone.position[2],
  }));

  setDrones(formattedDrones);
} else if (Array.isArray(data.payload)) {
  setDrones(data.payload);
}
      } catch (error) {
        console.error("Error parsing WebSocket data:", error);
      }
    };

    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    socket.onclose = () => {
      console.log("WebSocket disconnected");
    };

    return () => {
      socket.close();
    };
  }, []);

  return (
    <div
      className="app"
      style={{
        width: "100vw",
        height: "100vh",
      }}
    >
      <Canvas camera={{ position: [0, 5, 15], fov: 60 }}>
        <ambientLight intensity={0.8} />

        <directionalLight
          position={[5, 10, 5]}
          intensity={1}
        />

        {drones.map((drone, index) => (
          <Drone
            key={drone.id ?? index}
            drone={drone}
          />
        ))}
      </Canvas>
    </div>
  );
}

export default App;