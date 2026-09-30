import { useEffect, useState, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

function DroneMesh({ position }) {
  return (
    <mesh position={[position?.x || 0, position?.y || 0, position?.z || 0]}>
      <sphereGeometry args={[0.4, 16, 16]} />
      <meshStandardMaterial color="#00ffff" emissive="#0088ff" emissiveIntensity={0.8} />
    </mesh>
  );
}

function App() {
  const [drones, setDrones] = useState([]);
  const [status, setStatus] = useState("Connecting...");

  useEffect(() => {
    // Try primary WebSocket endpoint
    let ws = new WebSocket("ws://localhost:8000/ws/drones");

    ws.onopen = () => {
      setStatus("Connected to Swarm WebSocket");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // Handle various payload structures
        const payloadList = data.payload || data.drones || (Array.isArray(data) ? data : []);
        if (Array.isArray(payloadList) && payloadList.length > 0) {
          setDrones(payloadList);
        }
      } catch (e) {
        console.error("Payload parse error:", e);
      }
    };

    ws.onerror = (err) => {
      console.log("WebSocket error, retrying...", err);
      setStatus("Connection Error (Checking ws://localhost:8000/ws)");
    };

    ws.onclose = () => {
      setStatus("Disconnected - Reconnecting...");
    };

    return () => ws.close();
  }, []);

  return (
    <div className="app" style={{ width: "100vw", height: "100vh", background: "#090a0f" }}>
      <div style={{ position: "absolute", top: 20, left: 20, color: "#00ffcc", zIndex: 10, fontFamily: "monospace" }}>
        <h2 style={{ margin: 0 }}>SwarmRL Multi-Agent Visualizer</h2>
        <p style={{ margin: "5px 0" }}>Status: <b style={{ color: status.includes("Connected") ? "#00ff00" : "#ff4444" }}>{status}</b></p>
        <p style={{ margin: 0 }}>Active Drones: <b>{drones.length}</b></p>
      </div>

      <Canvas camera={{ position: [0, 30, 50], fov: 60 }}>
        <ambientLight intensity={1.5} />
        <directionalLight position={[10, 20, 15]} intensity={1.5} />
        <gridHelper args={[100, 50, "#00ffcc", "#223344"]} />

        {drones.map((drone, idx) => (
          <DroneMesh key={drone.id ?? idx} position={drone.position || drone} />
        ))}

        <OrbitControls />
      </Canvas>
    </div>
  );
}

export default App;