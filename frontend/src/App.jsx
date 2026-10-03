import { SensorCone } from './SensorCone';
import { GroundCoverage } from './GroundCoverage';
import { useEffect, useState, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

// Altitude (height) ke basis par dynamic color
function getDroneColor(yPosition) {
  if (yPosition < 5) return "#00ff88";    // Low Altitude -> Bright Green
  if (yPosition < 15) return "#00d9ff";   // Mid Altitude -> Neon Cyan
  return "#ff0055";                      // High Altitude -> Neon Pink
}

function DroneMesh({ position }) {
  const meshRef = useRef();
  const xPos = position?.x || 0;
  const yPos = position?.y || position?.z || 0;
  const zPos = position?.z || 0;
  const color = getDroneColor(yPos);

  return (
    <group position={[xPos, yPos, zPos]}>
      {/* Drone Body */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[0.45, 16, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={1.5}
          roughness={0.2}
        />
      </mesh>

      {/* Task 6: Sensor Coverage Cone attached to Drone */}
      <SensorCone 
        position={[xPos, yPos, zPos]} 
        color={color}
      />
    </group>
  );
}

export default function App() {
  const [drones, setDrones] = useState([]);
  const [status, setStatus] = useState("Connecting...");

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/drones");

    ws.onopen = () => setStatus("Connected to Swarm WebSocket");
    ws.onclose = () => setStatus("Disconnected");
    ws.onerror = () => setStatus("Connection Error");

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.drones) {
          setDrones(data.drones);
        } else if (Array.isArray(data)) {
          setDrones(data);
        }
      } catch (err) {
        console.error("Telemetry parse error:", err);
      }
    };

    return () => ws.close();
  }, []);

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#05070c", position: "relative" }}>
      {/* Top Left HUD Display */}
      <div style={{
        position: "absolute",
        top: "20px",
        left: "20px",
        zIndex: 10,
        color: "#00f0ff",
        fontFamily: "monospace",
        background: "rgba(10, 15, 28, 0.85)",
        padding: "15px 22px",
        borderRadius: "8px",
        border: "1px solid rgba(0, 240, 255, 0.3)",
        boxShadow: "0 0 15px rgba(0, 240, 255, 0.15)",
        pointerEvents: "none"
      }}>
        <h2 style={{ margin: "0 0 6px 0", fontSize: "18px", color: "#ffffff" }}>
          SwarmRL Multi-Agent Visualizer
        </h2>
        <div style={{ fontSize: "13px", color: "#a0aec0", marginBottom: "4px" }}>
          Status: <span style={{ color: status.includes("Connected") ? "#00ff88" : "#ff0055", fontWeight: "bold" }}>{status}</span>
        </div>
        <div style={{ fontSize: "13px", color: "#a0aec0" }}>
          Active Drones: <span style={{ color: "#00f0ff", fontWeight: "bold" }}>{drones.length || 50}</span>
        </div>
      </div>

      {/* 3D Canvas */}
      <Canvas camera={{ position: [0, 25, 35], fov: 60 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 20, 15]} intensity={1.2} />
        <OrbitControls makeDefault />

        {/* Ground Coverage Grid */}
        <GroundCoverage size={100} divisions={50}/>

        {/* Drones + Sensor Coverage Cones */}
        {drones.map((drone, idx) => (
          <DroneMesh key={drone.id || idx} position={drone} />
        ))}
      </Canvas>
    </div>
  );
}