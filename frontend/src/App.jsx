import { useEffect, useState, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

// Altitude (height) ke hisab se dynamic color generate karne ka function
function getDroneColor(yPosition) {
  if (yPosition < 5) return "#00ff88";   // Low Altitude -> Bright Green
  if (yPosition < 15) return "#00d9ff";  // Mid Altitude  -> Neon Cyan
  return "#ff0055";                      // High Altitude -> Neon Red/Pink
}

function DroneMesh({ position }) {
  const meshRef = useRef();
  const yPos = position?.y || position?.z || 0;
  const color = getDroneColor(yPos);

  return (
    <group position={[position?.x || 0, yPos, position?.z || 0]}>
      {/* Outer Glowing Sphere */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[0.45, 16, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={1.2}
          roughness={0.2}
          metalness={0.8}
        />
      </mesh>

      {/* Point Light for Swarm Glow Effect */}
      <pointLight color={color} intensity={0.8} distance={3} />
    </group>
  );
}

function App() {
  const [drones, setDrones] = useState([]);
  const [status, setStatus] = useState("Connecting...");

  useEffect(() => {
    let ws = new WebSocket("ws://localhost:8000/ws/drones");

    ws.onopen = () => {
      setStatus("Connected to Swarm WebSocket");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const payloadList = data.payload || data.drones || (Array.isArray(data) ? data : []);
        if (Array.isArray(payloadList) && payloadList.length > 0) {
          setDrones(payloadList);
        }
      } catch (e) {
        console.error("Payload parse error:", e);
      }
    };

    ws.onerror = (err) => {
      console.log("WebSocket error...", err);
      setStatus("Connection Error");
    };

    ws.onclose = () => {
      setStatus("Disconnected - Reconnecting...");
    };

    return () => ws.close();
  }, []);

  return (
    <div className="app" style={{ width: "100vw", height: "100vh", background: "#05070a" }}>
      {/* Futuristic HUD Overlay */}
      <div style={{ position: "absolute", top: 20, left: 20, color: "#00ffcc", zIndex: 10, fontFamily: "monospace" }}>
        <h2 style={{ margin: 0, letterSpacing: "1px" }}>SWARM-RL 3D VISUALIZER</h2>
        <p style={{ margin: "5px 0" }}>
          STATUS: <b style={{ color: status.includes("Connected") ? "#00ff88" : "#ff4444" }}>{status}</b>
        </p>
        <p style={{ margin: 0 }}>
          ACTIVE AGENTS: <b style={{ color: "#ffffff" }}>{drones.length}</b>
        </p>
        
        {/* Color Legend */}
        <div style={{ marginTop: "12px", fontSize: "12px", background: "rgba(0,0,0,0.5)", padding: "8px", borderRadius: "4px" }}>
          <span style={{ color: "#00ff88", marginRight: "10px" }}>● Low Altitude</span>
          <span style={{ color: "#00d9ff", marginRight: "10px" }}>● Mid Altitude</span>
          <span style={{ color: "#ff0055" }}>● High Altitude</span>
        </div>
      </div>

      <Canvas camera={{ position: [0, 25, 45], fov: 60 }}>
        {/* Dark Ambient & Cyberpunk Lighting */}
        <color attach="background" args={["#05070a"]} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[20, 30, 10]} intensity={1.5} color="#ffffff" />
        
        {/* Cyberpunk Grid Ground */}
        <gridHelper args={[120, 60, "#00ffcc", "#112233"]} position={[0, -0.1, 0]} />

        {/* Dynamic Drones Render */}
        {drones.map((drone, idx) => (
          <DroneMesh key={drone.id ?? idx} position={drone.position || drone} />
        ))}

        <OrbitControls enableDamping dampingFactor={0.05} />
      </Canvas>
    </div>
  );
}

export default App;