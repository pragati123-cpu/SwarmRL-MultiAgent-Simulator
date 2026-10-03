import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { SensorCone } from './SensorCone';
import { AnalyticsChart } from './AnalyticsChart';

// Altitude (height) ke basis par dynamic color
function getDroneColor(yPosition) {
  if (yPosition < 5) return "#00ff88";    // Low Altitude -> Bright Green
  if (yPosition < 15) return "#00d9ff";   // Mid Altitude -> Neon Cyan
  return "#ff0055";                       // High Altitude -> Neon Pink
}

function DroneMesh({ position }) {
  const meshRef = useRef();
  const xPos = position?.x || 0;
  const yPos = position?.y || position?.z || 0;
  const zPos = position?.z || 0;
  const color = getDroneColor(yPos);

  return (
    <group position={[xPos, yPos, zPos]}>
      {/* 3D Drone Body */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[0.3, 16, 16]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.6} />
      </mesh>
      
      {/* Task 6 Dynamic Sensor Coverage Cone */}
      <SensorCone altitude={yPos} color={color} />
    </group>
  );
}

export default function App() {
  const [drones, setDrones] = useState([]);
  const [status, setStatus] = useState('Connecting...');
  
  // Real-time Dashboard Analytics Data State (Task 5)
  const [metricsData, setMetricsData] = useState([
    { time: '0s', explored: 10 },
    { time: '5s', explored: 22 },
    { time: '10s', explored: 45 },
    { time: '15s', explored: 68 },
    { time: '20s', explored: 85 },
    { time: '25s', explored: 94 },
  ]);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/drones');

    ws.onopen = () => setStatus('Connected (Streaming Swarm Data)');
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.drones) {
          setDrones(data.drones);
        }
      } catch (err) {
        console.error("Failed to parse websocket frame:", err);
      }
    };
    ws.onerror = () => setStatus('WebSocket Error');
    ws.onclose = () => setStatus('Disconnected');

    return () => ws.close();
  }, []);

  return (
    <div style={{ width: '100vw', height: '100vh', backgroundColor: '#090d16', position: 'relative', overflow: 'hidden' }}>
      {/* HUD Header */}
      <div style={{
        position: 'absolute',
        top: '15px',
        left: '20px',
        zIndex: 10,
        color: '#38bdf8',
        fontFamily: 'monospace',
        backgroundColor: 'rgba(15, 23, 42, 0.85)',
        padding: '10px 18px',
        borderRadius: '8px',
        border: '1px solid rgba(56, 189, 248, 0.2)',
        boxShadow: '0 4px 20px rgba(0,0,0,0.5)'
      }}>
        <h2 style={{ margin: 0, fontSize: '16px', textTransform: 'uppercase', letterSpacing: '1px' }}>
          SwarmRL Multi-Agent Visualizer
        </h2>
        <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: '#94a3b8' }}>
          Status: <span style={{ color: status.includes('Connected') ? '#4ade80' : '#f87171' }}>{status}</span> | Active Drones: {drones.length}
        </p>
      </div>

      {/* Task 5 Analytics Chart Overlay */}
      <AnalyticsChart data={metricsData} />

      {/* 3D Scene */}
      <Canvas camera={{ position: [0, 25, 35], fov: 60 }}>
        <ambientLight intensity={0.7} />
        <pointLight position={[10, 30, 10]} intensity={1.2} />
        <gridHelper args={[60, 60, '#38bdf8', '#1e293b']} />
        
        {drones.map((drone, idx) => (
          <DroneMesh key={drone.id || idx} position={drone.position || drone} />
        ))}

        <OrbitControls makeDefault />
      </Canvas>
    </div>
  );
}