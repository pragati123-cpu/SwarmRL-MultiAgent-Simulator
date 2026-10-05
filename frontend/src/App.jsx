import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { SensorCone } from './SensorCone';
import { AnalyticsChart } from './AnalyticsChart';

// Altitude ke basis par dynamic color
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
      
      {/* Sensor Coverage Cone */}
      <SensorCone altitude={yPos} color={color} />
    </group>
  );
}

export default function App() {
  const [drones, setDrones] = useState([]);
  const [status, setStatus] = useState('Connecting...');
  const [collisions, setCollisions] = useState(0); // Task 6: Collision Tracking
  
  // Real-time Dashboard Analytics Data State
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
        if (data.collisions !== undefined) {
          setCollisions(data.collisions);
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
      {/* Polished HUD Header (Task 6 UI Polish) */}
      <div style={{
        position: 'absolute',
        top: '15px',
        left: '20px',
        zIndex: 10,
        color: '#38bdf8',
        fontFamily: 'monospace',
        backgroundColor: 'rgba(15, 23, 42, 0.85)',
        backdropFilter: 'blur(8px)',
        padding: '12px 20px',
        borderRadius: '10px',
        border: '1px solid rgba(56, 189, 248, 0.3)',
        boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)'
      }}>
        <h2 style={{ margin: 0, fontSize: '16px', textTransform: 'uppercase', letterSpacing: '1px', color: '#38bdf8' }}>
          SwarmRL Multi-Agent Visualizer
        </h2>
        <p style={{ margin: '6px 0 0 0', fontSize: '12px', color: '#94a3b8' }}>
          Status: <span style={{ color: status.includes('Connected') ? '#4ade80' : '#f87171' }}>{status}</span> | Active Drones: <span style={{ color: '#fff' }}>{drones.length}</span>
        </p>
      </div>

      {/* Task 6: Real-time Collision Tracking Box */}
      <div style={{
        position: 'absolute',
        top: '15px',
        right: '20px',
        zIndex: 10,
        fontFamily: 'monospace',
        backgroundColor: 'rgba(15, 23, 42, 0.85)',
        backdropFilter: 'blur(8px)',
        padding: '10px 18px',
        borderRadius: '10px',
        border: '1px solid rgba(239, 68, 68, 0.4)',
        boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
        display: 'flex',
        alignItems: 'center',
        gap: '10px'
      }}>
        <div style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: collisions > 0 ? '#ef4444' : '#22c55e' }}></div>
        <span style={{ fontSize: '13px', fontWeight: 'bold', color: '#f87171' }}>
          Total Collisions: <span style={{ color: '#fff', fontSize: '15px' }}>{collisions}</span>
        </span>
      </div>

      {/* Analytics Chart Overlay */}
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