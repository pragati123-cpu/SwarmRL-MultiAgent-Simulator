import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { SensorCone } from './SensorCone';
import { AnalyticsChart } from './AnalyticsChart';


// Sensor coverage radius
const SENSOR_RADIUS = 2;


// Ground grid configuration
const GROUND_SIZE = 100;
const GROUND_DIVISIONS = 50;

const TOTAL_CELLS =
  GROUND_DIVISIONS * GROUND_DIVISIONS;


// Altitude-based drone color
function getDroneColor(yPosition) {
  if (yPosition < 5) return "#00ff88";    // Low Altitude -> Bright Green
  if (yPosition < 15) return "#00d9ff";   // Mid Altitude -> Neon Cyan
  return "#ff0055";                       // High Altitude -> Neon Pink
}


// Drone component
function DroneMesh({
  position,
  onCoverageUpdate
}) {

  const meshRef = useRef();

  // Drone coordinates
  const xPos = position?.x ?? 0;
  const yPos = position?.y ?? 0;
  const zPos = position?.z ?? 0;

  const color = getDroneColor(yPos);


  // Real-time ground coverage detection
  useEffect(() => {

    const coverage = getGroundCoverage(
      xPos,
      zPos,
      SENSOR_RADIUS
    );

    if (onCoverageUpdate) {
      onCoverageUpdate(coverage);
    }

  }, [
    xPos,
    zPos,
    onCoverageUpdate
  ]);


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

  // Drone telemetry data
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


  // WebSocket status
  const [status, setStatus] = useState(
    "Connecting..."
  );


  // Track covered ground cells
  const [coveredCells, setCoveredCells] =
    useState(new Set());


  // Calculate real-time coverage percentage
  const coveragePercentage = Math.min(
    100,
    (
      coveredCells.size /
      TOTAL_CELLS
    ) * 100
  );


  // Optimized coverage update
  const updateCoveredCells = useCallback(
    (newCells) => {

      setCoveredCells(
        (previousCells) => {

          let hasNewCells = false;

          const updatedCells =
            new Set(previousCells);


          newCells.forEach(
            ({ x, z }) => {

              const cellKey =
                `${x},${z}`;


              // Add only cells that are not
              // already covered
              if (
                !updatedCells.has(
                  cellKey
                )
              ) {

                updatedCells.add(
                  cellKey
                );

                hasNewCells = true;

              }

            }
          );


          // Avoid unnecessary React
          // state updates
          if (!hasNewCells) {
            return previousCells;
          }


          return updatedCells;

        }
      );

    },
    []
  );



  // Coverage debug information
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/drones');

    ws.onopen = () => setStatus('Connected (Streaming Swarm Data)');
    ws.onmessage = (event) => {

      try {

        const data =
          JSON.parse(event.data);


        // Backend sends object
        if (data.drones) {
          setDrones(data.drones);
        }
      } catch (err) {
        console.error("Failed to parse websocket frame:", err);
      }
      catch (err) {

        console.error(
          "Telemetry parse error:",
          err
        );

      }

    };


    // Cleanup WebSocket
    return () => {

      ws.close();

    };
    ws.onerror = () => setStatus('WebSocket Error');
    ws.onclose = () => setStatus('Disconnected');

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