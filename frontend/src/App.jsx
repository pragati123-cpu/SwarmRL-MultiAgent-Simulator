import { getGroundCoverage } from "./sensorCoverage";
import { SensorCone } from "./SensorCone";
import { GroundCoverage } from "./GroundCoverage";

import {
  useEffect,
  useState,
  useRef,
  useCallback
} from "react";

import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";


// Sensor coverage radius
const SENSOR_RADIUS = 2;


// Ground grid configuration
const GROUND_SIZE = 100;
const GROUND_DIVISIONS = 50;

const TOTAL_CELLS =
  GROUND_DIVISIONS * GROUND_DIVISIONS;


// Altitude-based drone color
function getDroneColor(yPosition) {
  if (yPosition < 5) return "#00ff88";
  if (yPosition < 15) return "#00d9ff";
  return "#ff0055";
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
    <group
      position={[
        xPos,
        yPos,
        zPos
      ]}
    >

      {/* Drone Body */}
      <mesh ref={meshRef}>

        <sphereGeometry
          args={[
            0.45,
            16,
            16
          ]}
        />

        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={1.5}
          roughness={0.2}
        />

      </mesh>


      {/* Sensor Coverage Cone */}
      <SensorCone
        position={[
          xPos,
          yPos,
          zPos
        ]}
        color={color}
      />

    </group>
  );
}



export default function App() {

  // Drone telemetry data
  const [drones, setDrones] = useState([]);


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

    console.log(
      "Real-time covered cells:",
      coveredCells.size
    );

    console.log(
      "Ground coverage:",
      coveragePercentage.toFixed(2) + "%"
    );

  }, [
    coveredCells,
    coveragePercentage
  ]);



  // WebSocket connection
  useEffect(() => {

    const ws = new WebSocket(
      "ws://localhost:8000/ws/drones"
    );


    // WebSocket connected
    ws.onopen = () => {

      setStatus(
        "Connected to Swarm WebSocket"
      );

    };


    // WebSocket disconnected
    ws.onclose = () => {

      setStatus(
        "Disconnected"
      );

    };


    // WebSocket error
    ws.onerror = () => {

      setStatus(
        "Connection Error"
      );

    };


    // Receive drone telemetry
    ws.onmessage = (event) => {

      try {

        const data =
          JSON.parse(event.data);


        // Backend sends object
        if (data.drones) {

          setDrones(
            data.drones
          );

        }


        // Backend sends array
        else if (
          Array.isArray(data)
        ) {

          setDrones(data);

        }

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

  }, []);



  return (

    <div
      style={{
        width: "100vw",
        height: "100vh",
        background: "#05070c",
        position: "relative"
      }}
    >

      {/* =========================
          TOP LEFT HUD
      ========================== */}

      <div
        style={{
          position: "absolute",
          top: "20px",
          left: "20px",
          zIndex: 10,

          color: "#00f0ff",

          fontFamily: "monospace",

          background:
            "rgba(10, 15, 28, 0.85)",

          padding: "15px 22px",

          borderRadius: "8px",

          border:
            "1px solid rgba(0, 240, 255, 0.3)",

          boxShadow:
            "0 0 15px rgba(0, 240, 255, 0.15)",

          pointerEvents: "none"
        }}
      >

        {/* Title */}

        <h2
          style={{
            margin:
              "0 0 6px 0",

            fontSize: "18px",

            color: "#ffffff"
          }}
        >
          SwarmRL Multi-Agent Visualizer
        </h2>



        {/* WebSocket Status */}

        <div
          style={{
            fontSize: "13px",

            color: "#a0aec0",

            marginBottom: "4px"
          }}
        >

          Status:{" "}

          <span
            style={{
              color:
                status.includes(
                  "Connected"
                )
                  ? "#00ff88"
                  : "#ff0055",

              fontWeight:
                "bold"
            }}
          >
            {status}
          </span>

        </div>



        {/* Active Drones */}

        <div
          style={{
            fontSize: "13px",

            color: "#a0aec0",

            marginBottom: "4px"
          }}
        >

          Active Drones:{" "}

          <span
            style={{
              color: "#00f0ff",

              fontWeight:
                "bold"
            }}
          >
            {drones.length || 50}
          </span>

        </div>



        {/* Covered Cells */}

        <div
          style={{
            fontSize: "13px",

            color: "#a0aec0",

            marginBottom: "4px"
          }}
        >

          Covered Cells:{" "}

          <span
            style={{
              color: "#00ff88",

              fontWeight:
                "bold"
            }}
          >
            {coveredCells.size}
          </span>

        </div>



        {/* Ground Coverage Percentage */}

        <div
          style={{
            fontSize: "13px",

            color: "#a0aec0",

            marginTop: "4px"
          }}
        >

          Ground Coverage:{" "}

          <span
            style={{
              color: "#00ff88",

              fontWeight:
                "bold"
            }}
          >
            {coveragePercentage.toFixed(2)}%
          </span>

        </div>



        {/* Coverage Progress Bar */}

        <div
          style={{
            width: "180px",

            height: "6px",

            background:
              "rgba(255,255,255,0.1)",

            borderRadius: "4px",

            marginTop: "6px",

            overflow: "hidden"
          }}
        >

          <div
            style={{
              width:
                `${coveragePercentage}%`,

              height: "100%",

              background: "#00ff88",

              borderRadius: "4px",

              transition:
                "width 0.3s ease"
            }}
          />

        </div>

      </div>



      {/* =========================
          3D CANVAS
      ========================== */}

      <Canvas
        camera={{
          position: [
            0,
            25,
            35
          ],

          fov: 60
        }}
      >

        {/* Ambient Light */}

        <ambientLight
          intensity={0.6}
        />


        {/* Directional Light */}

        <directionalLight
          position={[
            10,
            20,
            15
          ]}

          intensity={1.2}
        />


        {/* Camera Controls */}

        <OrbitControls
          makeDefault
        />


        {/* =========================
            GROUND COVERAGE
        ========================== */}

        <GroundCoverage
          size={GROUND_SIZE}

          divisions={
            GROUND_DIVISIONS
          }

          coveredCells={
            coveredCells
          }
        />


        {/* =========================
            DRONES
        ========================== */}

        {drones.map(
          (drone, idx) => (

            <DroneMesh
              key={
                drone.id || idx
              }

              position={drone}

              onCoverageUpdate={
                updateCoveredCells
              }

            />

          )
        )}

      </Canvas>

    </div>

  );
}