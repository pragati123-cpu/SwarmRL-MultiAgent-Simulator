import { Grid } from "@react-three/drei";

export function GroundCoverage({
  size = 100,
  divisions = 50,
  coveredCells = new Set(),
}) {
  const cellSize = size / divisions;

  return (
    <group>
      {/* Ground Grid */}
      <Grid
        args={[size, size]}
        cellSize={cellSize}
        cellThickness={0.5}
        cellColor="#1e293b"
        sectionSize={5}
        sectionThickness={1}
        sectionColor="#00ff88"
        fadeDistance={150}
        fadeStrength={1}
        infiniteGrid={false}
      />

      {/* Covered Ground Cells */}
      {Array.from(coveredCells).map((cell) => {
        const [x, z] = cell.split(",").map(Number);

        return (
          <mesh
            key={cell}
            position={[
              x * cellSize + cellSize / 2,
              0.02,
              z * cellSize + cellSize / 2,
            ]}
            rotation={[-Math.PI / 2, 0, 0]}
          >
            <planeGeometry args={[cellSize, cellSize]} />

            <meshBasicMaterial
              color="#00ff00"
              transparent
              opacity={0.45}
            />
          </mesh>
        );
      })}
    </group>
  );
}