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
          <group
            key={cell}
            position={[
              x * cellSize + cellSize / 2,
              0.025,
              z * cellSize + cellSize / 2,
            ]}
          >
            {/* Coverage Surface */}
            <mesh rotation={[-Math.PI / 2, 0, 0]}>
              <planeGeometry
                args={[cellSize * 0.92, cellSize * 0.92]}
              />

              <meshBasicMaterial
                color="#00ff88"
                transparent
                opacity={0.42}
                depthWrite={false}
              />
            </mesh>

            {/* Coverage Border */}
            <mesh rotation={[-Math.PI / 2, 0, 0]}>
              <ringGeometry
                args={[
                  cellSize * 0.40,
                  cellSize * 0.44,
                  4,
                ]}
              />

              <meshBasicMaterial
                color="#00ff88"
                transparent
                opacity={0.65}
                depthWrite={false}
              />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}