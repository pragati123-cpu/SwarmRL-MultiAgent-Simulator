import { Grid } from "@react-three/drei";

export function GroundCoverage({
  size = 100,
  divisions = 50,
}) {
  return (
    <Grid
      args={[size, size]}
      cellSize={size / divisions}
      cellThickness={0.5}
      cellColor="#1e293b"
      sectionSize={5}
      sectionThickness={1}
      sectionColor="#00ff88"
      fadeDistance={150}
      fadeStrength={1}
      infiniteGrid={false}
    />
  );
}