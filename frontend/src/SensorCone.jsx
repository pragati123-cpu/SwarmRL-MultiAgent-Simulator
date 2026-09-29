function SensorCone({
  radius = 0.5,
  height = 1,
  segments = 32,
}) {
  return (
    <mesh
      position={[0, -0.5, 0]}
      rotation={[Math.PI, 0, 0]}
    >
      <coneGeometry args={[radius, height, segments]} />
      <meshStandardMaterial color="yellow" />
    </mesh>
  );
}

export default SensorCone;