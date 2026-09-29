function SensorCone() {
  return (
    <mesh
      position={[0, -0.5, 0]}
      rotation={[Math.PI, 0, 0]}
    >
      <coneGeometry args={[0.5, 1, 32]} />
      <meshStandardMaterial color="yellow" />
    </mesh>
  );
}

export default SensorCone;