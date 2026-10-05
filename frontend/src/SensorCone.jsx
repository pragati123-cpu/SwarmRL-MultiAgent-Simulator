import React from 'react';
import * as THREE from 'three';

export function SensorCone({ position = [0, 0, 0], fovRange = 5, color = "#00f0ff" }) {
  // Height/Altitude y-axis par hoti hai
  const height = position[1] || position[2] || 5; 
  const radius = height * Math.tan((45 * Math.PI) / 360); // 45 degree FOV angle

  return (
    <group position={[0, 0, 0]}>
      {/* 1. Dynamic Sensor FOV Cone (Drone se Ground tak) */}
      <mesh position={[0, -height / 2, 0]} rotation={[Math.PI, 0, 0]}>
        <coneGeometry args={[radius, height, 32, 1, true]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.8}
          roughness={0.45}
          transparent
          opacity={0.16}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>

      {/* 2. Ground Coverage Circle Projection */}
      <mesh position={[0, -height + 0.02, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0, radius, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.65}
          roughness={0.45}
          transparent
          opacity={0.24}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}
