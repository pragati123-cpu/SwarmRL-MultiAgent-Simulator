export function getGroundCoverage(
  sensorX,
  sensorZ,
  radius,
  cellSize = 1
) {
  const cells = [];

  const minX = Math.floor((sensorX - radius) / cellSize);
  const maxX = Math.floor((sensorX + radius) / cellSize);

  const minZ = Math.floor((sensorZ - radius) / cellSize);
  const maxZ = Math.floor((sensorZ + radius) / cellSize);

  for (let x = minX; x <= maxX; x++) {
    for (let z = minZ; z <= maxZ; z++) {
      const cellCenterX = x * cellSize + cellSize / 2;
      const cellCenterZ = z * cellSize + cellSize / 2;

      const dx = cellCenterX - sensorX;
      const dz = cellCenterZ - sensorZ;

      const distance = Math.sqrt(
        dx * dx + dz * dz
      );

      if (distance <= radius) {
        cells.push({
          x,
          z
        });
      }
    }
  }

  return cells;
}