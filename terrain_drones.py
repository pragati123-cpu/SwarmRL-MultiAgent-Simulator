import numpy as np

class DroneTerrainSimulation:
    def __init__(self, num_drones=50, grid_size=(30, 30)):
        self.num_drones = num_drones
        self.grid_size = grid_size
        self.drones = self._initialize_drones()

    def _initialize_drones(self):
        # Generate 50 dummy drone 3D positions (X, Y, Z)
        positions = []
        for i in range(self.num_drones):
            x = np.random.uniform(-15, 15)
            y = np.random.uniform(1, 10)  # altitude
            z = np.random.uniform(-15, 15)
            positions.append({"id": i, "position": [x, y, z]})
        return positions

    def get_simulation_state(self):
        return {
            "terrain_dimensions": self.grid_size,
            "drones": self.drones
        }

if __name__ == "__main__":
    sim = DroneTerrainSimulation()
    state = sim.get_simulation_state()
    print(f"Initialized 3D Terrain with {len(state['drones'])} dummy drones.")