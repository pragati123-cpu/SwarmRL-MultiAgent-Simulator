class CurriculumManager:
    """
    Controls progressive environment difficulty.
    """

    CURRICULUM = {
        0: {
            "num_obstacles": 1,
            "dynamic_obstacles": False,
            "wind_strength": 0.0,
        },
        1: {
            "num_obstacles": 3,
            "dynamic_obstacles": False,
            "wind_strength": 0.0,
        },
        2: {
            "num_obstacles": 3,
            "dynamic_obstacles": True,
            "wind_strength": 0.0,
        },
        3: {
            "num_obstacles": 5,
            "dynamic_obstacles": True,
            "wind_strength": 0.5,
        },
    }

    def __init__(self, start_level=0):
        self.level = start_level

    def get_config(self):
        return self.CURRICULUM[self.level].copy()

    def increase_level(self):
        if self.level < max(self.CURRICULUM):
            self.level += 1

        return self.level

    def set_level(self, level):
        if level not in self.CURRICULUM:
            raise ValueError(f"Invalid curriculum level: {level}")

        self.level = level

    def get_level(self):
        return self.level

    def is_final_level(self):
        return self.level == max(self.CURRICULUM)