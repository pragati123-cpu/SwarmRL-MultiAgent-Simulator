from src.curriculum.manager import CurriculumManager
from src.environment.curriculum_env import CurriculumSwarmEnv


def test_curriculum_progression():

    manager = CurriculumManager()

    assert manager.get_level() == 0

    manager.increase_level()
    assert manager.get_level() == 1

    manager.increase_level()
    assert manager.get_level() == 2

    manager.increase_level()
    assert manager.get_level() == 3

    manager.increase_level()
    assert manager.get_level() == 3


def test_level_zero_is_easy():

    manager = CurriculumManager()

    config = manager.get_config()

    assert config["num_obstacles"] == 1
    assert config["dynamic_obstacles"] is False
    assert config["wind_strength"] == 0.0


def test_level_three_is_hard():

    manager = CurriculumManager()

    manager.set_level(3)

    config = manager.get_config()

    assert config["num_obstacles"] == 5
    assert config["dynamic_obstacles"] is True
    assert config["wind_strength"] > 0


def test_environment_uses_curriculum():

    env = CurriculumSwarmEnv({
        "num_drones": 3,
        "curriculum_level": 2,
    })

    assert env.curriculum_level == 2
    assert env.difficulty["dynamic_obstacles"] is True