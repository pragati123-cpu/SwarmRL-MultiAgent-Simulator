from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from src.env.swarm_env import SwarmEnv


ENV_NAME = "swarm_mappo_env"


def env_creator(env_config):
    env_config = env_config or {}

    return ParallelPettingZooEnv(
        SwarmEnv(
            num_drones=env_config.get("num_drones", 3),
            max_cycles=env_config.get("max_cycles", 100),
            obstacle_count=env_config.get("obstacle_count", 0),
            obstacle_bounds=env_config.get("obstacle_bounds", 10.0),
            obstacle_radius=env_config.get("obstacle_radius", 1.0),
            obstacle_max_speed=env_config.get("obstacle_max_speed", 1.0),
            obstacle_dt=env_config.get("obstacle_dt", 1.0),
            drone_radius=env_config.get("drone_radius", 0.5),
            obstacle_collision_penalty=env_config.get(
                "obstacle_collision_penalty",
                10.0,
            ),
        )
    )


def register_swarm_env():
    register_env(ENV_NAME, env_creator)


def policy_mapping_fn(agent_id, episode, **kwargs):
    """
    Shared-policy MAPPO style setup.

    Every drone uses the same policy.
    """

    return "shared_drone_policy"


def build_mappo_config(
    num_drones=3,
    num_env_runners=2,
    obstacle_count=0,
    obstacle_bounds=10.0,
    obstacle_radius=1.0,
    obstacle_max_speed=1.0,
    obstacle_dt=1.0,
    drone_radius=0.5,
    obstacle_collision_penalty=10.0,
):
    register_swarm_env()

    config = (
        PPOConfig()

        .environment(
            ENV_NAME,
            env_config={
                "num_drones": num_drones,
                "max_cycles": 100,
                "obstacle_count": obstacle_count,
                "obstacle_bounds": obstacle_bounds,
                "obstacle_radius": obstacle_radius,
                "obstacle_max_speed": obstacle_max_speed,
                "obstacle_dt": obstacle_dt,
                "drone_radius": drone_radius,
                "obstacle_collision_penalty": obstacle_collision_penalty,
            },
        )

        .framework("torch")

        .env_runners(
            num_env_runners=num_env_runners,
        )

        .multi_agent(
            policies={"shared_drone_policy"},
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_drone_policy"],
        )

        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size_per_learner=4000,
        )
    )

    return config