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
):
    register_swarm_env()

    config = (
        PPOConfig()

        .environment(
            ENV_NAME,
            env_config={
                "num_drones": num_drones,
                "max_cycles": 100,
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