import ray

from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from src.environment.curriculum_env import CurriculumSwarmEnv
from src.curriculum.callback import CurriculumCallback


ENV_NAME = "curriculum_swarm_env"


def env_creator(env_config):

    return CurriculumSwarmEnv(env_config)


register_env(ENV_NAME, env_creator)


def build_config():

    config = (
        PPOConfig()

        .environment(
            env=ENV_NAME,
            env_config={
                "num_drones": 3,
                "max_cycles": 50,
                "curriculum_level": 0,
            },
        )

        .env_runners(
            num_env_runners=0,
        )

        .learners(
            num_learners=0,
        )

        .resources(
            num_gpus=0,
        )

        .callbacks(
            CurriculumCallback,
        )

        .multi_agent(
            policies={
                "shared_drone_policy",
            },

            policy_mapping_fn=lambda agent_id, *args, **kwargs:
                "shared_drone_policy",

            policies_to_train=[
                "shared_drone_policy",
            ],
        )

        .training(
            lr=0.0003,
        )
    )

    return config


def main():

    ray.init(ignore_reinit_error=True)

    config = build_config()

    algo = config.build_algo()

    for iteration in range(5):

        result = algo.train()

        print(
            f"Iteration {iteration + 1}"
            f" | Reward: {result.get('env_runners', {}).get('episode_return_mean')}"
        )

    algo.stop()

    ray.shutdown()


if __name__ == "__main__":
    main()