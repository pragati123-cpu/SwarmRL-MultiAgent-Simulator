import ray

from src.training.mappo_config import build_mappo_config


def main():
    ray.init()

    config = build_mappo_config(
        num_drones=3,
        num_env_runners=2,
    )

    algorithm = config.build_algo()

    try:
        for iteration in range(5):
            result = algorithm.train()

            reward = result.get("env_runners", {}).get(
                "episode_return_mean",
                "N/A",
            )

            episodes = result.get("env_runners", {}).get(
                "num_episodes",
                "N/A",
            )

            print(
                f"Iteration {iteration + 1}: "
                f"reward={reward}, "
                f"episodes={episodes}"
            )

        checkpoint = algorithm.save()

        print("\nTraining complete.")
        print(f"Checkpoint saved at: {checkpoint}")

    finally:
        algorithm.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()