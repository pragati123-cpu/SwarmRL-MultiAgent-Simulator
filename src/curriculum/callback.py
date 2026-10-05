from ray.rllib.callbacks.callbacks import RLlibCallback


class CurriculumCallback(RLlibCallback):

    REWARD_THRESHOLD = 40.0

    def on_episode_end(
        self,
        *,
        episode,
        env_runner,
        metrics_logger,
        env,
        env_index,
        rl_module,
        **kwargs,
    ):

        infos = episode.get_infos()

        if not infos:
            return

        agent_info = None

        for value in infos.values():

            if isinstance(value, list):
                if value:
                    candidate = value[-1]

                    if isinstance(candidate, dict):
                        agent_info = candidate
                        break

            elif isinstance(value, dict):
                agent_info = value
                break

        if not isinstance(agent_info, dict):
            return

        curriculum_level = agent_info.get(
            "curriculum_level",
            0,
        )

        num_obstacles = agent_info.get(
            "num_obstacles",
            0,
        )

        wind_strength = agent_info.get(
            "wind_strength",
            0.0,
        )

        metrics_logger.log_value(
            "curriculum_level",
            curriculum_level,
            reduce="mean",
        )

        metrics_logger.log_value(
            "num_obstacles",
            num_obstacles,
            reduce="mean",
        )

        metrics_logger.log_value(
            "wind_strength",
            wind_strength,
            reduce="mean",
        )