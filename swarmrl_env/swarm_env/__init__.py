from swarmrl_env.swarm_env.environment import SwarmEnv

__all__ = ["SwarmEnv", "env", "parallel_env"]


def parallel_env(**kwargs) -> SwarmEnv:
    return SwarmEnv(**kwargs)
env = parallel_env
