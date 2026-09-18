"""
swarm_env package
==================

Exposes the SwarmEnv PettingZoo ParallelEnv and a conventional `env()`
factory function, following the standard PettingZoo project layout so
this module can be dropped straight into `pettingzoo.test.parallel_api_test`
or registered with Ray RLlib in later weeks.
"""

from swarm_env.environment import SwarmEnv

__all__ = ["SwarmEnv", "env", "parallel_env"]


def parallel_env(**kwargs) -> SwarmEnv:
    """Standard PettingZoo factory: returns a ParallelEnv instance."""
    return SwarmEnv(**kwargs)


# Alias kept for teammates/tools that expect the generic `env()` name.
env = parallel_env
