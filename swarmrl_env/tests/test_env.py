"""
Sanity tests for SwarmEnv.reset() and SwarmEnv.step().

Run with:
    pytest tests/test_env.py -v

These are lightweight correctness checks for this week's scope
(reset/step logic), not a full RL training test.
"""

import numpy as np
import pytest

from swarm_env import SwarmEnv


N_AGENTS = 6  # small swarm for fast tests


def make_env(**overrides):
    kwargs = dict(
        n_agents=N_AGENTS,
        n_neighbors=3,
        world_size=20.0,
        max_episode_steps=10,
        collision_radius=1.0,
        max_speed=2.0,
        grid_cell_size=2.0,
    )
    kwargs.update(overrides)
    return SwarmEnv(**kwargs)


def random_actions(env):
    return {
        agent: env.action_space(agent).sample() for agent in env.agents
    }


def test_reset_returns_all_agents_with_correct_shapes():
    env = make_env()
    observations, infos = env.reset(seed=42)

    assert set(observations.keys()) == set(env.possible_agents)
    assert set(infos.keys()) == set(env.possible_agents)

    for agent in env.agents:
        obs = observations[agent]
        assert obs.shape == env.observation_space(agent).shape
        assert np.all(np.isfinite(obs))


def test_reset_is_deterministic_given_seed():
    env_a = make_env()
    env_b = make_env()

    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)

    for agent in env_a.possible_agents:
        assert np.allclose(obs_a[agent], obs_b[agent])


def test_step_returns_expected_dict_shapes():
    env = make_env()
    env.reset(seed=1)
    actions = random_actions(env)

    observations, rewards, terminations, truncations, infos = env.step(actions)

    for d in (observations, rewards, terminations, truncations, infos):
        assert set(d.keys()) == set(env.possible_agents)

    for agent in env.possible_agents:
        assert observations[agent].shape == env.observation_space(agent).shape
        assert isinstance(rewards[agent], float)
        assert isinstance(terminations[agent], bool)
        assert isinstance(truncations[agent], bool)


def test_positions_stay_within_world_bounds():
    env = make_env(max_episode_steps=25)
    env.reset(seed=7)

    for _ in range(25):
        if not env.agents:
            break
        actions = random_actions(env)
        env.step(actions)

    for pos in env.positions.values():
        assert np.all(pos >= -env.half_size - 1e-4)
        assert np.all(pos <= env.half_size + 1e-4)


def test_episode_truncates_at_max_steps():
    env = make_env(max_episode_steps=5)
    env.reset(seed=3)

    truncations = {}
    for _ in range(5):
        actions = random_actions(env)
        _, _, _, truncations, _ = env.step(actions)

    assert all(truncations.values())
    assert env.agents == []  # cleared after episode end


def test_collision_triggers_large_negative_reward():
    # Force two agents to occupy (nearly) the same cell by using a
    # collision_radius large enough that any two random spawns collide.
    env = make_env(n_agents=2, collision_radius=1000.0, max_speed=0.0)
    env.reset(seed=5)

    actions = {agent: np.zeros(3, dtype=np.float32) for agent in env.agents}
    _, rewards, _, _, infos = env.step(actions)

    assert all(infos[agent]["collided"] for agent in env.possible_agents)
    assert all(rewards[agent] <= -99.0 for agent in env.possible_agents)


def test_exploration_reward_given_once_per_cell():
    env = make_env(n_agents=2, grid_cell_size=1000.0, max_speed=0.0)
    env.reset(seed=9)

    actions = {agent: np.zeros(3, dtype=np.float32) for agent in env.agents}

    # First step in a fresh (huge) cell area away from spawn cells that
    # were pre-marked visited during reset: use a tiny nonzero throttle
    # via nonzero action isn't guaranteed to change cell with max_speed=0,
    # so instead assert the invariant: an agent cannot repeatedly earn a
    # reward for a cell it has already visited.
    _, rewards_first, _, _, _ = env.step(actions)
    _, rewards_second, _, _, _ = env.step(actions)

    for agent in env.possible_agents:
        # Same cell as spawn (max_speed=0) -> already visited -> no bonus.
        assert rewards_first[agent] <= 0.0
        assert rewards_second[agent] <= 0.0


def test_step_after_episode_end_is_safe_noop():
    env = make_env(max_episode_steps=1)
    env.reset(seed=11)
    actions = random_actions(env)
    env.step(actions)  # ends episode (agents cleared)

    observations, rewards, terminations, truncations, infos = env.step({})
    assert observations == {}
    assert rewards == {}
    assert terminations == {}
    assert truncations == {}
    assert infos == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
