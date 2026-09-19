"""
Quick manual smoke-test / demo for SwarmEnv.

Runs a short episode with random actions for every agent and prints
per-step summary stats, so you can eyeball that reset()/step() behave
sanely before handing the env off to the RL / rendering tracks.

Usage:
    python demo.py
"""

from swarm_env import SwarmEnv


def main():
    env = SwarmEnv(
        n_agents=10,
        n_neighbors=3,
        world_size=40.0,
        max_episode_steps=20,
        collision_radius=1.5,
        max_speed=2.0,
        grid_cell_size=3.0,
        render_mode="human",
    )

    observations, infos = env.reset(seed=0)
    print(f"Reset complete. Active agents: {len(env.agents)}")
    print(f"Observation shape per agent: {observations[env.agents[0]].shape}")

    step = 0
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        env.render()

        step += 1
        total_reward = sum(rewards.values())
        n_collisions = sum(1 for i in infos.values() if i.get("collided"))
        print(
            f"step={step:03d} total_reward={total_reward:7.2f} "
            f"collisions={n_collisions} active_agents={len(env.agents)}"
        )

    env.close()
    print("Episode finished.")


if __name__ == "__main__":
    main()
