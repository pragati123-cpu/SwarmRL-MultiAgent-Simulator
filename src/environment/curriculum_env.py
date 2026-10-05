import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from src.curriculum.manager import CurriculumManager


class CurriculumSwarmEnv(MultiAgentEnv):
    """
    Multi-agent swarm environment with curriculum learning.

    Curriculum levels:

    Level 0:
        - 1 obstacle
        - Static obstacles
        - No wind

    Level 1:
        - 3 obstacles
        - Static obstacles
        - No wind

    Level 2:
        - 3 obstacles
        - Dynamic obstacles
        - No wind

    Level 3:
        - 5 obstacles
        - Dynamic obstacles
        - Wind enabled
    """

    def __init__(self, config=None):
        super().__init__()

        config = config or {}

        # -----------------------------
        # Environment configuration
        # -----------------------------
        self.num_drones = config.get("num_drones", 3)
        self.max_cycles = config.get("max_cycles", 100)

        # Starting curriculum level
        self.curriculum_level = config.get(
            "curriculum_level",
            0,
        )

        # -----------------------------
        # Curriculum manager
        # -----------------------------
        self.curriculum = CurriculumManager()

        self.curriculum.set_level(
            self.curriculum_level
        )

        self.difficulty = (
            self.curriculum.get_config()
        )

        # -----------------------------
        # Agent configuration
        # -----------------------------
        self.possible_agents = [
            f"drone_{i}"
            for i in range(self.num_drones)
        ]

        self.agents = self.possible_agents.copy()

        # -----------------------------
        # Observation spaces
        # -----------------------------
        self.observation_spaces = {
            agent: spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(8,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        # -----------------------------
        # Action spaces
        # -----------------------------
        self.action_spaces = {
            agent: spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        # -----------------------------
        # Episode state
        # -----------------------------
        self.step_count = 0

    # =========================================================
    # Curriculum control
    # =========================================================

    def set_curriculum_level(self, level):
        """
        Change the current curriculum level.

        Parameters
        ----------
        level : int
            Curriculum level from 0 to 3.
        """

        self.curriculum.set_level(level)

        self.curriculum_level = (
            self.curriculum.get_level()
        )

        self.difficulty = (
            self.curriculum.get_config()
        )

    # =========================================================
    # Reset
    # =========================================================

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment at the beginning of an episode.
        """

        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0

        self.agents = self.possible_agents.copy()

        # -----------------------------------------
        # Initial observations
        # -----------------------------------------

        observations = {
            agent: np.zeros(
                8,
                dtype=np.float32,
            )
            for agent in self.agents
        }

        # -----------------------------------------
        # Environment information
        # -----------------------------------------

        infos = {
            agent: {
                "curriculum_level": self.curriculum_level,
                "num_obstacles": self.difficulty[
                    "num_obstacles"
                ],
                "dynamic_obstacles": self.difficulty[
                    "dynamic_obstacles"
                ],
                "wind_strength": self.difficulty[
                    "wind_strength"
                ],
            }
            for agent in self.agents
        }

        return observations, infos

    # =========================================================
    # Step
    # =========================================================

    def step(self, action_dict):
        """
        Execute one environment step.
        """

        self.step_count += 1

        # -----------------------------------------
        # Generate observations
        # -----------------------------------------

        observations = {
            agent: np.random.uniform(
                -1.0,
                1.0,
                size=8,
            ).astype(np.float32)
            for agent in self.agents
        }

        # -----------------------------------------
        # Calculate rewards
        # -----------------------------------------

        rewards = {}

        for agent in self.agents:

            # Base reward
            reward = 1.0

            # Higher curriculum levels are harder
            reward -= (
                self.curriculum_level * 0.05
            )

            # Wind adds additional difficulty
            reward -= (
                self.difficulty["wind_strength"]
                * 0.02
            )

            # Dynamic obstacles add a small difficulty penalty
            if self.difficulty[
                "dynamic_obstacles"
            ]:
                reward -= 0.02

            # More obstacles add a small difficulty penalty
            obstacle_penalty = (
                max(
                    0,
                    self.difficulty["num_obstacles"] - 1,
                )
                * 0.005
            )

            reward -= obstacle_penalty

            rewards[agent] = reward

        # -----------------------------------------
        # Termination
        # -----------------------------------------

        terminated = (
            self.step_count >= self.max_cycles
        )

        terminateds = {
            agent: terminated
            for agent in self.agents
        }

        terminateds["__all__"] = terminated

        # -----------------------------------------
        # Truncation
        # -----------------------------------------

        truncateds = {
            agent: False
            for agent in self.agents
        }

        truncateds["__all__"] = False

        # -----------------------------------------
        # Episode information
        # -----------------------------------------

        infos = {
            agent: {
                "curriculum_level": self.curriculum_level,
                "num_obstacles": self.difficulty[
                    "num_obstacles"
                ],
                "dynamic_obstacles": self.difficulty[
                    "dynamic_obstacles"
                ],
                "wind_strength": self.difficulty[
                    "wind_strength"
                ],
            }
            for agent in self.agents
        }

        return (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        )

    # =========================================================
    # RLlib space methods
    # =========================================================

    def get_observation_space(self, agent_id):
        """
        Return observation space for an agent.
        """

        return self.observation_spaces[agent_id]

    def get_action_space(self, agent_id):
        """
        Return action space for an agent.
        """

        return self.action_spaces[agent_id]