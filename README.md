# SwarmRL Week 3 - MAPPO Training Pipeline

## Task

Integrate the custom PettingZoo multi-agent drone environment
with Ray RLlib for MAPPO-style multi-agent training.

## Architecture

```text
Custom PettingZoo Parallel Environment
                |
                v
          PettingZooEnv
                |
                v
          Ray RLlib PPO
                |
                v
       Shared Drone Policy
                |
                v
       Parallel EnvRunners
                |
                v
        Multi-Agent Training
