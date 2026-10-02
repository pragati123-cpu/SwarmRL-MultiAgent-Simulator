import torch
import torch.nn as nn
import torch.optim as optim

class CentralizedCritic(nn.Module):
    """
    MAPPO Centralized Critic Network for 50 Drones Swarm.
    Evaluates global state concatenated from all agents during training.
    """
    def __init__(self, num_drones=50, state_dim_per_drone=3, hidden_dim=256):
        super(CentralizedCritic, self).__init__()
        
        self.num_drones = num_drones
        self.state_dim_per_drone = state_dim_per_drone
        
        # Global state dimension = 50 drones * state per drone (e.g., x, y, z = 3)
        self.global_state_dim = num_drones * state_dim_per_drone
        
        # Centralized Critic MLP Architecture
        self.network = nn.Sequential(
            nn.Linear(self.global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single State-Value output V(S_global)
        )

    def forward(self, global_state):
        """
        Args:
            global_state (Tensor): Tensor of shape (batch_size, num_drones * state_dim_per_drone)
                                   or (num_drones * state_dim_per_drone,)
        Returns:
            Tensor: State-value V(S_global)
        """
        return self.network(global_state)


if __name__ == "__main__":
    # Test script for 50 drones
    num_drones = 50
    state_per_drone = 3  # (x, y, z) coordinates
    
    critic = CentralizedCritic(num_drones=num_drones, state_dim_per_drone=state_per_drone)
    
    # Dummy global state representing 50 drones (e.g., batch_size = 1)
    dummy_global_state = torch.randn(1, num_drones * state_per_drone)
    
    value_estimate = critic(dummy_global_state)
    
    print("=" * 50)
    print("MAPPO Centralized Critic Network Setup Successful!")
    print(f"Input Global State Shape: {dummy_global_state.shape}")
    print(f"Output Value Estimate Shape: {value_estimate.shape}")
    print(f"Sample Value Output: {value_estimate.item():.4f}")
    print("=" * 50)