import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import deque
import random
from agents.base_agent import BaseAgent

class ReplayBuffer:
    """Experience replay buffer for SAC."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)

class QNetwork(nn.Module):
    """Q-network for SAC."""
    
    def __init__(self, input_dim, action_dim, hidden_dims=(256, 256)):
        super(QNetwork, self).__init__()
        
        # Build network layers
        self.q1_layers = []
        self.q2_layers = []
        
        prev_dim = input_dim + action_dim
        
        # Separate networks for Q1 and Q2
        for hidden_dim in hidden_dims:
            self.q1_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.q1_layers.append(nn.ReLU())
            
            self.q2_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.q2_layers.append(nn.ReLU())
            
            prev_dim = hidden_dim
        
        # Output layers for Q-values
        self.q1_layers.append(nn.Linear(prev_dim, 1))
        self.q2_layers.append(nn.Linear(prev_dim, 1))
        
        # Convert to ModuleList for parameter registration
        self.q1 = nn.Sequential(*self.q1_layers)
        self.q2 = nn.Sequential(*self.q2_layers)
    
    def forward(self, state, action):
        """Forward pass to get Q-values."""
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)
    
    def q1_value(self, state, action):
        """Get Q1 value only."""
        x = torch.cat([state, action], dim=1)
        return self.q1(x)

class PolicyNetwork(nn.Module):
    """Policy network (actor) for SAC."""
    
    def __init__(self, input_dim, action_dim, hidden_dims=(256, 256), action_bounds=None,
                 log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layers for mean and log_std
        self.layers = nn.Sequential(*layers)
        self.mean = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Linear(prev_dim, action_dim)
        
        # Action bounds and log_std bounds
        self.action_bounds = action_bounds
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
    
    def forward(self, state):
        """Forward pass to get action distribution parameters."""
        x = self.layers(state)
        mean = self.mean(x)
        
        # Apply tanh to bound mean if action bounds provided
        if self.action_bounds is not None:
            low, high = self.action_bounds
            mean = torch.tanh(mean)
            mean = low + (high - low) * (mean + 1) / 2
        
        # Bound log_std to prevent numerical issues
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        """Sample action from the policy distribution."""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Sample from normal distribution
        normal = torch.distributions.Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        action = x_t  # No tanh needed if already bounded in forward
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        
        # Correct log probability for bounded actions
        if self.action_bounds is not None:
            # No need for additional corrections if we already bound in forward
            pass
        
        # Sum log probs across action dimensions
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob

class SACAgent(BaseAgent):
    """Soft Actor-Critic agent implementation."""
    
    def __init__(self, env, config):
        super().__init__(env, config)
        
        # Extract environment information
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # Get dimensions
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # Get action bounds
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        
        # Extract agent parameters from config
        self.gamma = config.get("discount_factor", 0.99)
        self.tau = config.get("tau", 0.005)  # Soft update parameter
        self.alpha = config.get("alpha", 0.2)  # Temperature parameter
        self.lr = config.get("learning_rate", 0.0003)
        self.hidden_dims = config.get("hidden_dims", (256, 256))
        self.batch_size = config.get("batch_size", 256)
        self.buffer_size = config.get("buffer_size", 1000000)
        self.target_update_interval = config.get("target_update_interval", 1)
        self.automatic_entropy_tuning = config.get("automatic_entropy_tuning", True)
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create networks
        self.critic = QNetwork(
            input_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        self.critic_target = QNetwork(
            input_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        # Initialize target network with actor/critic weights
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        self.policy = PolicyNetwork(
            input_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            action_bounds=(self.action_low, self.action_high)
        ).to(self.device)
        
        # Create optimizers
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            # Target entropy is -dim(A) (e.g. -6 for HalfCheetah-v2)
            self.target_entropy = -np.prod(self.action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        # Training info
        self.updates = 0
    
    def select_action(self, state):
        """Select an action using the current policy."""
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Use mean action during evaluation, sampled action during training
        self.policy.eval()
        with torch.no_grad():
            if not self.training:
                # Use mean action (no sampling) for evaluation
                mean, _ = self.policy(state_tensor)
                action = mean.cpu().detach().numpy()[0]
            else:
                # Sample action for training
                action, _ = self.policy.sample(state_tensor)
                action = action.cpu().detach().numpy()[0]
        self.policy.train()
        
        #