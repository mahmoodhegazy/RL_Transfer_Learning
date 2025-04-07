import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from agents.base_agent import BaseAgent

class ActorNetwork(nn.Module):
    """Actor network for continuous action spaces."""
    
    def __init__(self, input_dim, action_dim, hidden_dims=(256, 256), action_bounds=None):
        super(ActorNetwork, self).__init__()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer for mean of action distribution
        self.layers = nn.Sequential(*layers)
        self.mu = nn.Linear(prev_dim, action_dim)
        
        # Separate parameter for log standard deviation
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Action bounds (used to scale actions)
        self.action_bounds = action_bounds
    
    def forward(self, state):
        """Forward pass to get mean and std of action distribution."""
        x = self.layers(state)
        mu = self.mu(x)
        
        # Apply tanh to bound mean if action bounds provided
        if self.action_bounds is not None:
            mu = torch.tanh(mu)
            low, high = self.action_bounds
            mu = low + (high - low) * (mu + 1) / 2
        
        # Use fixed std
        std = torch.exp(self.log_std).expand_as(mu)
        
        return mu, std
    
    def sample_action(self, state):
        """Sample action from the policy distribution."""
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        action = normal.sample()
        
        # Calculate log probability
        log_prob = normal.log_prob(action).sum(dim=-1)
        
        return action, log_prob

class CriticNetwork(nn.Module):
    """Critic network to estimate state values."""
    
    def __init__(self, input_dim, hidden_dims=(256, 256)):
        super(CriticNetwork, self).__init__()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer for state value
        layers.append(nn.Linear(prev_dim, 1))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass to get state value estimate."""
        return self.layers(state)

class ActorCriticAgent(BaseAgent):
    """Actor-Critic agent implementation for continuous control."""
    
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
        self.lr_actor = config.get("lr_actor", 0.0003)
        self.lr_critic = config.get("lr_critic", 0.001)
        self.hidden_dims = config.get("hidden_dims", (256, 256))
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.clip_grad = config.get("clip_grad", 0.5)
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create actor and critic networks
        self.actor = ActorNetwork(
            input_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            action_bounds=(self.action_low, self.action_high)
        ).to(self.device)
        
        self.critic = CriticNetwork(
            input_dim=self.obs_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        # Create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # Initialize episode tracking
        self.reset_episode()
    
    def reset_episode(self):
        """Reset episode-specific data."""
        self.episode_states = []
        self.episode_actions = []
        self.episode_log_probs = []
        self.episode_rewards = []
    
    def select_action(self, state):
        """Select an action using the current policy."""
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Forward pass through actor network to get action
        self.actor.eval()
        with torch.no_grad():
            action_tensor, log_prob = self.actor.sample_action(state_tensor)
        self.actor.train()
        
        # Convert to numpy and clip to valid range
        action = action_tensor.cpu().detach().numpy()[0]
        action = np.clip(action, self.action_low, self.action_high)
        
        # Save for training
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_log_probs.append(log_prob.item())
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """Update agent's knowledge after interaction."""
        # Store reward
        self.episode_rewards.append(reward)
        
        # Update networks at the end of an episode
        if done:
            self._update_networks()
            self.reset_episode()
        
        # Increment training steps
        self.training_steps += 1
    
    def _update_networks(self):
        """Update actor and critic networks using collected experience."""
        # Convert episode data to tensors
        states = torch.FloatTensor(self.episode_states).to(self.device)
        actions = torch.FloatTensor(self.episode_actions).to(self.device)
        log_probs = torch.FloatTensor(self.episode_log_probs).to(self.device)
        
        # Calculate discounted returns
        returns = self._compute_returns(self.episode_rewards)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Calculate state values
        state_values = self.critic(states).squeeze()
        
        # Calculate advantages
        advantages = returns - state_values.detach()
        
        # Calculate actor (policy) loss
        actor_loss = -(log_probs * advantages).mean()
        
        # Calculate critic (value) loss
        critic_loss = F.mse_loss(state_values, returns)
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad)
        self.critic_optimizer.step()
    
    def _compute_returns(self, rewards):
        """Compute discounted returns."""
        returns = []
        G = 0
        
        # Calculate returns in reverse (from last step to first)
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Normalize returns for stability
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def save(self, path):
        """Save agent parameters to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_steps': self.training_steps,
            'config': self.config
        }, path)
    
    def load(self, path):
        """Load agent parameters from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
        
        # Update config if needed
        for key, value in checkpoint['config'].items():
            if key not in ['env']:  # Skip environment which might not be serializable
                self.config[key] = value
    
    def extract_knowledge(self, knowledge_type):
        """Extract agent's knowledge for transfer learning."""
        if knowledge_type == "parameters":
            # Return all network parameters
            return {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_config": {
                    "input_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "hidden_dims": self.hidden_dims,
                    "action_bounds": (self.action_low, self.action_high)
                },
                "critic_config": {
                    "input_dim": self.obs_dim,
                    "hidden_dims": self.hidden_dims
                },
                "training_steps": self.training_steps
            }
        elif knowledge_type == "feature_extractor":
            # Extract only feature extractor layers (all layers except the final ones)
            actor_features = {}
            critic_features = {}
            
            # Extract layers except output layers
            for name, param in self.actor.layers.named_parameters():
                actor_features[name] = param.data.clone()
            
            for name, param in self.critic.layers.named_parameters():
                critic_features[name] = param.data.clone()
            
            return {
                "actor_features": actor_features,
                "critic_features": critic_features,
                "actor_config": {
                    "input_dim": self.obs_dim,
                    "hidden_dims": self.hidden_dims
                },
                "critic_config": {
                    "input_dim": self.obs_dim,
                    "hidden_dims": self.hidden_dims
                }
            }
        elif knowledge_type == "policy":
            # Return the actor network (policy) parameters
            return {
                "actor_state_dict": self.actor.state_dict(),
                "actor_config": {
                    "input_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "hidden_dims": self.hidden_dims,
                    "action_bounds": (self.action_low, self.action_high)
                }
            }
        elif knowledge_type == "value_function":
            # Return the critic network (value function) parameters
            return {
                "critic_state_dict": self.critic.state_dict(),
                "critic_config": {
                    "input_dim": self.obs_dim,
                    "hidden_dims": self.hidden_dims
                }
            }
        else:
            raise ValueError(f"Unknown knowledge type: {knowledge_type}")
    
    def initialize_from_knowledge(self, knowledge):
        """Initialize agent using transferred knowledge."""
        # Check for full parameter transfer
        if "actor_state_dict" in knowledge and "critic_state_dict" in knowledge:
            source_actor_config = knowledge.get("actor_config", {})
            source_critic_config = knowledge.get("critic_config", {})
            
            # Check for compatibility
            if self._is_network_compatible(source_actor_config, source_critic_config):
                # Direct load if dimensions match
                try:
                    self.actor.load_state_dict(knowledge["actor_state_dict"])
                    self.critic.load_state_dict(knowledge["critic_state_dict"])
                    self.training_steps = knowledge.get("training_steps", 0)
                    return True
                except Exception as e:
                    print(f"Error loading state dictionaries: {e}")
                    # Fall back to partial loading
            
            # If direct loading fails or dimensions don't match, try partial loading
            self._load_partial_network(knowledge)
        
        # Check for feature extractor transfer
        elif "actor_features" in knowledge and "critic_features" in knowledge:
            source_actor_config = knowledge.get("actor_config", {})
            source_critic_config = knowledge.get("critic_config", {})
            
            # Load feature extractor layers
            self._load_feature_layers(knowledge)
        
        # Check for policy-only transfer
        elif "actor_state_dict" in knowledge:
            source_actor_config = knowledge.get("actor_config", {})
            
            # Try to load actor network
            self._load_actor_network(knowledge)
        
        # Check for value function-only transfer
        elif "critic_state_dict" in knowledge:
            source_critic_config = knowledge.get("critic_config", {})
            
            # Try to load critic network
            self._load_critic_network(knowledge)
    
    def _is_network_compatible(self, source_actor_config, source_critic_config):
        """Check if source and target networks are compatible for direct loading."""
        # Check actor dimensions
        actor_input_compatible = source_actor_config.get("input_dim", self.obs_dim) == self.obs_dim
        actor_output_compatible = source_actor_config.get("action_dim", self.action_dim) == self.action_dim
        
        # Check critic input dimension
        critic_input_compatible = source_critic_config.get("input_dim", self.obs_dim) == self.obs_dim
        
        return actor_input_compatible and actor_output_compatible and critic_input_compatible
    
    def _load_partial_network(self, knowledge):
        """Load compatible parts of the network when full loading fails."""
        # For the actor network
        if "actor_state_dict" in knowledge:
            source_state_dict = knowledge["actor_state_dict"]
            target_state_dict = self.actor.state_dict()
            
            # Load params that match in name and shape
            for name, param in source_state_dict.items():
                if name in target_state_dict and param.shape == target_state_dict[name].shape:
                    target_state_dict[name].copy_(param)
            
            # Load partial state dict
            self.actor.load_state_dict(target_state_dict, strict=False)
        
        # For the critic network
        if "critic_state_dict" in knowledge:
            source_state_dict = knowledge["critic_state_dict"]
            target_state_dict = self.critic.state_dict()
            
            # Load params that match in name and shape
            for name, param in source_state_dict.items():
                if name in target_state_dict and param.shape == target_state_dict[name].shape:
                    target_state_dict[name].copy_(param)
            
            # Load partial state dict
            self.critic.load_state_dict(target_state_dict, strict=False)
    
    def _load_feature_layers(self, knowledge):
        """Load feature extractor layers from knowledge."""
        if "actor_features" in knowledge:
            actor_features = knowledge["actor_features"]
            
            # Load compatible feature extractor parameters
            for name, param in self.actor.layers.named_parameters():
                if name in actor_features and param.shape == actor_features[name].shape:
                    param.data.copy_(actor_features[name])
        
        if "critic_features" in knowledge:
            critic_features = knowledge["critic_features"]
            
            # Load compatible feature extractor parameters
            for name, param in self.critic.layers.named_parameters():
                if name in critic_features and param.shape == critic_features[name].shape:
                    param.data.copy_(critic_features[name])
    
    def _load_actor_network(self, knowledge):
        """Load actor network from knowledge."""
        if "actor_state_dict" in knowledge:
            source_state_dict = knowledge["actor_state_dict"]
            target_state_dict = self.actor.state_dict()
            
            # Load params that match in name and shape
            for name, param in source_state_dict.items():
                if name in target_state_dict and param.shape == target_state_dict[name].shape:
                    target_state_dict[name].copy_(param)
            
            # Load partial state dict
            self.actor.load_state_dict(target_state_dict, strict=False)
    
    def _load_critic_network(self, knowledge):
        """Load critic network from knowledge."""
        if "critic_state_dict" in knowledge:
            source_state_dict = knowledge["critic_state_dict"]
            target_state_dict = self.critic.state_dict()
            
            # Load params that match in name and shape
            for name, param in source_state_dict.items():
                if name in target_state_dict and param.shape == target_state_dict[name].shape:
                    target_state_dict[name].copy_(param)
            
            # Load partial state dict
            self.critic.load_state_dict(target_state_dict, strict=False)