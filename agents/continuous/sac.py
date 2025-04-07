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
        
        # Convert to numpy and clip to valid range
        action = action.cpu().detach().numpy()[0]
        action = np.clip(action, self.action_low, self.action_high)
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """Update agent's knowledge after interaction."""
        # Store experience in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # Only update if we have enough samples
        if len(self.replay_buffer) > self.batch_size:
            self._update_networks()
        
        # Increment training steps
        self.training_steps += 1
    
    def _update_networks(self):
        """Update networks using SAC algorithm."""
        # Sample a batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Get current alpha value
        if self.automatic_entropy_tuning:
            alpha = torch.exp(self.log_alpha).detach()
        else:
            alpha = self.alpha
        
        # ---------- Update Critic ----------
        # Get current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Sample actions and log probs from current policy
        next_actions, next_log_probs = self.policy.sample(next_states)
        
        # Get target Q-values
        target_q1, target_q2 = self.critic_target(next_states, next_actions)
        
        # Take minimum of the two Q-values (double Q-learning trick)
        target_q = torch.min(target_q1, target_q2)
        
        # Add entropy term to target
        target_q = target_q - alpha * next_log_probs
        
        # Calculate target value using Bellman equation
        target_value = rewards + (1 - dones) * self.gamma * target_q
        
        # Detach target value (stop gradients)
        target_value = target_value.detach()
        
        # Compute critic loss (MSE)
        q1_loss = F.mse_loss(current_q1, target_value)
        q2_loss = F.mse_loss(current_q2, target_value)
        critic_loss = q1_loss + q2_loss
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---------- Update Actor ----------
        # We re-sample actions for the actor update
        pi_actions, log_probs = self.policy.sample(states)
        
        # Get Q-values for the new actions (using just Q1 to reduce computations)
        q1_pi = self.critic.q1_value(states, pi_actions)
        
        # Calculate policy loss
        policy_loss = (alpha * log_probs - q1_pi).mean()
        
        # Update actor
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # ---------- Update Alpha (if automatic entropy tuning) ----------
        if self.automatic_entropy_tuning:
            # Calculate alpha loss
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            
            # Update alpha
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # ---------- Soft Update Target Networks ----------
        # Update target networks through polyak averaging
        self.updates += 1
        if self.updates % self.target_update_interval == 0:
            with torch.no_grad():
                for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )
    
    def save(self, path):
        """Save agent parameters to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None,
            'training_steps': self.training_steps,
            'updates': self.updates,
            'config': self.config
        }, path)
    
    def load(self, path):
        """Load agent parameters from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        
        if self.automatic_entropy_tuning:
            if checkpoint['log_alpha'] is not None:
                self.log_alpha = checkpoint['log_alpha']
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.training_steps = checkpoint['training_steps']
        self.updates = checkpoint.get('updates', 0)
        
        # Update config if needed
        for key, value in checkpoint['config'].items():
            if key not in ['env']:  # Skip environment which might not be serializable
                self.config[key] = value
    
    def extract_knowledge(self, knowledge_type):
        """Extract agent's knowledge for transfer learning."""
        if knowledge_type == "parameters":
            # Return all network parameters
            return {
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "policy_state_dict": self.policy.state_dict(),
                "critic_config": {
                    "input_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "hidden_dims": self.hidden_dims
                },
                "policy_config": {
                    "input_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "hidden_dims": self.hidden_dims,
                    "action_bounds": (self.action_low, self.action_high)
                },
                "training_steps": self.training_steps,
                "updates": self.updates
            }
        elif knowledge_type == "feature_extractor":
            # Extract only feature extractor layers
            critic_features = {}
            policy_features = {}
            
            # Extract layers except output layers
            for name, param in self.critic.q1.named_parameters():
                if not name.startswith(str(len(self.hidden_dims))):  # Skip last layer
                    critic_features[name] = param.data.clone()
            
            for name, param in self.policy.layers.named_parameters():
                policy_features[name] = param.data.clone()
            
            return {
                "critic_features": critic_features,
                "policy_features": policy_features,
                "critic_config": {
                    "input_dim": self.obs_dim,
                    "hidden_dims": self.hidden_dims
                },
                "policy_config": {
                    "input_dim": self.obs_dim,
                    "hidden_dims": self.hidden_dims
                }
            }
        elif knowledge_type == "policy":
            # Return the policy network parameters
            return {
                "policy_state_dict": self.policy.state_dict(),
                "policy_config": {
                    "input_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "hidden_dims": self.hidden_dims,
                    "action_bounds": (self.action_low, self.action_high)
                }
            }
        elif knowledge_type == "value_function":
            # Return the critic network parameters
            return {
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "critic_config": {
                    "input_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "hidden_dims": self.hidden_dims
                }
            }
        else:
            raise ValueError(f"Unknown knowledge type: {knowledge_type}")
    
    def initialize_from_knowledge(self, knowledge):
        """Initialize agent using transferred knowledge."""
        # Check for full parameter transfer
        if all(k in knowledge for k in ["critic_state_dict", "critic_target_state_dict", "policy_state_dict"]):
            source_critic_config = knowledge.get("critic_config", {})
            source_policy_config = knowledge.get("policy_config", {})
            
            # Check for compatibility
            if self._is_network_compatible(source_critic_config, source_policy_config):
                # Direct load if dimensions match
                try:
                    self.critic.load_state_dict(knowledge["critic_state_dict"])
                    self.critic_target.load_state_dict(knowledge["critic_target_state_dict"])
                    self.policy.load_state_dict(knowledge["policy_state_dict"])
                    self.training_steps = knowledge.get("training_steps", 0)
                    self.updates = knowledge.get("updates", 0)
                    return True
                except Exception as e:
                    print(f"Error loading state dictionaries: {e}")
                    # Fall back to partial loading
            
            # If direct loading fails or dimensions don't match, try partial loading
            self._load_partial_network(knowledge)
        
        # Check for feature extractor transfer
        elif "critic_features" in knowledge and "policy_features" in knowledge:
            source_critic_config = knowledge.get("critic_config", {})
            source_policy_config = knowledge.get("policy_config", {})
            
            # Load feature extractor layers
            self._load_feature_layers(knowledge)
        
        # Check for policy-only transfer
        elif "policy_state_dict" in knowledge:
            source_policy_config = knowledge.get("policy_config", {})
            
            # Try to load policy network
            self._load_policy_network(knowledge)
        
        # Check for value function-only transfer
        elif "critic_state_dict" in knowledge:
            source_critic_config = knowledge.get("critic_config", {})
            
            # Try to load critic network
            self._load_critic_network(knowledge)
    
    def _is_network_compatible(self, source_critic_config, source_policy_config):
        """Check if source and target networks are compatible for direct loading."""
        # Check critic dimensions
        critic_input_compatible = source_critic_config.get("input_dim", self.obs_dim) == self.obs_dim
        critic_action_compatible = source_critic_config.get("action_dim", self.action_dim) == self.action_dim
        
        # Check policy dimensions
        policy_input_compatible = source_policy_config.get("input_dim", self.obs_dim) == self.obs_dim
        policy_output_compatible = source_policy_config.get("action_dim", self.action_dim) == self.action_dim
        
        return (critic_input_compatible and critic_action_compatible and 
                policy_input_compatible and policy_output_compatible)
    
    def _load_partial_network(self, knowledge):
        """Load compatible parts of the network when full loading fails."""
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
        
        # For the critic target network
        if "critic_target_state_dict" in knowledge:
            source_state_dict = knowledge["critic_target_state_dict"]
            target_state_dict = self.critic_target.state_dict()
            
            # Load params that match in name and shape
            for name, param in source_state_dict.items():
                if name in target_state_dict and param.shape == target_state_dict[name].shape:
                    target_state_dict[name].copy_(param)
            
            # Load partial state dict
            self.critic_target.load_state_dict(target_state_dict, strict=False)
        
        # For the policy network
        if "policy_state_dict" in knowledge:
            source_state_dict = knowledge["policy_state_dict"]
            target_state_dict = self.policy.state_dict()
            
            # Load params that match in name and shape
            for name, param in source_state_dict.items():
                if name in target_state_dict and param.shape == target_state_dict[name].shape:
                    target_state_dict[name].copy_(param)
            
            # Load partial state dict
            self.policy.load_state_dict(target_state_dict, strict=False)
    
    def _load_feature_layers(self, knowledge):
        """Load feature extractor layers from knowledge."""
        if "critic_features" in knowledge:
            critic_features = knowledge["critic_features"]
            
            # Load compatible feature extractor parameters for q1
            for name, param in self.critic.q1.named_parameters():
                if name in critic_features and param.shape == critic_features[name].shape:
                    param.data.copy_(critic_features[name])
            
            # Load compatible feature extractor parameters for q2
            for name, param in self.critic.q2.named_parameters():
                if name in critic_features and param.shape == critic_features[name].shape:
                    param.data.copy_(critic_features[name])
        
        if "policy_features" in knowledge:
            policy_features = knowledge["policy_features"]
            
            # Load compatible feature extractor parameters
            for name, param in self.policy.layers.named_parameters():
                if name in policy_features and param.shape == policy_features[name].shape:
                    param.data.copy_(policy_features[name])
    
    def _load_policy_network(self, knowledge):
        """Load policy network from knowledge."""
        if "policy_state_dict" in knowledge:
            source_state_dict = knowledge["policy_state_dict"]
            target_state_dict = self.policy.state_dict()
            
            # Load params that match in name and shape
            for name, param in source_state_dict.items():
                if name in target_state_dict and param.shape == target_state_dict[name].shape:
                    target_state_dict[name].copy_(param)
            
            # Load partial state dict
            self.policy.load_state_dict(target_state_dict, strict=False)
    
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
        
        if "critic_target_state_dict" in knowledge:
            source_state_dict = knowledge["critic_target_state_dict"]
            target_state_dict = self.critic_target.state_dict()
            
            # Load params that match in name and shape
            for name, param in source_state_dict.items():
                if name in target_state_dict and param.shape == target_state_dict[name].shape:
                    target_state_dict[name].copy_(param)
            
            # Load partial state dict
            self.critic_target.load_state_dict(target_state_dict, strict=False)