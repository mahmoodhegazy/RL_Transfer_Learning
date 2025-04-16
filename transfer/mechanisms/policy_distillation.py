import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from ..utils.state_mapping import StateMapper

class PolicyDistillation:
    """Transfer learning by distilling policy knowledge from source to target agent."""
    
    def __init__(self, config):
        self.config = config
        
        # Configuration options
        self.temperature = config.get("temperature", 1.0)
        self.iterations = config.get("iterations", 1000)
        self.batch_size = config.get("batch_size", 64)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.loss_type = config.get("loss_type", "kl")
        self.collect_states = config.get("collect_states", 5000)
        self.buffer_size = config.get("buffer_size", 10000)
        self.use_state_mapping = config.get("use_state_mapping", True)
        
        # Device for torch operations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def transfer(self, source_agent, target_agent):
        """Transfer policy knowledge from source to target agent."""
        print("Starting policy distillation...")
        
        # Check if the agents have actor networks
        if not hasattr(source_agent, 'actor') or not hasattr(target_agent, 'actor'):
            print("Warning: Policy distillation requires source and target agents to have actor networks.")
            print("Falling back to parameter transfer...")
            # Fall back to direct parameter transfer
            if hasattr(source_agent, 'extract_knowledge') and hasattr(target_agent, 'initialize_from_knowledge'):
                knowledge = source_agent.extract_knowledge("parameters")
                target_agent.initialize_from_knowledge(knowledge)
            return target_agent
        
        # Create state mapping between environments if needed
        if self.use_state_mapping:
            self.state_mapping = StateMapper.create_mapping(source_agent, target_agent)
            print(f"Created state mapping with {len(self.state_mapping)} mapped dimensions")
        
        # Collect states from source environment
        print(f"Collecting {self.collect_states} states for policy distillation...")
        state_buffer = self._collect_states(source_agent)
        
        # Map states to target environment representation if needed
        if self.use_state_mapping:
            mapped_state_buffer = self._map_states(state_buffer, source_agent, target_agent)
        else:
            mapped_state_buffer = state_buffer
        
        # Perform policy distillation
        self._distill_policy(source_agent, target_agent, mapped_state_buffer)
            
        print("Policy distillation complete.")
        return target_agent
    
    def _collect_states(self, source_agent):
        """Collect diverse states from source environment for distillation."""
        state_buffer = []
        env = source_agent.env
        
        # Set source agent to evaluation mode
        source_agent.training = False
        
        states_collected = 0
        episode_count = 0
        
        # Create a set of state representations to track diversity
        state_representations = set()
        
        while states_collected < self.collect_states and episode_count < 200:
            # Reset environment
            if hasattr(env, 'reset') and callable(env.reset):
                if isinstance(env.reset(), tuple):
                    state = env.reset()[0]
                else:
                    state = env.reset()
            else:
                print("Environment does not have a proper reset method")
                break
                
            done = False
            truncated = False
            episode_steps = 0
            
            while not (done or truncated) and states_collected < self.collect_states and episode_steps < 200:
                # Get state representation for diversity tracking
                if hasattr(env, 'get_state_representation'):
                    state_repr = str(env.get_state_representation().round(2))
                else:
                    state_repr = str(np.round(state, 2))
                
                # Only add state if it's sufficiently different from existing states
                if state_repr not in state_representations:
                    state_buffer.append(state)
                    state_representations.add(state_repr)
                    states_collected += 1
                
                # Select action using source policy
                action = source_agent.select_action(state)
                
                # Take step in environment
                try:
                    result = env.step(action)
                    if len(result) == 5:  # New Gymnasium API
                        next_state, _, done, truncated, _ = result
                    else:  # Old Gym API
                        next_state, _, done, _ = result
                        truncated = False
                    
                    state = next_state
                    episode_steps += 1
                except Exception as e:
                    print(f"Error stepping environment: {e}")
                    break
            
            episode_count += 1
            if episode_count % 10 == 0:
                print(f"  Collected {states_collected} states from {episode_count} episodes")
        
        # Set source agent back to training mode
        source_agent.training = True
        
        print(f"Collected {len(state_buffer)} diverse states for distillation")
        return state_buffer
    
    def _map_states(self, state_buffer, source_agent, target_agent):
        """Map states from source to target representation using state mapping."""
        mapped_states = []
        
        # Skip if no mapping is available
        if not hasattr(self, 'state_mapping') or not self.state_mapping:
            return state_buffer
        
        # Get dimensions
        source_dims = source_agent.obs_dim if hasattr(source_agent, 'obs_dim') else \
                     source_agent.observation_space.shape[0]
        target_dims = target_agent.obs_dim if hasattr(target_agent, 'obs_dim') else \
                     target_agent.observation_space.shape[0]
        
        # Prepare a zero-filled target state template
        target_template = np.zeros(target_dims)
        
        # Process each state
        for source_state in state_buffer:
            # Create a new target state
            target_state = target_template.copy()
            
            # Apply mapping
            for source_idx, target_idx in self.state_mapping.items():
                if source_idx < len(source_state) and target_idx < len(target_state):
                    target_state[target_idx] = source_state[source_idx]
            
            mapped_states.append(target_state)
        
        return mapped_states
    
    def _distill_policy(self, source_agent, target_agent, state_buffer):
        """Distill policy from source to target agent."""
        print(f"Distilling policy over {self.iterations} iterations...")
        
        # Set agents to appropriate modes
        source_agent.training = False  # Evaluation mode for source
        target_agent.training = True   # Training mode for target
        
        # Ensure actor networks are in the right mode
        source_agent.actor.eval()
        target_agent.actor.train()
        
        # Get optimizer for target policy
        optimizer = optim.Adam(target_agent.actor.parameters(), lr=self.learning_rate)
        
        # Prepare for tracking progress
        best_loss = float('inf')
        early_stop_counter = 0
        early_stop_patience = 10
        
        # Distillation loop
        for iteration in range(self.iterations):
            # Sample batch of states
            if len(state_buffer) <= self.batch_size:
                batch_states = state_buffer
            else:
                batch_indices = np.random.choice(len(state_buffer), self.batch_size, replace=False)
                batch_states = [state_buffer[i] for i in batch_indices]
            
            # Convert states to tensor
            states_tensor = torch.FloatTensor(np.array(batch_states)).to(self.device)
            
            # Get source policy outputs
            with torch.no_grad():
                # Try different ways to get policy outputs depending on implementation
                try:
                    # Try actor that returns mean and log_std
                    source_means, source_log_stds = source_agent.actor(states_tensor)
                except:
                    try:
                        # Try policy that returns action and log_prob
                        source_actions, _ = source_agent.actor.sample_action(states_tensor)
                        # Use actions as means for distillation
                        source_means = source_actions
                        # Create dummy log_stds
                        source_log_stds = torch.zeros_like(source_means) - 1.0
                    except:
                        print("Unable to extract policy outputs from source agent")
                        return
            
            # Get target policy outputs
            try:
                # Try actor that returns mean and log_std
                target_means, target_log_stds = target_agent.actor(states_tensor)
            except:
                try:
                    # Try policy that returns action and log_prob
                    target_actions, _ = target_agent.actor.sample_action(states_tensor)
                    # Use actions as means for distillation
                    target_means = target_actions
                    # Create dummy log_stds
                    target_log_stds = torch.zeros_like(target_means) - 1.0
                except:
                    print("Unable to extract policy outputs from target agent")
                    return
            
            # Calculate distillation loss
            if self.loss_type == "mse":
                # Simple mean squared error between means
                loss = F.mse_loss(target_means, source_means)
                
                # Add log_std loss if available
                if source_log_stds is not None and target_log_stds is not None:
                    loss += F.mse_loss(target_log_stds, source_log_stds)
                
            elif self.loss_type == "kl":
                # KL divergence between distributions
                if source_log_stds is not None and target_log_stds is not None:
                    source_std = torch.exp(source_log_stds)
                    target_std = torch.exp(target_log_stds)
                    
                    # KL divergence between two Gaussians
                    kl_div = (source_log_stds - target_log_stds + 
                             (target_std.pow(2) + (target_means - source_means).pow(2)) / 
                             (2.0 * source_std.pow(2)) - 0.5)
                    
                    loss = kl_div.sum(dim=1).mean()
                else:
                    # Fall back to MSE if log_stds not available
                    loss = F.mse_loss(target_means, source_means)
            
            else:
                # Default to MSE
                loss = F.mse_loss(target_means, source_means)
            
            # Update target policy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                
            # Stop if loss hasn't improved for a while
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping at iteration {iteration + 1} due to lack of improvement")
                break
            
            # Log progress
            if (iteration + 1) % (self.iterations // 10) == 0:
                print(f"  Distillation iteration {iteration + 1}/{self.iterations}, Loss: {loss.item():.6f}")
        
        # Set agents back to original modes
        source_agent.training = False
        target_agent.training = True
    
    def _compute_kl_divergence(self, mu1, logvar1, mu2, logvar2):
        """Compute KL divergence between two Gaussian distributions."""
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        
        kl_div = 0.5 * (
            logvar2 - logvar1 + 
            (var1 + (mu1 - mu2).pow(2)) / var2 - 1
        )
        
        return kl_div.sum(dim=1).mean()