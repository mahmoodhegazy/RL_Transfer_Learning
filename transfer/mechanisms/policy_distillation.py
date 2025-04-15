import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import types
from collections import deque
import random

class PolicyDistillation:
    """Transfer learning by distilling policy knowledge from source to target agent."""
    
    def __init__(self, config):
        self.config = config
        # Configuration options:
        # - temperature: Temperature parameter for distillation (higher = softer probabilities)
        # - iterations: Number of distillation iterations
        # - batch_size: Batch size for distillation
        # - learning_rate: Learning rate for distillation
        # - loss_type: Type of distillation loss ("kl", "mse")
        # - collect_states: How many states to collect from source environment
        # - buffer_size: Size of state buffer to use for distillation
        
        # Set default parameters
        self.temperature = config.get("temperature", 1.0)
        self.iterations = config.get("iterations", 1000)
        self.batch_size = config.get("batch_size", 64)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.loss_type = config.get("loss_type", "kl")
        self.collect_states = config.get("collect_states", 5000)
        self.buffer_size = config.get("buffer_size", 10000)
        
        # Device for torch operations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def transfer(self, source_agent, target_agent):
        """Transfer policy knowledge from source to target agent."""
        # Check if the agents use neural networks for policy
        if not (hasattr(source_agent, 'policy') and hasattr(target_agent, 'policy')):
            print("Warning: Policy distillation requires source and target agents to have policy networks.")
            # Fall back to direct parameter transfer if possible
            if hasattr(source_agent, 'extract_knowledge') and hasattr(target_agent, 'initialize_from_knowledge'):
                knowledge = source_agent.extract_knowledge("parameters")
                target_agent.initialize_from_knowledge(knowledge)
            return target_agent
        
        # Create state buffer
        state_buffer = self._collect_states(source_agent)
        
        # Perform policy distillation
        self._distill_policy(source_agent, target_agent, state_buffer)
        
        return target_agent
    
    def _collect_states(self, source_agent):
        """Collect states from source environment for distillation."""
        state_buffer = deque(maxlen=self.buffer_size)
        env = source_agent.env
        
        print(f"Collecting {self.collect_states} states for policy distillation...")
        
        # Set source agent to evaluation mode
        source_agent.training = False
        
        states_collected = 0
        episode_count = 0
        
        while states_collected < self.collect_states:
            # Reset environment
            state = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
            done = False
            truncated = False
            
            while not (done or truncated) and states_collected < self.collect_states:
                # Add state to buffer
                state_buffer.append(state)
                states_collected += 1
                
                # Select action using source policy
                action = source_agent.select_action(state)
                
                # Take step in environment
                if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'spec') and hasattr(env.unwrapped.spec, 'version') and env.unwrapped.spec.version.startswith('0.'):
                    # Old OpenAI Gym format
                    next_state, _, done, _ = env.step(action)
                    truncated = False
                else:
                    # New Gymnasium format
                    next_state, _, done, truncated, _ = env.step(action)
                
                state = next_state
            
            episode_count += 1
            if episode_count % 10 == 0:
                print(f"  Collected {states_collected} states from {episode_count} episodes")
        
        # Set source agent back to training mode
        source_agent.training = True
        
        print(f"Collected {len(state_buffer)} states for distillation")
        return list(state_buffer)
    
    def _distill_policy(self, source_agent, target_agent, state_buffer):
        """Distill policy from source to target agent."""
        print(f"Distilling policy over {self.iterations} iterations...")
        
        # Set agents to evaluation mode during distillation
        source_agent.training = False
        target_agent.training = True
        
        # Get optimizer for target policy
        optimizer = optim.Adam(target_agent.policy.parameters(), lr=self.learning_rate)
        
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
                if hasattr(source_agent.policy, 'sample_action'):
                    # For stochastic policies like in Actor-Critic
                    source_actions, _ = source_agent.policy.sample_action(states_tensor)
                    # For policy distillation we want means, not samples
                    source_means, source_log_stds = source_agent.policy(states_tensor)
                elif hasattr(source_agent.policy, 'forward'):
                    # Simpler case for deterministic policies
                    source_means = source_agent.policy(states_tensor)
                    source_log_stds = None
                else:
                    print("Warning: Source policy doesn't have expected methods. Distillation may not work.")
                    break
            
            # Get target policy outputs
            if hasattr(target_agent.policy, 'sample_action'):
                target_means, target_log_stds = target_agent.policy(states_tensor)
            elif hasattr(target_agent.policy, 'forward'):
                target_means = target_agent.policy(states_tensor)
                target_log_stds = None
            else:
                print("Warning: Target policy doesn't have expected methods. Distillation may not work.")
                break
            
            # Calculate distillation loss
            if self.loss_type == "mse":
                # Simple mean squared error between means
                loss = F.mse_loss(target_means, source_means)
                
                # Add log_std loss if available
                if source_log_stds is not None and target_log_stds is not None:
                    loss += F.mse_loss(target_log_stds, source_log_stds)
                
            elif self.loss_type == "kl":
                # KL divergence between distributions (for stochastic policies)
                if source_log_stds is not None and target_log_stds is not None:
                    source_std = torch.exp(source_log_stds)
                    target_std = torch.exp(target_log_stds)
                    
                    # KL divergence between two Gaussians
                    kl_div = (source_log_stds - target_log_stds + 
                             (target_std.pow(2) + (target_means - source_means).pow(2)) / 
                             (2.0 * source_std.pow(2)) - 0.5)
                    
                    loss = kl_div.mean()
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
            
            # Log progress
            if (iteration + 1) % (self.iterations // 10) == 0:
                print(f"  Distillation iteration {iteration + 1}/{self.iterations}, Loss: {loss.item():.6f}")
        
        # Set agents back to original modes
        source_agent.training = False
        target_agent.training = True
        
        print("Policy distillation complete.")