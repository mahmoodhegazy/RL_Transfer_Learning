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
        # Ensure agents have policies and are in the right mode
        if not hasattr(source_agent, 'actor') or not hasattr(target_agent, 'actor'):
            raise ValueError("Both agents must have actor networks for distillation")
        
        # Set source to evaluation mode, target to training
        source_agent.actor.eval()
        target_agent.actor.train()
        
        # Get optimizer for target policy
        optimizer = optim.Adam(target_agent.actor.parameters(), lr=self.learning_rate)
        
        # Distillation loop
        for iteration in range(self.iterations):
            # Sample batch of states
            batch_indices = np.random.choice(len(state_buffer), 
                                            min(self.batch_size, len(state_buffer)), 
                                            replace=False)
            batch_states = [state_buffer[i] for i in batch_indices]
            
            # Convert states to tensor
            states_tensor = torch.FloatTensor(np.array(batch_states)).to(self.device)
            
            # Get source policy outputs
            with torch.no_grad():
                source_means, source_log_stds = source_agent.actor(states_tensor)
                
            # Get target policy outputs
            target_means, target_log_stds = target_agent.actor(states_tensor)
            
            # Calculate distillation loss - KL divergence
            loss = self._compute_kl_divergence(source_means, source_log_stds,
                                            target_means, target_log_stds)
            
            # Update target policy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log progress
            if (iteration + 1) % 100 == 0:
                print(f"Distillation iteration {iteration+1}/{self.iterations}, Loss: {loss.item():.6f}")
        
        # Set agents back to original modes
        source_agent.training = False
        target_agent.training = True
        
        print("Policy distillation complete.")