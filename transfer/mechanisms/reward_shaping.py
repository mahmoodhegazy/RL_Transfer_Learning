import numpy as np
import torch
import types

class RewardShaping:
    """Transfer learning by shaping rewards based on source task knowledge."""
    
    def __init__(self, config):
        self.config = config
        # Configuration options:
        # - shaping_method: Method for shaping rewards ("potential_based", "policy_based", etc.)
        # - scaling_factor: Scale factor for shaping reward
        # - gamma: Discount factor for potential-based shaping
        # - combine_method: How to combine original and shaped rewards ("add", "max", etc.)
        # - decay_factor: Factor to decay shaping influence over time
        
        # Set default parameters
        self.shaping_method = config.get("shaping_method", "potential_based")
        self.scaling_factor = config.get("scaling_factor", 1.0)
        self.gamma = config.get("gamma", 0.99)
        self.combine_method = config.get("combine_method", "add")
        self.decay_factor = config.get("decay_factor", 0.999)
        self.use_state_mapping = config.get("use_state_mapping", False)
        self.state_mapping = config.get("state_mapping", {})
        self.min_scaling_factor = config.get("min_scaling_factor", 0.01)
    
    def transfer(self, source_agent, target_agent):
        """Transfer reward shaping knowledge from source to target agent."""
        # Extract appropriate knowledge based on shaping method
        if self.shaping_method == "potential_based":
            # For potential-based shaping, we need value function
            knowledge = source_agent.extract_knowledge("value_function")
        elif self.shaping_method == "policy_based":
            # For policy-based shaping, we need policy
            knowledge = source_agent.extract_knowledge("policy")
        else:
            # Default to parameters
            knowledge = source_agent.extract_knowledge("parameters")
        
        # Process knowledge for shaping
        processed_knowledge = self._process_knowledge(knowledge)
        
        # Store knowledge in target agent
        target_agent.shaping_knowledge = processed_knowledge
        
        # Attach shaping components to target agent
        self._setup_shaping(target_agent)
        
        return target_agent
    
    def _process_knowledge(self, knowledge):
        """Process the extracted knowledge for reward shaping."""
        # Process knowledge based on shaping method
        processed_knowledge = knowledge.copy()
        
        # Add shaping configuration
        processed_knowledge["shaping_config"] = {
            "method": self.shaping_method,
            "scaling_factor": self.scaling_factor,
            "gamma": self.gamma,
            "combine_method": self.combine_method,
            "decay_factor": self.decay_factor,
            "use_state_mapping": self.use_state_mapping,
            "state_mapping": self.state_mapping,
            "min_scaling_factor": self.min_scaling_factor,
        }
        
        # For potential-based shaping, normalize value function
        if self.shaping_method == "potential_based":
            if "q_table" in knowledge:
                # Normalize Q-values for tabular case
                q_table = knowledge["q_table"]
                if isinstance(q_table, np.ndarray):
                    # Array-based Q-table
                    if np.max(q_table) != np.min(q_table):
                        normalized_q = (q_table - np.min(q_table)) / (np.max(q_table) - np.min(q_table))
                    else:
                        normalized_q = q_table.copy()
                    processed_knowledge["normalized_q_table"] = normalized_q
                elif isinstance(q_table, dict):
                    # Dictionary-based Q-table
                    all_values = []
                    for state, values in q_table.items():
                        if isinstance(values, np.ndarray):
                            all_values.extend(values.flatten())
                        else:
                            all_values.extend(values)
                    
                    if len(all_values) > 0:
                        max_val = max(all_values)
                        min_val = min(all_values)
                        
                        normalized_q = {}
                        if max_val != min_val:
                            for state, values in q_table.items():
                                normalized_q[state] = (values - min_val) / (max_val - min_val)
                        else:
                            normalized_q = q_table.copy()
                        
                        processed_knowledge["normalized_q_table"] = normalized_q
        
        return processed_knowledge
    
    def _setup_shaping(self, agent):
        """Set up reward shaping in the target agent."""
        # Store current scaling factor for decay
        agent.current_scaling_factor = self.scaling_factor
        agent.shaping_step_counter = 0
        
        # Create a shaped update method
        original_update = agent.update
        
        def shaped_update(self, state, action, reward, next_state, done):
            # Calculate shaped reward
            shaped_reward = self._get_shaped_reward(state, action, reward, next_state, done)
            
            # Increment shaping counter and decay scaling factor
            self.shaping_step_counter += 1
            self.current_scaling_factor = max(
                self.current_scaling_factor * self.shaping_knowledge["shaping_config"]["decay_factor"],
                self.shaping_knowledge["shaping_config"]["min_scaling_factor"]
            )
            
            # Call original update with shaped reward
            return original_update(state, action, shaped_reward, next_state, done)
        
        # Create shaped reward calculation
        def get_shaped_reward(self, state, action, reward, next_state, done):
            """Calculate shaped reward based on shaping method."""
            shaped_component = 0
            shaping_config = self.shaping_knowledge["shaping_config"]
            method = shaping_config["method"]
            
            if method == "potential_based":
                # F(s,a,s') = γ * Φ(s') - Φ(s)
                # This is guaranteed to preserve the optimal policy
                state_potential = self._get_potential(state)
                next_potential = 0 if done else self._get_potential(next_state)
                
                shaped_component = shaping_config["gamma"] * next_potential - state_potential
                
            elif method == "policy_based":
                # Reward following the source policy
                source_action = self._get_source_action(state)
                if source_action is not None and source_action == action:
                    shaped_component = 1.0  # Bonus for following source policy
            
            # Scale the shaping component
            shaped_component *= self.current_scaling_factor
            
            # Combine with original reward
            combine_method = shaping_config["combine_method"]
            if combine_method == "add":
                return reward + shaped_component
            elif combine_method == "max":
                return max(reward, shaped_component)
            elif combine_method == "weighted_sum":
                weight = self.current_scaling_factor
                return (1 - weight) * reward + weight * shaped_component
            else:
                return reward  # Default to original reward
        
        def get_potential(self, state):
            """Get potential value for a state."""
            shaping_config = self.shaping_knowledge["shaping_config"]
            
            # Map state if needed
            if shaping_config["use_state_mapping"]:
                state_key = self._get_state_key(state)
                if state_key in shaping_config["state_mapping"]:
                    state_key = shaping_config["state_mapping"][state_key]
                state = state_key
            
            # Get potential from value function
            if "normalized_q_table" in self.shaping_knowledge:
                q_table = self.shaping_knowledge["normalized_q_table"]
                state_key = self._get_state_key(state)
                
                if state_key in q_table:
                    if isinstance(q_table[state_key], np.ndarray):
                        return np.max(q_table[state_key])
                    elif isinstance(q_table[state_key], list):
                        return max(q_table[state_key])
                    else:
                        return q_table[state_key]
            
            elif "critic_state_dict" in self.shaping_knowledge and hasattr(self, "critic"):
                # Use critic network to get potential
                try:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        value = self.critic(state_tensor).item()
                    return value
                except:
                    pass
            
            return 0.0  # Default potential
        
        def get_source_action(self, state):
            """Get action from source policy for a state."""
            if "policy" not in self.shaping_knowledge:
                return None
                
            policy = self.shaping_knowledge["policy"]
            state_key = self._get_state_key(state)
            
            # Map state if needed
            if self.shaping_knowledge["shaping_config"]["use_state_mapping"]:
                if state_key in self.shaping_knowledge["shaping_config"]["state_mapping"]:
                    state_key = self.shaping_knowledge["shaping_config"]["state_mapping"][state_key]
            
            # Get action from policy
            return policy.get(state_key)
        
        # Replace agent's update method with shaped update
        agent.update = types.MethodType(shaped_update, agent)
        
        # Add helper methods to the agent
        agent._get_shaped_reward = types.MethodType(get_shaped_reward, agent)
        agent._get_potential = types.MethodType(get_potential, agent)
        agent._get_source_action = types.MethodType(get_source_action, agent)
        
        # Make sure the agent has a method to get state keys
        if not hasattr(agent, "_get_state_key"):
            def get_state_key(self, state):
                """Convert state to a hashable key for lookup."""
                if hasattr(self, "observation_space") and hasattr(self.observation_space, "n"):
                    # Discrete observation space
                    return int(state)
                else:
                    # Continuous or complex observation space
                    if isinstance(state, np.ndarray):
                        return tuple(state.flatten())
                    elif isinstance(state, (list, tuple)):
                        return tuple(state)
                    else:
                        return state
            
            agent._get_state_key = types.MethodType(get_state_key, agent)