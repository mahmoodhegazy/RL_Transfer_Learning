import numpy as np
import torch
import copy

class ValueTransfer:
    """Transfer learning by copying and adapting value functions."""
    
    def __init__(self, config):
        self.config = config
        # Configuration options:
        # - transfer_type: Type of value to transfer ("q_values", "v_function", "advantage")
        # - scaling_factor: Scale factor for transferred values
        # - adaptation_method: Method for adapting values to target environment
        #   ("direct", "normalized", "rescaled", "learned_mapping")
        # - initialize_unseen: How to initialize unseen states in target
        #   ("zeros", "random", "average")
        
        # Set default parameters
        self.transfer_type = config.get("transfer_type", "q_values")
        self.scaling_factor = config.get("scaling_factor", 1.0)
        self.adaptation_method = config.get("adaptation_method", "normalized")
        self.initialize_unseen = config.get("initialize_unseen", "zeros")
        self.use_state_mapping = config.get("use_state_mapping", False)
        self.state_mapping = config.get("state_mapping", {})
        
        # For torch operations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def transfer(self, source_agent, target_agent):
        """Transfer value function knowledge from source to target agent."""
        # Extract value function knowledge from source
        knowledge = source_agent.extract_knowledge("value_function")
        
        # Process and adapt the knowledge
        processed_knowledge = self._process_knowledge(knowledge, source_agent, target_agent)
        
        # Initialize target agent with processed knowledge
        target_agent.initialize_from_knowledge(processed_knowledge)
        
        return target_agent
    
    def _scale_grid_size(self, knowledge, source_size=3, target_size=5):
        """Scale Q-table from smaller grid to larger grid."""
        if "q_table" not in knowledge:
            return knowledge
        
        q_table = knowledge["q_table"]
        scaled_q = {}
        
        # Calculate scaling ratio
        scale_ratio = (target_size - 1) / (source_size - 1)
        
        def _scale_destination_index(dest_idx, source_size, target_size):
            # Convert flat index back to 2D coordinates
            dest_row = dest_idx // source_size
            dest_col = dest_idx % source_size
            
            # Scale the coordinates
            scaled_row = int(round(dest_row * scale_ratio))
            scaled_col = int(round(dest_col * scale_ratio))
            
            # Convert back to flat index for target grid
            return scaled_row * target_size + scaled_col
        
        if isinstance(q_table, dict):
            for source_state, values in q_table.items():
                # Convert tuple state to list for manipulation
                state = list(source_state)
                row, col = state[0], state[1]
                dest_idx = state[3]  # Get destination index
                
                # Scale the taxi position
                scaled_row = int(round(row * scale_ratio))
                scaled_col = int(round(col * scale_ratio))
                
                # Scale the destination index
                scaled_dest = _scale_destination_index(dest_idx, source_size, target_size)
                
                # Create new state tuple with scaled position and destination
                scaled_state = tuple([scaled_row, scaled_col, state[2], scaled_dest] + list(source_state[4:]))
                scaled_q[scaled_state] = values.copy() if isinstance(values, np.ndarray) else values
                
                # Fill surrounding states with same values
                if row == 1 and col == 1:  # Middle state in 3x3 grid
                    # Fill all states that map to middle
                    for r in range(1, 4):
                        for c in range(1, 4):
                            if (r, c) != (2, 2):  # Skip the exact middle
                                middle_state = tuple([r, c, state[2], scaled_dest] + list(source_state[4:]))
                                scaled_q[middle_state] = values.copy() if isinstance(values, np.ndarray) else values
        
        knowledge["q_table"] = scaled_q
        return knowledge

    def _process_knowledge(self, knowledge, source_agent, target_agent):
        """Process value function knowledge for transfer."""
        processed_knowledge = copy.deepcopy(knowledge)
        
        if "q_table" in knowledge and (self.transfer_type == "q_values" or self.transfer_type == "value_function"):
            # Process Q-table (for tabular methods)
            q_table = knowledge["q_table"]
            
            if self.adaptation_method == "normalized":
                # Normalize Q-values to [0, 1] range
                processed_knowledge = self._normalize_q_values(q_table)
                    # Scale grid size if needed
            if (hasattr(source_agent.env, 'grid_size') and 
                hasattr(target_agent.env, 'grid_size') and
                source_agent.env.grid_size != target_agent.env.grid_size):
                processed_knowledge = self._scale_grid_size(
                    processed_knowledge,
                    source_agent.env.grid_size,
                    target_agent.env.grid_size
                )
            elif self.adaptation_method == "rescaled":
                # Rescale based on reward ranges of environments
                if hasattr(source_agent.env, 'reward_range') and hasattr(target_agent.env, 'reward_range'):
                    source_range = source_agent.env.reward_range
                    target_range = target_agent.env.reward_range
                    
                    source_scale = source_range[1] - source_range[0]
                    target_scale = target_range[1] - target_range[0]
                    
                    if source_scale > 0 and target_scale > 0:
                        scale_factor = target_scale / source_scale
                        
                        if isinstance(q_table, np.ndarray):
                            processed_knowledge["q_table"] = q_table * scale_factor
                        elif isinstance(q_table, dict):
                            scaled_q = {}
                            for state, values in q_table.items():
                                scaled_q[state] = values * scale_factor
                            processed_knowledge["q_table"] = scaled_q
            
            # Apply state mapping if specified
            if self.use_state_mapping and self.state_mapping:
                processed_knowledge = self._apply_state_mapping(processed_knowledge)
        
        elif "critic_state_dict" in knowledge and self.transfer_type in ["v_function", "q_function"]:
            # For neural network value functions, we'll rely on the agent's initialize_from_knowledge
            # method to handle the adaptation correctly
            pass
        
        return processed_knowledge
    
    def _normalize_q_values(self, q_table):
        """Normalize Q-values to [0, 1] range."""
        normalized_knowledge = {}
        
        if isinstance(q_table, np.ndarray):
            # Array-based Q-table
            q_min = np.min(q_table)
            q_max = np.max(q_table)
            
            if q_min == q_max:
                normalized_knowledge["q_table"] = np.zeros_like(q_table)
            else:
                normalized_knowledge["q_table"] = (q_table - q_min) / (q_max - q_min)
                normalized_knowledge["q_table"] = normalized_knowledge["q_table"] * self.scaling_factor
        
        elif isinstance(q_table, dict):
            # Dictionary-based Q-table
            all_values = []
            for state, values in q_table.items():
                if isinstance(values, np.ndarray):
                    all_values.extend(values.flatten())
                else:
                    all_values.append(values)
            
            if not all_values:
                normalized_knowledge["q_table"] = q_table.copy()
            else:
                q_min = min(all_values)
                q_max = max(all_values)
                
                normalized_q = {}
                if q_min == q_max:
                    # All values are the same, set to zeros
                    for state, values in q_table.items():
                        if isinstance(values, np.ndarray):
                            normalized_q[state] = np.zeros_like(values)
                        else:
                            normalized_q[state] = 0.0
                else:
                    # Normalize to [0, 1] range
                    for state, values in q_table.items():
                        if isinstance(values, np.ndarray):
                            normalized_q[state] = (values - q_min) / (q_max - q_min)
                            normalized_q[state] = normalized_q[state] * self.scaling_factor
                        else:
                            normalized_q[state] = ((values - q_min) / (q_max - q_min)) * self.scaling_factor
                
                normalized_knowledge["q_table"] = normalized_q
        
        return normalized_knowledge
    
    def _apply_state_mapping(self, knowledge):
        """Apply state mapping to transfer knowledge between differently structured state spaces."""
        if "q_table" not in knowledge:
            return knowledge
        
        q_table = knowledge["q_table"]
        mapped_q = {}
        
        if isinstance(q_table, dict):
            # For dictionary-based Q-tables
            for source_state, values in q_table.items():
                if source_state in self.state_mapping:
                    target_state = self.state_mapping[source_state]
                    mapped_q[target_state] = values
                else:
                    # Keep unmapped states as is
                    mapped_q[source_state] = values
            
            knowledge["q_table"] = mapped_q
        
        return knowledge