import numpy as np
import torch
import copy
from ..utils.state_mapping import StateMapper

class ParameterTransfer:
    """Transfer learning by directly copying parameters between agents."""
    
    def __init__(self, config):
        self.config = config
        # Configuration options:
        # - parameter_selection: Which parameters to transfer ("all", "q_table", "policy", "value", etc.)
        # - transfer_weights: Whether to transfer neural network weights (True/False)
        # - transfer_bias: Whether to transfer neural network biases (True/False)
        # - layers_to_transfer: List of specific layers to transfer (e.g., ["0", "2", "4"] or ["all"])
        # - transfer_mapping: Dictionary mapping source layers to target layers
        # - weight_scaling: Factor to scale weights during transfer
        # - selective_transfer: Transfer only weights above certain threshold
        # - freeze_transferred: Whether to freeze transferred parameters during further training
        
        # Configuration options
        self.parameter_selection = config.get("parameter_selection", "all")
        self.transfer_weights = config.get("transfer_weights", True)
        self.transfer_bias = config.get("transfer_bias", True)
        self.layers_to_transfer = config.get("layers_to_transfer", ["all"])
        self.transfer_mapping = config.get("transfer_mapping", {})
        self.weight_scaling = config.get("weight_scaling", 1.0)
        self.threshold = config.get("threshold", 0.0)
        self.freeze_transferred = config.get("freeze_transferred", False)
        self.use_state_mapping = config.get("use_state_mapping", True)
        
        # For handling tensors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def transfer(self, source_agent, target_agent):
        """Transfer parameters from source to target agent."""
        print("Starting parameter transfer...")
        
        # Extract knowledge from source
        knowledge = source_agent.extract_knowledge("parameters")
        
        # Create state mapping between environments if needed
        if self.use_state_mapping:
            self.state_mapping = StateMapper.create_mapping(source_agent, target_agent)
            print(f"Created state mapping with {len(self.state_mapping)} mapped dimensions")
        
        # Process knowledge for target
        processed_knowledge = self._process_knowledge(knowledge, source_agent, target_agent)
        
        # Initialize target with processed knowledge
        target_agent.initialize_from_knowledge(processed_knowledge)
        
        # Freeze parameters if requested
        if self.freeze_transferred:
            self._freeze_parameters(target_agent)
            
        print("Parameter transfer completed")
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
        """Process the extracted knowledge before transfer."""
        processed_knowledge = copy.deepcopy(knowledge)
        
        # Apply parameter selection
        if self.parameter_selection != "all":
            processed_knowledge = self._select_parameters(processed_knowledge)
        
        # Apply neural network parameter processing if applicable
        if (hasattr(source_agent, 'actor') and isinstance(source_agent.actor, torch.nn.Module) and
            hasattr(target_agent, 'actor') and isinstance(target_agent.actor, torch.nn.Module)):
            
            processed_knowledge = self._process_nn_parameters(
                processed_knowledge, source_agent, target_agent
            )
        
        # Process tabular knowledge if applicable
        if "q_table" in knowledge:
            processed_knowledge = self._process_tabular_parameters(
                processed_knowledge, source_agent, target_agent
            )
            if (hasattr(source_agent.env, 'grid_size') and 
                hasattr(target_agent.env, 'grid_size') and
                source_agent.env.grid_size != target_agent.env.grid_size):
                processed_knowledge = self._scale_grid_size(
                    processed_knowledge,
                    source_agent.env.grid_size,
                    target_agent.env.grid_size
                )
        
        return processed_knowledge
    
    def _process_nn_parameters(self, knowledge, source_agent, target_agent):
        """Process neural network parameters using state mapping."""
        # Process actor network parameters
        if "actor_state_dict" in knowledge:
            source_state_dict = knowledge["actor_state_dict"]
            target_state_dict = {}
            
            # Process each layer
            for name, param in source_state_dict.items():
                # Skip layers that shouldn't be transferred
                if not self._should_transfer_layer(name):
                    continue
                
                # Process weights and biases differently
                if "weight" in name and self.transfer_weights:
                    # Apply state mapping for input layer
                    if ".0.weight" in name or "layers.0.weight" in name:
                        # This is the input layer - apply state mapping
                        if self.use_state_mapping:
                            source_weight = param.clone()
                            target_weight = target_agent.actor.state_dict()[name]
                            mapped_weight = StateMapper.map_network_weights(
                                source_weight, target_weight, self.state_mapping
                            )
                            target_state_dict[name] = mapped_weight
                        else:
                            # No mapping - handle dimension mismatch
                            target_shape = target_agent.actor.state_dict()[name].shape
                            if param.shape != target_shape:
                                # Resize weight matrix
                                target_state_dict[name] = self._resize_weight_matrix(
                                    param, target_shape
                                )
                            else:
                                target_state_dict[name] = param.clone()
                    
                    # Handle output layer (action layer)
                    elif "mu.weight" in name:
                        # This is the output layer - special handling for action dimensions
                        source_shape = param.shape
                        target_shape = target_agent.actor.state_dict()[name].shape
                        
                        if source_shape != target_shape:
                            # Use statistics-based transfer for output layer
                            target_state_dict[name] = self._transfer_action_layer(
                                param, target_shape
                            )
                        else:
                            target_state_dict[name] = param.clone()
                    
                    # Handle other weight layers
                    else:
                        target_shape = target_agent.actor.state_dict()[name].shape
                        if param.shape != target_shape:
                            # Resize intermediate layers if needed
                            target_state_dict[name] = self._resize_weight_matrix(
                                param, target_shape
                            )
                        else:
                            target_state_dict[name] = param.clone()
                
                # Process bias parameters
                elif "bias" in name and self.transfer_bias:
                    target_shape = target_agent.actor.state_dict()[name].shape
                    if param.shape != target_shape:
                        # Resize bias vector if needed
                        resized_bias = torch.zeros(target_shape)
                        min_dim = min(param.shape[0], target_shape[0])
                        resized_bias[:min_dim] = param[:min_dim]
                        target_state_dict[name] = resized_bias
                    else:
                        target_state_dict[name] = param.clone()
            
            # Update knowledge with processed state dict
            knowledge["actor_state_dict"] = target_state_dict
        
        # Process critic network parameters similarly
        if "critic_state_dict" in knowledge:
            # Apply similar processing for critic network
            source_state_dict = knowledge["critic_state_dict"]
            target_state_dict = {}
            
            for name, param in source_state_dict.items():
                # Similar processing as for actor network
                # Skip details for brevity, but the approach would be the same
                if not self._should_transfer_layer(name):
                    continue
                
                if "weight" in name and self.transfer_weights:
                    # Input layer with state mapping
                    if ".0.weight" in name or "layers.0.weight" in name:
                        if self.use_state_mapping:
                            source_weight = param.clone()
                            target_weight = target_agent.critic.state_dict()[name]
                            mapped_weight = StateMapper.map_network_weights(
                                source_weight, target_weight, self.state_mapping
                            )
                            target_state_dict[name] = mapped_weight
                        else:
                            target_shape = target_agent.critic.state_dict()[name].shape
                            if param.shape != target_shape:
                                target_state_dict[name] = self._resize_weight_matrix(
                                    param, target_shape
                                )
                            else:
                                target_state_dict[name] = param.clone()
                    else:
                        # Other layers
                        target_shape = target_agent.critic.state_dict()[name].shape
                        if param.shape != target_shape:
                            target_state_dict[name] = self._resize_weight_matrix(
                                param, target_shape
                            )
                        else:
                            target_state_dict[name] = param.clone()
                
                elif "bias" in name and self.transfer_bias:
                    target_shape = target_agent.critic.state_dict()[name].shape
                    if param.shape != target_shape:
                        resized_bias = torch.zeros(target_shape)
                        min_dim = min(param.shape[0], target_shape[0])
                        resized_bias[:min_dim] = param[:min_dim]
                        target_state_dict[name] = resized_bias
                    else:
                        target_state_dict[name] = param.clone()
            
            knowledge["critic_state_dict"] = target_state_dict
        
        return knowledge
    
    def _transfer_action_layer(self, source_weights, target_shape):
        """Transfer output layer weights using statistics-based approach."""
        # Get statistics from source weights
        source_mean = source_weights.mean().item()
        source_std = source_weights.std().item()
        
        # Initialize target weights with similar statistics
        target_weights = torch.zeros(target_shape)
        target_weights.normal_(mean=source_mean, std=source_std)
        
        # If source and target have some shared dimensions, copy those directly
        min_dim = min(source_weights.shape[0], target_shape[0])
        if min_dim > 0:
            # For shared output neurons, copy the weights directly
            target_weights[:min_dim, :source_weights.shape[1]] = \
                source_weights[:min_dim, :source_weights.shape[1]]
        
        return target_weights
    
    def _resize_weight_matrix(self, source_weights, target_shape):
        """Resize weight matrix to target shape, preserving as much information as possible."""
        # Initialize target weights
        target_weights = torch.zeros(target_shape)
        
        # Get minimum dimensions
        min_dim_0 = min(source_weights.shape[0], target_shape[0])
        min_dim_1 = min(source_weights.shape[1], target_shape[1])
        
        # Copy overlapping weights
        target_weights[:min_dim_0, :min_dim_1] = source_weights[:min_dim_0, :min_dim_1]
        
        # If target is larger, initialize remaining weights based on source statistics
        if target_shape[0] > min_dim_0 or target_shape[1] > min_dim_1:
            source_mean = source_weights.mean().item()
            source_std = source_weights.std().item() * 0.1  # Reduce variance for stability
            
            # Initialize remaining weights
            if target_shape[0] > min_dim_0:
                target_weights[min_dim_0:, :min_dim_1].normal_(mean=source_mean, std=source_std)
            
            if target_shape[1] > min_dim_1:
                target_weights[:, min_dim_1:].normal_(mean=source_mean, std=source_std)
        
        return target_weights
    
    def _should_transfer_layer(self, layer_name):
        """Check if this layer should be transferred based on configuration."""
        # Transfer all layers by default
        if "all" in self.layers_to_transfer:
            return True
        
        # Extract layer number or name
        if "." in layer_name:
            layer_id = layer_name.split('.')[0]
        else:
            layer_id = layer_name
        
        # Check if this layer is in the list to transfer
        return layer_id in self.layers_to_transfer
    
    def _select_parameters(self, knowledge):
        """Select specific parameters based on configuration."""
        selected = {}
        
        if self.parameter_selection == "tabular_only":
            # Only transfer tabular parameters like Q-tables
            for key in knowledge:
                if key in ["q_table", "q_table_a", "q_table_b"]:
                    selected[key] = knowledge[key]
            
        elif self.parameter_selection == "neural_only":
            # Only transfer neural network parameters
            for key in knowledge:
                if key in ["actor_state_dict", "critic_state_dict", "policy_state_dict"]:
                    selected[key] = knowledge[key]
        
        elif self.parameter_selection == "policy_only":
            # Only transfer policy parameters
            for key in knowledge:
                if key in ["policy", "actor_state_dict", "policy_state_dict"]:
                    selected[key] = knowledge[key]
        
        elif self.parameter_selection == "value_only":
            # Only transfer value function parameters
            for key in knowledge:
                if key in ["q_table", "critic_state_dict", "value_state_dict"]:
                    selected[key] = knowledge[key]
        
        else:
            # Default: return all parameters
            return knowledge
        
        # Add configuration information
        if "learning_rate" in knowledge:
            selected["learning_rate"] = knowledge["learning_rate"]
        if "discount_factor" in knowledge:
            selected["discount_factor"] = knowledge["discount_factor"]
        
        return selected
    
    def _process_tabular_parameters(self, knowledge, source_agent, target_agent):
        """Process tabular parameters like Q-tables."""
        # Process Q-tables for different dimensions
        if "q_table" in knowledge:
            q_table = knowledge["q_table"]
            
            # Check if source and target environments have different state/action spaces
            source_state_dim = self._get_state_space_size(source_agent)
            target_state_dim = self._get_state_space_size(target_agent)
            
            source_action_dim = self._get_action_space_size(source_agent)
            target_action_dim = self._get_action_space_size(target_agent)
            
            # Handle array-based Q-tables
            if isinstance(q_table, np.ndarray):
                # Reshape Q-table if dimensions differ
                if source_state_dim != target_state_dim or source_action_dim != target_action_dim:
                    knowledge["q_table"] = self._reshape_q_table(
                        q_table, source_state_dim, target_state_dim, 
                        source_action_dim, target_action_dim
                    )
            
            # Handle dictionary-based Q-tables
            elif isinstance(q_table, dict):
                # For dictionary-based Q-tables, we just keep states that have valid actions
                reshaped_q = {}
                
                for state, values in q_table.items():
                    if isinstance(values, np.ndarray) and len(values) > 0:
                        # Check if we need to resize the action dimension
                        if len(values) != target_action_dim:
                            # Resize action values
                            resized_values = np.zeros(target_action_dim)
                            min_dim = min(len(values), target_action_dim)
                            resized_values[:min_dim] = values[:min_dim]
                            reshaped_q[state] = resized_values
                        else:
                            reshaped_q[state] = values
                
                knowledge["q_table"] = reshaped_q
        
        # Apply scaling to Q-values
        if self.weight_scaling != 1.0:
            if "q_table" in knowledge:
                knowledge["q_table"] = knowledge["q_table"] * self.weight_scaling
        
        return knowledge
    
    def _get_state_space_size(self, agent):
        """Get the size of the agent's state space."""
        if hasattr(agent, 'observation_space'):
            if hasattr(agent.observation_space, 'n'):
                return agent.observation_space.n
            elif hasattr(agent.observation_space, 'shape'):
                return np.prod(agent.observation_space.shape)
        return 0
    
    def _get_action_space_size(self, agent):
        """Get the size of the agent's action space."""
        if hasattr(agent, 'action_space'):
            if hasattr(agent.action_space, 'n'):
                return agent.action_space.n
            elif hasattr(agent.action_space, 'shape'):
                return np.prod(agent.action_space.shape)
        return 0
    
    def _reshape_q_table(self, q_table, source_state_dim, target_state_dim, 
                          source_action_dim, target_action_dim):
        """Reshape Q-table to match target dimensions."""
        # Initialize reshaped Q-table
        if target_state_dim > 0 and target_action_dim > 0:
            reshaped_q = np.zeros((target_state_dim, target_action_dim))
            
            # Copy values from source to target, up to the minimum dimensions
            min_state_dim = min(source_state_dim, target_state_dim)
            min_action_dim = min(source_action_dim, target_action_dim)
            
            reshaped_q[:min_state_dim, :min_action_dim] = q_table[:min_state_dim, :min_action_dim]
            
            return reshaped_q
        else:
            return q_table  # Return original if dimensions are invalid
    
    def _freeze_parameters(self, agent):
        """Freeze transferred parameters to prevent updates during training."""
        # For PyTorch-based agents
        for model_attr in ['policy', 'actor', 'critic', 'value']:
            if hasattr(agent, model_attr):
                model = getattr(agent, model_attr)
                if isinstance(model, torch.nn.Module):
                    for name, param in model.named_parameters():
                        # Extract layer info from parameter name
                        parts = name.split('.')
                        if len(parts) >= 2:
                            layer_id = parts[0]
                            
                            # Freeze parameter if layer was transferred
                            if layer_id in self.layers_to_transfer or "all" in self.layers_to_transfer:
                                param.requires_grad = False