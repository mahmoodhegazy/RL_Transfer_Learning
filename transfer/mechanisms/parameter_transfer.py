import numpy as np
import torch
import copy

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
        
        # Set default parameters
        self.parameter_selection = config.get("parameter_selection", "all")
        self.transfer_weights = config.get("transfer_weights", True)
        self.transfer_bias = config.get("transfer_bias", True)
        self.layers_to_transfer = config.get("layers_to_transfer", ["all"])
        self.transfer_mapping = config.get("transfer_mapping", {})
        self.weight_scaling = config.get("weight_scaling", 1.0)
        self.threshold = config.get("threshold", 0.0)  # For selective transfer
        self.freeze_transferred = config.get("freeze_transferred", False)
        
        # For handling tensors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def transfer(self, source_agent, target_agent):
        """Transfer parameters from source to target agent."""
        # Extract knowledge from source
        knowledge = source_agent.extract_knowledge("parameters")
        
        # Process knowledge if needed
        processed_knowledge = self._process_knowledge(knowledge, source_agent, target_agent)
        
        # Initialize target with processed knowledge
        target_agent.initialize_from_knowledge(processed_knowledge)
        
        # Freeze parameters if requested
        if self.freeze_transferred:
            self._freeze_parameters(target_agent)
        
        return target_agent
    
    def _process_knowledge(self, knowledge, source_agent, target_agent):
        """Process the extracted knowledge before transfer."""
        processed_knowledge = copy.deepcopy(knowledge)
        
        # Apply parameter selection
        if self.parameter_selection != "all":
            processed_knowledge = self._select_parameters(processed_knowledge)
        
        # Apply neural network parameter processing if applicable
        if (hasattr(source_agent, 'policy') and isinstance(source_agent.policy, torch.nn.Module) and
            hasattr(target_agent, 'policy') and isinstance(target_agent.policy, torch.nn.Module)):
            processed_knowledge = self._process_nn_parameters(processed_knowledge)
        
        # Apply tabular knowledge processing if applicable
        if "q_table" in knowledge:
            processed_knowledge = self._process_tabular_parameters(
                processed_knowledge, source_agent, target_agent
            )
        
        return processed_knowledge
    
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
    
    def _process_nn_parameters(self, knowledge):
        """Process neural network parameters."""
        # If not transferring weights or biases, modify state dicts
        if not self.transfer_weights or not self.transfer_bias:
            for key in ["actor_state_dict", "critic_state_dict", 
                        "policy_state_dict", "value_state_dict"]:
                if key in knowledge:
                    state_dict = knowledge[key]
                    filtered_state_dict = {}
                    
                    for param_name, param in state_dict.items():
                        # Check if this is a weight or bias parameter
                        is_weight = "weight" in param_name
                        is_bias = "bias" in param_name
                        
                        # Only keep parameters based on configuration
                        if (is_weight and self.transfer_weights) or (is_bias and self.transfer_bias):
                            filtered_state_dict[param_name] = param
                    
                    # Replace with filtered state dict
                    knowledge[key] = filtered_state_dict
        
        # Apply layer selection
        if self.layers_to_transfer != ["all"]:
            for key in ["actor_state_dict", "critic_state_dict", 
                        "policy_state_dict", "value_state_dict"]:
                if key in knowledge:
                    state_dict = knowledge[key]
                    filtered_state_dict = {}
                    
                    for param_name, param in state_dict.items():
                        # Extract layer number/name from parameter name
                        parts = param_name.split('.')
                        if len(parts) >= 2:
                            layer_id = parts[0]
                            
                            # Check if this layer should be transferred
                            if layer_id in self.layers_to_transfer:
                                filtered_state_dict[param_name] = param
                    
                    # Replace with filtered state dict
                    knowledge[key] = filtered_state_dict
        
        # Apply layer mapping if specified
        if self.transfer_mapping:
            for key in ["actor_state_dict", "critic_state_dict", 
                        "policy_state_dict", "value_state_dict"]:
                if key in knowledge:
                    state_dict = knowledge[key]
                    mapped_state_dict = {}
                    
                    for param_name, param in state_dict.items():
                        # Check if we need to map this parameter
                        for source_pattern, target_pattern in self.transfer_mapping.items():
                            if source_pattern in param_name:
                                # Create mapped parameter name
                                mapped_name = param_name.replace(source_pattern, target_pattern)
                                mapped_state_dict[mapped_name] = param
                                break
                        else:
                            # No mapping found, keep original name
                            mapped_state_dict[param_name] = param
                    
                    # Replace with mapped state dict
                    knowledge[key] = mapped_state_dict
        
        # Apply weight scaling
        if self.weight_scaling != 1.0:
            for key in ["actor_state_dict", "critic_state_dict", 
                        "policy_state_dict", "value_state_dict"]:
                if key in knowledge:
                    state_dict = knowledge[key]
                    
                    for param_name, param in state_dict.items():
                        # Only scale weight parameters, not biases
                        if "weight" in param_name:
                            state_dict[param_name] = param * self.weight_scaling
        
        # Apply selective transfer based on threshold
        if self.threshold > 0.0:
            for key in ["actor_state_dict", "critic_state_dict", 
                        "policy_state_dict", "value_state_dict"]:
                if key in knowledge:
                    state_dict = knowledge[key]
                    
                    for param_name, param in state_dict.items():
                        # Apply threshold mask
                        if isinstance(param, torch.Tensor):
                            mask = torch.abs(param) >= self.threshold
                            # Zero out values below threshold
                            state_dict[param_name] = param * mask.float()
        
        return knowledge
    
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