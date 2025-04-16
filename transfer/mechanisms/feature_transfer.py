import numpy as np
import torch
from ..utils.state_mapping import StateMapper

class FeatureTransfer:
    """Transfer learning by extracting and transferring feature representations."""
    
    def __init__(self, config):
        self.config = config
        # Configuration options:
        # - layers_to_transfer: List of layer names or indices to transfer
        # - freeze_transferred: Whether to freeze transferred layers
        # - adaptation_method: How to adapt feature dimensions if needed
        #   (e.g., "interpolate", "pad", "truncate")
        # - fine_tuning_lr: Learning rate for fine-tuning if needed
        
        # Configuration options
        self.layers_to_transfer = config.get("layers_to_transfer", ["all"])
        self.freeze_transferred = config.get("freeze_transferred", False)
        self.adaptation_method = config.get("adaptation_method", "truncate")
        self.fine_tuning_lr = config.get("fine_tuning_lr", 0.0001)
        self.use_state_mapping = config.get("use_state_mapping", True)
        
        # Device for torch operations
        self.device = torch.device("cuda" if torch.cuda.is_available() and 
                                   torch.cuda.is_available() else "cpu")
    
    def transfer(self, source_agent, target_agent):
        """Transfer feature extraction layers from source to target agent."""
        print("Starting feature transfer...")
        
        # Extract feature knowledge from source agent
        knowledge = source_agent.extract_knowledge("feature_extractor")
        
        # Create state mapping between environments if needed
        if self.use_state_mapping:
            self.state_mapping = StateMapper.create_mapping(source_agent, target_agent)
            print(f"Created state mapping with {len(self.state_mapping)} mapped dimensions")
        
        # Process and adapt the knowledge
        processed_knowledge = self._process_knowledge(knowledge, source_agent, target_agent)
        
        # Initialize target agent with processed knowledge
        target_agent.initialize_from_knowledge(processed_knowledge)
        
        # If we need to freeze transferred layers
        if self.freeze_transferred:
            self._freeze_layers(target_agent)
            
        print("Feature transfer completed")
        return target_agent
    
    def _process_knowledge(self, knowledge, source_agent, target_agent):
        """Process the extracted feature knowledge before transfer."""
        # Make a copy to avoid modifying the original
        processed_knowledge = knowledge.copy()
        
        # Process feature layers
        for key in ["actor_features", "critic_features"]:
            if key in knowledge:
                processed_features = {}
                source_features = knowledge[key]
                
                # Process each layer
                for layer_name, layer_params in source_features.items():
                    # Skip layers that shouldn't be transferred
                    if not self._should_transfer_layer(layer_name):
                        continue
                    
                    # Special handling for first layer (input layer)
                    if ".0.weight" in layer_name or "layers.0.weight" in layer_name:
                        # This is the input layer - apply state mapping
                        if self.use_state_mapping:
                            if key == "actor_features":
                                target_weight = target_agent.actor.layers[0].weight.data
                            else:  # critic_features
                                target_weight = target_agent.critic.layers[0].weight.data
                            
                            mapped_weight = StateMapper.map_network_weights(
                                layer_params, target_weight, self.state_mapping
                            )
                            processed_features[layer_name] = mapped_weight
                        else:
                            # Apply adaptation based on configuration
                            if self.adaptation_method == "truncate":
                                if key == "actor_features":
                                    target_shape = target_agent.actor.layers[0].weight.shape
                                else:  # critic_features
                                    target_shape = target_agent.critic.layers[0].weight.shape
                                
                                # Apply truncation
                                if list(layer_params.shape) != list(target_shape):
                                    processed_features[layer_name] = self._truncate_weights(
                                        layer_params, target_shape
                                    )
                                else:
                                    processed_features[layer_name] = layer_params.clone()
                                    
                            elif self.adaptation_method == "pad":
                                if key == "actor_features":
                                    target_shape = target_agent.actor.layers[0].weight.shape
                                else:  # critic_features
                                    target_shape = target_agent.critic.layers[0].weight.shape
                                
                                # Apply padding
                                if list(layer_params.shape) != list(target_shape):
                                    processed_features[layer_name] = self._pad_weights(
                                        layer_params, target_shape
                                    )
                                else:
                                    processed_features[layer_name] = layer_params.clone()
                    else:
                        # For other layers, handle based on adaptation method
                        if key == "actor_features":
                            target_shape = None
                            for name, param in target_agent.actor.named_parameters():
                                if name == layer_name:
                                    target_shape = param.shape
                                    break
                        else:  # critic_features
                            target_shape = None
                            for name, param in target_agent.critic.named_parameters():
                                if name == layer_name:
                                    target_shape = param.shape
                                    break
                        
                        # If we found a matching layer, apply adaptation
                        if target_shape is not None and list(layer_params.shape) != list(target_shape):
                            if "weight" in layer_name:
                                if self.adaptation_method == "truncate":
                                    processed_features[layer_name] = self._truncate_weights(
                                        layer_params, target_shape
                                    )
                                elif self.adaptation_method == "pad":
                                    processed_features[layer_name] = self._pad_weights(
                                        layer_params, target_shape
                                    )
                            elif "bias" in layer_name:
                                # For bias vectors, simpler adaptation
                                processed_bias = torch.zeros(target_shape)
                                min_dim = min(layer_params.shape[0], target_shape[0])
                                processed_bias[:min_dim] = layer_params[:min_dim]
                                processed_features[layer_name] = processed_bias
                        else:
                            # No adaptation needed or no matching layer found
                            processed_features[layer_name] = layer_params.clone() if isinstance(layer_params, torch.Tensor) else layer_params
                
                # Update processed knowledge
                processed_knowledge[key] = processed_features
        
        # For specific layer selection
        if self.layers_to_transfer != ["all"]:
            processed_knowledge = self._select_layers(processed_knowledge)
        
        return processed_knowledge
    
    def _truncate_weights(self, weights, target_shape):
        """Truncate weights to target shape."""
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)
        
        result = torch.zeros(target_shape, device=weights.device)
        
        # For 2D weight matrices (typical in neural networks)
        if len(weights.shape) == 2 and len(target_shape) == 2:
            # Get minimum dimensions
            min_rows = min(weights.shape[0], target_shape[0])
            min_cols = min(weights.shape[1], target_shape[1])
            
            # Copy overlapping portion
            result[:min_rows, :min_cols] = weights[:min_rows, :min_cols]
        
        # For 1D bias vectors
        elif len(weights.shape) == 1 and len(target_shape) == 1:
            min_dim = min(weights.shape[0], target_shape[0])
            result[:min_dim] = weights[:min_dim]
        
        return result
    
    def _pad_weights(self, weights, target_shape):
        """Pad weights to target shape."""
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)
        
        result = torch.zeros(target_shape, device=weights.device)
        
        # For 2D weight matrices
        if len(weights.shape) == 2 and len(target_shape) == 2:
            # Get minimum dimensions
            min_rows = min(weights.shape[0], target_shape[0])
            min_cols = min(weights.shape[1], target_shape[1])
            
            # Copy overlapping portion
            result[:min_rows, :min_cols] = weights[:min_rows, :min_cols]
            
            # Fill remaining areas with small random values
            if min_rows < target_shape[0] or min_cols < target_shape[1]:
                mean_val = weights.mean().item()
                std_val = weights.std().item() * 0.1  # Reduced variance for stability
                
                if min_rows < target_shape[0]:
                    result[min_rows:, :min_cols].normal_(mean=mean_val, std=std_val)
                
                if min_cols < target_shape[1]:
                    result[:, min_cols:].normal_(mean=mean_val, std=std_val)
        
        # For 1D bias vectors
        elif len(weights.shape) == 1 and len(target_shape) == 1:
            min_dim = min(weights.shape[0], target_shape[0])
            result[:min_dim] = weights[:min_dim]
            
            # Fill remaining areas
            if min_dim < target_shape[0]:
                mean_val = weights.mean().item()
                std_val = weights.std().item() * 0.1
                result[min_dim:].normal_(mean=mean_val, std=std_val)
        
        return result
    
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
    
    def _select_layers(self, knowledge):
        """Select specific layers to transfer."""
        selected_knowledge = {}
        
        for key in knowledge:
            # Skip non-feature keys
            if key in ["actor_features", "critic_features"]:
                selected_knowledge[key] = {}
                
                for layer_name, layer_params in knowledge[key].items():
                    # Check if this layer should be transferred
                    layer_id = layer_name.split('.')[0] if '.' in layer_name else layer_name
                    if layer_id in self.layers_to_transfer or "all" in self.layers_to_transfer:
                        selected_knowledge[key][layer_name] = layer_params
            else:
                # Keep other keys (like configuration)
                selected_knowledge[key] = knowledge[key]
        
        return selected_knowledge
    
    def _freeze_layers(self, agent):
        """Freeze the transferred layers to prevent updates during training."""
        # This only works for PyTorch-based agents
        if not hasattr(agent, 'actor'):
            return
        
        # Freeze actor features if present
        if hasattr(agent.actor, 'layers'):
            for name, param in agent.actor.layers.named_parameters():
                layer_id = name.split('.')[0] if '.' in name else name
                if layer_id in self.layers_to_transfer or "all" in self.layers_to_transfer:
                    param.requires_grad = False
        
        # Freeze critic features if present
        if hasattr(agent, 'critic') and hasattr(agent.critic, 'layers'):
            for name, param in agent.critic.layers.named_parameters():
                layer_id = name.split('.')[0] if '.' in name else name
                if layer_id in self.layers_to_transfer or "all" in self.layers_to_transfer:
                    param.requires_grad = False