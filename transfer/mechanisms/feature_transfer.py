import numpy as np
import torch

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
        
        # Set default parameters
        self.layers_to_transfer = config.get("layers_to_transfer", ["all"])
        self.freeze_transferred = config.get("freeze_transferred", False)
        self.adaptation_method = config.get("adaptation_method", "truncate")
        self.fine_tuning_lr = config.get("fine_tuning_lr", 0.0001)
        
        # Device for torch operations if needed
        self.device = torch.device("cuda" if torch.cuda.is_available() and 
                                   torch.cuda.is_available() else "cpu")
    
    def transfer(self, source_agent, target_agent):
        """Transfer feature extraction layers from source to target agent."""
        # Extract feature knowledge from source agent
        knowledge = source_agent.extract_knowledge("feature_extractor")
        
        # Process the knowledge if needed
        processed_knowledge = self._process_knowledge(knowledge)
        
        # Initialize target agent with processed knowledge
        target_agent.initialize_from_knowledge(processed_knowledge)
        
        # If we need to freeze transferred layers
        if self.freeze_transferred and hasattr(target_agent, 'policy') and hasattr(target_agent.policy, 'requires_grad_'):
            # For neural network agents
            self._freeze_layers(target_agent)
        
        return target_agent
    
    def _process_knowledge(self, knowledge):
        """Process the extracted feature knowledge before transfer."""
        # Make a copy to avoid modifying the original
        processed_knowledge = knowledge.copy()
        
        # If we need to adapt feature dimensions
        if "actor_features" in knowledge and "critic_features" in knowledge:
            # For actor-critic agents
            if self.adaptation_method != "direct":
                processed_knowledge = self._adapt_network_features(processed_knowledge)
        
        # For specific layer selection
        if self.layers_to_transfer != ["all"]:
            processed_knowledge = self._select_layers(processed_knowledge)
        
        return processed_knowledge
    
    def _adapt_network_features(self, knowledge):
        """Adapt feature dimensions if source and target have different shapes."""
        # This could include padding, truncation, or interpolation
        # of neural network weights
        
        for key in ["actor_features", "critic_features"]:
            if key in knowledge:
                for layer_name, layer_params in knowledge[key].items():
                    if isinstance(layer_params, torch.Tensor):
                        # Apply adaptation based on configuration
                        if self.adaptation_method == "truncate":
                            # Truncate to smaller dimension (may lose information)
                            target_shape = self.config.get(f"{key}_shapes", {}).get(layer_name)
                            if target_shape and list(layer_params.shape) != target_shape:
                                # Truncate to target shape
                                slices = tuple(slice(0, min(s, t)) for s, t in zip(layer_params.shape, target_shape))
                                knowledge[key][layer_name] = layer_params[slices]
                                
                        elif self.adaptation_method == "pad":
                            # Pad with zeros to larger dimension
                            target_shape = self.config.get(f"{key}_shapes", {}).get(layer_name)
                            if target_shape and list(layer_params.shape) != target_shape:
                                # Create new tensor of target shape
                                new_params = torch.zeros(target_shape, device=layer_params.device)
                                # Copy values from original tensor
                                slices = tuple(slice(0, min(s, t)) for s, t in zip(layer_params.shape, target_shape))
                                new_params[slices] = layer_params[slices]
                                knowledge[key][layer_name] = new_params
                                
                        elif self.adaptation_method == "interpolate":
                            # Only for 2D weight matrices
                            if len(layer_params.shape) == 2:
                                target_shape = self.config.get(f"{key}_shapes", {}).get(layer_name)
                                if target_shape and list(layer_params.shape) != target_shape:
                                    # Use interpolation for 2D weights
                                    # Reshape to 4D for torch's interpolate function
                                    reshaped = layer_params.unsqueeze(0).unsqueeze(0)
                                    interpolated = torch.nn.functional.interpolate(
                                        reshaped, 
                                        size=target_shape,
                                        mode='bilinear',
                                        align_corners=True
                                    )
                                    knowledge[key][layer_name] = interpolated.squeeze(0).squeeze(0)
        
        return knowledge
    
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
        if not hasattr(agent, 'policy'):
            return
        
        # Freeze actor features if present
        if hasattr(agent.policy, 'layers'):
            for name, param in agent.policy.layers.named_parameters():
                layer_id = name.split('.')[0] if '.' in name else name
                if layer_id in self.layers_to_transfer or "all" in self.layers_to_transfer:
                    param.requires_grad = False
        
        # Freeze critic features if present
        if hasattr(agent, 'critic') and hasattr(agent.critic, 'layers'):
            for name, param in agent.critic.layers.named_parameters():
                layer_id = name.split('.')[0] if '.' in name else name
                if layer_id in self.layers_to_transfer or "all" in self.layers_to_transfer:
                    param.requires_grad = False