import numpy as np
import torch

class StateMapper:
    """Utility for mapping state spaces between source and target environments."""
    
    @staticmethod
    def create_mapping(source_agent, target_agent):
        """Create mapping between source and target state spaces."""
        # Get dimensions
        source_dims = source_agent.obs_dim if hasattr(source_agent, 'obs_dim') else \
                     source_agent.observation_space.shape[0]
        target_dims = target_agent.obs_dim if hasattr(target_agent, 'obs_dim') else \
                     target_agent.observation_space.shape[0]
        
        # Initialize mapping dictionary (source index -> target index)
        index_mapping = {}
        
        # For Ant environments, map observation components by semantic meaning
        segments = {
            'position': {'source_range': (0, min(3, source_dims)), 
                         'target_range': (0, min(3, target_dims))},
            'orientation': {'source_range': (3, min(7, source_dims)), 
                           'target_range': (3, min(7, target_dims))},
            'joint_positions': {'source_range': (7, min(15, source_dims)), 
                               'target_range': (7, min(15, target_dims))},
            'joint_velocities': {'source_range': (15, min(23, source_dims)), 
                                'target_range': (15, min(23, target_dims))},
            'additional': {'source_range': (23, source_dims), 
                          'target_range': (23, target_dims)}
        }
        
        # Create mapping for each segment
        for segment, ranges in segments.items():
            s_start, s_end = ranges['source_range']
            t_start, t_end = ranges['target_range']
            
            # Skip if source doesn't have this segment
            if s_start >= source_dims:
                continue
                
            # Map available dimensions
            shared_length = min(s_end - s_start, t_end - t_start)
            for i in range(shared_length):
                index_mapping[s_start + i] = t_start + i
        
        return index_mapping
    
    @staticmethod
    def map_network_weights(source_weights, target_weights, mapping):
        """Map weights from source to target network using the mapping."""
        # Clone target weights to avoid modifying the original
        mapped_weights = target_weights.clone()
        
        # Apply mapping to input weights
        for source_idx, target_idx in mapping.items():
            if source_idx < source_weights.shape[1] and target_idx < mapped_weights.shape[1]:
                # Map input dimension weights
                mapped_weights[:, target_idx] = source_weights[:, source_idx]
        
        return mapped_weights
    
    @staticmethod
    def get_joint_indices(reduced_dof=False, planar=False):
        """Get indices of active joints based on environment type."""
        all_joints = list(range(8))  # All 8 joints in standard Ant
        
        if reduced_dof:
            return all_joints[:4]  # First 4 joints (front legs)
        elif planar:
            # Joints that control movement in the XY plane
            return [0, 2, 4, 6]  # Horizontal joints
        else:
            return all_joints