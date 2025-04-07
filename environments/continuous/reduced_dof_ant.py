import numpy as np
from environments.continuous.ant import AntEnv

class ReducedDOFAntEnv(AntEnv):
    """
    Ant environment with reduced degrees of freedom.
    
    Simplifications include:
    - Fewer active joints (some joints are fixed)
    - Symmetric control (same action applied to symmetric joints)
    """
    
    def __init__(self, config):
        # Extract reduced DOF configuration
        self.active_joints = config.get("active_joints", [0, 1, 2, 3])  # Indices of active joints
        self.symmetric_control = config.get("symmetric_control", False)
        
        super().__init__(config)
        
        # Modify action space to match reduced DOF
        if self.symmetric_control:
            # For symmetric control, we'll use one action per leg pair
            n_pairs = len(self.active_joints) // 2
            self.action_space = spaces.Box(
                low=-1, high=1, shape=(n_pairs,), dtype=np.float32
            )
        else:
            # Otherwise, one action per active joint
            self.action_space = spaces.Box(
                low=-1, high=1, shape=(len(self.active_joints),), dtype=np.float32
            )
    
    def step(self, action):
        """Map reduced actions to full action space."""
        full_action = np.zeros(8)  # Ant has 8 action dimensions (8 joints)
        
        if self.symmetric_control:
            # Convert symmetric actions to full actions
            for i, joint_idx in enumerate(self.active_joints):
                pair_idx = i // 2
                full_action[joint_idx] = action[pair_idx]
        else:
            # Map actions to active joints
            for i, joint_idx in enumerate(self.active_joints):
                full_action[joint_idx] = action[i]
        
        # Pass full action to parent class
        return super().step(full_action)