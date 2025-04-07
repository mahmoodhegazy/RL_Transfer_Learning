import numpy as np
from environments.continuous.ant import AntEnv

class HalfAntEnv(AntEnv):
    """
    Ant environment with only half of the ant's body active.
    
    This environment deactivates either the front or back half of the ant,
    forcing it to learn to move with only half its legs.
    """
    
    def __init__(self, config):
        # Extract half type configuration
        self.half_type = config.get("half_type", "front")  # "front" or "back"
        
        super().__init__(config)
        
        # Define which joints are active based on half type
        if self.half_type == "front":
            self.active_joints = [0, 1, 2, 3]  # Front legs
        else:  # "back"
            self.active_joints = [4, 5, 6, 7]  # Back legs
        
        # Modify action space to match half ant
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(len(self.active_joints),), dtype=np.float32
        )
    
    def step(self, action):
        """Map half-ant actions to full action space."""
        full_action = np.zeros(8)  # Ant has 8 action dimensions
        
        # Map actions to active joints
        for i, joint_idx in enumerate(self.active_joints):
            full_action[joint_idx] = action[i]
        
        # Pass full action to parent class
        return super().step(full_action)