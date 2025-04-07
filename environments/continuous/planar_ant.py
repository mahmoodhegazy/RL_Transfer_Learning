import numpy as np
from environments.continuous.ant import AntEnv

class PlanarAntEnv(AntEnv):
    """
    Ant environment restricted to 2D planar movement.
    
    This environment constrains the ant to move only in the X-Y plane,
    simplifying the control problem.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Modify the base environment to constrain Z movement
        self.constrain_z = True
    
    def modify_env_parameters(self):
        """Add planar constraints to the environment."""
        super().modify_env_parameters()
        
        if hasattr(self.base_env.unwrapped, "model"):
            # Set a very high friction in Z direction to constrain movement
            model = self.base_env.unwrapped.model
            for i in range(model.geom_friction.shape[0]):
                # Increase vertical friction significantly
                model.geom_friction[i, 2] = 10.0
    
    def step(self, action):
        """Take a step and then enforce planar constraints."""
        observation, reward, terminated, truncated, info = super().step(action)
        
        # If we can directly modify the state, constrain Z position and velocity
        if hasattr(self.base_env.unwrapped, "data"):
            data = self.base_env.unwrapped.data
            
            # Find torso position (typically the 2nd body)
            torso_idx = 1  # This depends on the specific ant model structure
            
            # Reset Z position to a fixed height and zero Z velocity
            if self.constrain_z:
                data.qpos[2] = 0.5  # Fixed Z height
                data.qvel[2] = 0    # Zero Z velocity
        
        return observation, reward, terminated, truncated, info