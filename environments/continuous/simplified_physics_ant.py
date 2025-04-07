from environments.continuous.ant import AntEnv

class SimplifiedPhysicsAntEnv(AntEnv):
    """
    Ant environment with simplified physics.
    
    Simplifications include:
    - Reduced friction
    - Modified gravity
    - No joint damping
    - Higher control frequency for easier control
    """
    
    def __init__(self, config):
        # Set default simplifications
        simplified_config = config.copy()
        simplified_config.setdefault("friction", 0.8)  # Lower friction
        simplified_config.setdefault("gravity", -9.0)  # Reduced gravity
        simplified_config.setdefault("control_frequency", 10)  # Higher control frequency
        
        super().__init__(simplified_config)
    
    def modify_env_parameters(self):
        """Apply simplified physics parameters."""
        super().modify_env_parameters()
        
        # Simplify additional physics parameters
        model = self.base_env.unwrapped.model
        
        # Remove joint damping
        for i in range(model.dof_damping.shape[0]):
            model.dof_damping[i] = 0.2  # Reduced damping
        
        # Make contacts softer
        for i in range(model.geom_margin.shape[0]):
            model.geom_margin[i] = 0.01  # Increased contact margin