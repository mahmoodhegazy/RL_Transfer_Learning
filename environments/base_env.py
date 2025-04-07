class BaseEnvironment:
    """Abstract base class for all environments."""
    
    def __init__(self, config):
        self.config = config
        
    def reset(self):
        """Reset environment to initial state."""
        raise NotImplementedError
        
    def step(self, action):
        """Take a step in environment with given action."""
        raise NotImplementedError
        
    def get_state_representation(self):
        """Get the state representation suitable for transfer learning."""
        raise NotImplementedError
    
    @property
    def complexity_level(self):
        """Return the complexity level of this environment."""
        return self.config.get("complexity_level", 1.0)
    
    def increase_complexity(self, increment=0.1):
        """Increase the environment complexity."""
        raise NotImplementedError