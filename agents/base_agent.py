class BaseAgent:
    """Abstract base class for all agents."""
    
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.training_steps = 0
        
    def select_action(self, state):
        """Select an action based on current state."""
        raise NotImplementedError
        
    def update(self, state, action, reward, next_state, done):
        """Update agent's knowledge after interaction."""
        raise NotImplementedError
        
    def save(self, path):
        """Save agent's parameters."""
        raise NotImplementedError
        
    def load(self, path):
        """Load agent's parameters."""
        raise NotImplementedError
    
    def extract_knowledge(self, knowledge_type):
        """Extract agent's knowledge for transfer."""
        raise NotImplementedError
    
    def initialize_from_knowledge(self, knowledge):
        """Initialize agent using transferred knowledge."""
        raise NotImplementedError