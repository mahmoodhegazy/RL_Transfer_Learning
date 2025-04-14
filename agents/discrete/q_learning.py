import numpy as np
import pickle
import os
from agents.base_agent import BaseAgent
import gym

class QLearningAgent(BaseAgent):
    """Q-Learning agent implementation."""
    
    def __init__(self, env, config):
        super().__init__(env, config)
        
        # Extract environment information
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # Extract agent parameters from config
        self.learning_rate = config.get("learning_rate", 0.1)
        self.discount_factor = config.get("discount_factor", 0.99)
        self.exploration_rate = config.get("exploration_rate", 0.1)
        self.exploration_decay = config.get("exploration_decay", 0.995)
        self.min_exploration_rate = config.get("min_exploration_rate", 0.01)
        
        # Initialize Q-table based on observation and action space type
        self._initialize_q_table()
    
    def _initialize_q_table(self):
        """Initialize the Q-table based on observation and action space."""
        if hasattr(self.observation_space, 'n'):  # Discrete observation space
            # For discrete observation space, create a table of shape (num_states, num_actions)
            self.q_table = np.zeros((self.observation_space.n, self.action_space.n))
        else:  # Continuous or multi-discrete observation space
            # For continuous or complex observation spaces, use a dictionary-based approach
            self.q_table = {}
            self.discretization_bins = self.config.get("discretization_bins", 10)
    
    def _get_state_key(self, state):
        """Convert a continuous state to a discrete key for Q-table lookup."""
        if hasattr(self.observation_space, 'n'):  # Discrete observation space
            return state
        else:  # Continuous or multi-discrete observation space
            # Discretize the state to use as a key in the dictionary-based Q-table
            if isinstance(state, np.ndarray):
                # Discretize each dimension
                discretized = tuple(state)
                return discretized
            else:
                return state
    
    def select_action(self, state):
        """Select an action using epsilon-greedy policy."""
        state_key = self._get_state_key(state)
        
        # Exploration: random action
        if np.random.random() < self.exploration_rate:
            return self.action_space.sample()
        
        # Exploitation: best known action
        if hasattr(self.observation_space, 'n'):  # Discrete observation space
            return np.argmax(self.q_table[state_key])
        else:  # Continuous or multi-discrete observation space
            # If state not seen before, add it to Q-table with zeros
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_space.n)
            # Get all actions that have the maximum Q-value
            q_values = self.q_table[state_key]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            
            # Randomly choose among the best actions
            return np.random.choice(best_actions)
            
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-values using the Q-learning update rule."""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Ensure states exist in Q-table for dictionary-based approach
        if not hasattr(self.observation_space, 'n'):  # Continuous or multi-discrete
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_space.n)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_space.n)
        
        # Q-learning update rule
        current_q = self.q_table[state_key][action]
        
        # For terminal states, future_q is 0
        if done:
            future_q = 0
        else:
            future_q = np.max(self.q_table[next_state_key])
        
        # Calculate new Q-value
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * future_q - current_q
        )
        
        # Update Q-table
        self.q_table[state_key][action] = new_q
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate, 
            self.exploration_rate * self.exploration_decay
        )
        
        # Increment training steps
        self.training_steps += 1
    
    def save(self, path):
        """Save agent parameters to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'exploration_rate': self.exploration_rate,
                'training_steps': self.training_steps,
                'config': self.config
            }, f)
    
    def load(self, path):
        """Load agent parameters from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.exploration_rate = data['exploration_rate']
            self.training_steps = data['training_steps']
            # Update config if needed
            for key, value in data['config'].items():
                if key not in ['env']:  # Skip environment which might not be serializable
                    self.config[key] = value
    
    def extract_knowledge(self, knowledge_type):
        """Extract agent's knowledge for transfer learning."""
        if knowledge_type == "q_values":
            # Return the entire Q-table
            return {"q_table": self.q_table}
        elif knowledge_type == "policy":
            # Extract deterministic policy from Q-values
            policy = {}
            
            if hasattr(self.observation_space, 'n'):  # Discrete observation space
                for state in range(self.observation_space.n):
                    policy[state] = np.argmax(self.q_table[state])
            else:  # Continuous or multi-discrete observation space
                for state_key, q_values in self.q_table.items():
                    policy[state_key] = np.argmax(q_values)
            
            return {"policy": policy}
        elif knowledge_type == "parameters":
            # Return all agent parameters
            return {
                "q_table": self.q_table,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "exploration_rate": self.exploration_rate,
                "exploration_decay": self.exploration_decay,
                "min_exploration_rate": self.min_exploration_rate,
                "training_steps": self.training_steps
            }
        elif knowledge_type == "value_function":
            # Return the Q-table as the value function
            return {"value_function": self.q_table}
        else:
            raise ValueError(f"Unknown knowledge type: {knowledge_type}")
    
    def initialize_from_knowledge(self, knowledge):
        """Initialize agent using transferred knowledge."""
        if "q_table" in knowledge:
            # Initialize Q-table from transferred knowledge
            # For compatibility between different environments, we may need to transform the Q-table
            if hasattr(self.observation_space, 'n') and hasattr(self.action_space, 'n'):
                # For simple discrete spaces - ensure dimension compatibility
                source_q_table = knowledge["q_table"]
                
                if hasattr(source_q_table, 'shape'):  # Array-based Q-table
                    # If source and target dimensions match, use directly
                    if source_q_table.shape == (self.observation_space.n, self.action_space.n):
                        self.q_table = source_q_table.copy()
                    else:
                        # Otherwise, resize/interpolate
                        self._adapt_discrete_q_table(source_q_table)
                else:  # Dictionary-based Q-table
                    self._initialize_q_table()  # Reset to empty
                    # Copy over values for states that exist in both environments
                    for state_key, q_values in source_q_table.items():
                        if isinstance(state_key, tuple) and len(state_key) == self.observation_space.shape[0]:
                            # For matching state dimensions, copy directly
                            self.q_table[state_key] = q_values[:self.action_space.n].copy()
            else:  # Complex spaces - use dictionary approach
                self._initialize_q_table()  # Reset to empty
                # Set values for states that can be directly mapped
                source_q_table = knowledge["q_table"]
                if isinstance(source_q_table, dict):
                    for state_key, q_values in source_q_table.items():
                        # Only use if state dimension matches or can be adapted
                        if isinstance(state_key, tuple) and len(state_key) == self.observation_space.shape[0]:
                            self.q_table[state_key] = q_values[:self.action_space.n].copy()
        
        # Set other parameters if provided
        for param in ["learning_rate", "discount_factor", "exploration_rate", 
                      "exploration_decay", "min_exploration_rate"]:
            if param in knowledge:
                setattr(self, param, knowledge[param])
    
    def _adapt_discrete_q_table(self, source_q_table):
        """Adapt a Q-table from a source environment with different dimensions."""
        source_states, source_actions = source_q_table.shape
        target_states, target_actions = self.observation_space.n, self.action_space.n
        
        # Initialize target Q-table
        self.q_table = np.zeros((target_states, target_actions))
        
        # Simple case: fewer target states and actions (just copy the overlapping part)
        if target_states <= source_states and target_actions <= source_actions:
            self.q_table = source_q_table[:target_states, :target_actions].copy()
        
        # More complex case: more target states/actions (need to interpolate)
        else:
            # Interpolate for matching dimensions
            min_states = min(target_states, source_states)
            min_actions = min(target_actions, source_actions)
            
            # Copy what we can directly
            self.q_table[:min_states, :min_actions] = source_q_table[:min_states, :min_actions].copy()
            
            # For any extra states/actions, initialize with zeros or small random values
            if self.config.get("initialize_new_states_randomly", False):
                if target_states > min_states:
                    self.q_table[min_states:, :] = np.random.uniform(
                        0, 0.1, size=(target_states - min_states, target_actions)
                    )
                if target_actions > min_actions:
                    self.q_table[:, min_actions:] = np.random.uniform(
                        0, 0.1, size=(target_states, target_actions - min_actions)
                    )