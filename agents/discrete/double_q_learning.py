import numpy as np
import pickle
import os
from agents.base_agent import BaseAgent

class DoubleQLearningAgent(BaseAgent):
    """Double Q-Learning agent implementation."""
    
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
        
        # Initialize two Q-tables for double Q-learning
        self._initialize_q_tables()
    
    def _initialize_q_tables(self):
        """Initialize the two Q-tables based on observation and action space."""
        if hasattr(self.observation_space, 'n'):  # Discrete observation space
            # For discrete observation space, create two tables of shape (num_states, num_actions)
            self.q_table_a = np.zeros((self.observation_space.n, self.action_space.n))
            self.q_table_b = np.zeros((self.observation_space.n, self.action_space.n))
        else:  # Continuous or multi-discrete observation space
            # For continuous or complex observation spaces, use dictionary-based approach
            self.q_table_a = {}
            self.q_table_b = {}
            self.discretization_bins = self.config.get("discretization_bins", 10)
    
    def _get_state_key(self, state):
        """Convert a continuous state to a discrete key for Q-table lookup."""
        if hasattr(self.observation_space, 'n'):  # Discrete observation space
            return state
        else:  # Continuous or multi-discrete observation space
            # Discretize the state to use as a key in the dictionary-based Q-table
            if isinstance(state, np.ndarray):
                # Discretize each dimension
                discretized = tuple(np.digitize(
                    s, np.linspace(low, high, self.discretization_bins)
                ) for s, (low, high) in zip(state, zip(
                    self.observation_space.low, self.observation_space.high
                )))
                return discretized
            else:
                return state
    
    def select_action(self, state):
        """Select an action using epsilon-greedy policy based on average Q-values."""
        state_key = self._get_state_key(state)
        
        # Exploration: random action
        if np.random.random() < self.exploration_rate:
            return self.action_space.sample()
        
        # Exploitation: best known action based on the average of both Q-tables
        if hasattr(self.observation_space, 'n'):  # Discrete observation space
            avg_q_values = (self.q_table_a[state_key] + self.q_table_b[state_key]) / 2
            return np.argmax(avg_q_values)
        else:  # Continuous or multi-discrete observation space
            # If state not seen before, add it to both Q-tables with zeros
            if state_key not in self.q_table_a:
                self.q_table_a[state_key] = np.zeros(self.action_space.n)
            if state_key not in self.q_table_b:
                self.q_table_b[state_key] = np.zeros(self.action_space.n)
            
            avg_q_values = (self.q_table_a[state_key] + self.q_table_b[state_key]) / 2
            return np.argmax(avg_q_values)
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-values using the Double Q-learning update rule."""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Ensure states exist in Q-tables for dictionary-based approach
        if not hasattr(self.observation_space, 'n'):  # Continuous or multi-discrete
            if state_key not in self.q_table_a:
                self.q_table_a[state_key] = np.zeros(self.action_space.n)
            if state_key not in self.q_table_b:
                self.q_table_b[state_key] = np.zeros(self.action_space.n)
            if next_state_key not in self.q_table_a:
                self.q_table_a[next_state_key] = np.zeros(self.action_space.n)
            if next_state_key not in self.q_table_b:
                self.q_table_b[next_state_key] = np.zeros(self.action_space.n)
        
        # Randomly choose which Q-table to update (with 0.5 probability each)
        if np.random.random() < 0.5:
            # Update Q-table A
            current_q = self.q_table_a[state_key][action]
            
            if done:
                future_q = 0
            else:
                # Use Q-table A to select action, but Q-table B to evaluate it
                best_action = np.argmax(self.q_table_a[next_state_key])
                future_q = self.q_table_b[next_state_key][best_action]
            
            # Update Q-table A
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * future_q - current_q
            )
            self.q_table_a[state_key][action] = new_q
            
        else:
            # Update Q-table B
            current_q = self.q_table_b[state_key][action]
            
            if done:
                future_q = 0
            else:
                # Use Q-table B to select action, but Q-table A to evaluate it
                best_action = np.argmax(self.q_table_b[next_state_key])
                future_q = self.q_table_a[next_state_key][best_action]
            
            # Update Q-table B
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * future_q - current_q
            )
            self.q_table_b[state_key][action] = new_q
        
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
                'q_table_a': self.q_table_a,
                'q_table_b': self.q_table_b,
                'exploration_rate': self.exploration_rate,
                'training_steps': self.training_steps,
                'config': self.config
            }, f)
    
    def load(self, path):
        """Load agent parameters from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.q_table_a = data['q_table_a']
            self.q_table_b = data['q_table_b']
            self.exploration_rate = data['exploration_rate']
            self.training_steps = data['training_steps']
            # Update config if needed
            for key, value in data['config'].items():
                if key not in ['env']:  # Skip environment which might not be serializable
                    self.config[key] = value
    
    def extract_knowledge(self, knowledge_type):
        """Extract agent's knowledge for transfer learning."""
        if knowledge_type == "q_values":
            # Return both Q-tables
            return {
                "q_table_a": self.q_table_a,
                "q_table_b": self.q_table_b
            }
        elif knowledge_type == "avg_q_values":
            # Return the average of both Q-tables
            if hasattr(self.observation_space, 'n'):  # Discrete observation space
                avg_q_table = (self.q_table_a + self.q_table_b) / 2
                return {"q_table": avg_q_table}
            else:  # Continuous or multi-discrete observation space
                avg_q_table = {}
                # Calculate average Q-values for each state
                all_states = set(list(self.q_table_a.keys()) + list(self.q_table_b.keys()))
                for state_key in all_states:
                    q_values_a = self.q_table_a.get(state_key, np.zeros(self.action_space.n))
                    q_values_b = self.q_table_b.get(state_key, np.zeros(self.action_space.n))
                    avg_q_table[state_key] = (q_values_a + q_values_b) / 2
                
                return {"q_table": avg_q_table}
        elif knowledge_type == "policy":
            # Extract deterministic policy from average Q-values
            policy = {}
            
            if hasattr(self.observation_space, 'n'):  # Discrete observation space
                avg_q_table = (self.q_table_a + self.q_table_b) / 2
                for state in range(self.observation_space.n):
                    policy[state] = np.argmax(avg_q_table[state])
            else:  # Continuous or multi-discrete observation space
                all_states = set(list(self.q_table_a.keys()) + list(self.q_table_b.keys()))
                for state_key in all_states:
                    q_values_a = self.q_table_a.get(state_key, np.zeros(self.action_space.n))
                    q_values_b = self.q_table_b.get(state_key, np.zeros(self.action_space.n))
                    avg_q_values = (q_values_a + q_values_b) / 2
                    policy[state_key] = np.argmax(avg_q_values)
            
            return {"policy": policy}
        elif knowledge_type == "parameters":
            # Return all agent parameters
            return {
                "q_table_a": self.q_table_a,
                "q_table_b": self.q_table_b,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "exploration_rate": self.exploration_rate,
                "exploration_decay": self.exploration_decay,
                "min_exploration_rate": self.min_exploration_rate,
                "training_steps": self.training_steps
            }
        else:
            raise ValueError(f"Unknown knowledge type: {knowledge_type}")
    
    def initialize_from_knowledge(self, knowledge):
        """Initialize agent using transferred knowledge."""
        # Initialize Q-tables from transferred knowledge
        for table_name in ["q_table_a", "q_table_b", "q_table"]:
            if table_name in knowledge:
                source_q_table = knowledge[table_name]
                
                # Handle avg_q_values case (single table to be copied to both A and B)
                if table_name == "q_table":
                    if hasattr(self.observation_space, 'n') and hasattr(source_q_table, 'shape'):
                        # Adapt dimensions if necessary
                        if source_q_table.shape == (self.observation_space.n, self.action_space.n):
                            self.q_table_a = source_q_table.copy()
                            self.q_table_b = source_q_table.copy()
                        else:
                            self._adapt_discrete_q_table(source_q_table, "both")
                    else:  # Dictionary-based
                        self._initialize_q_tables()  # Reset to empty
                        for state_key, q_values in source_q_table.items():
                            if isinstance(state_key, tuple) and len(state_key) == self.observation_space.shape[0]:
                                self.q_table_a[state_key] = q_values[:self.action_space.n].copy()
                                self.q_table_b[state_key] = q_values[:self.action_space.n].copy()
                    # Skip the rest of initialization since we've set both tables
                    break
                
                # Handle individual Q-table transfers
                target_table = self.q_table_a if table_name == "q_table_a" else self.q_table_b
                
                if hasattr(self.observation_space, 'n') and hasattr(source_q_table, 'shape'):
                    # Adapt dimensions if necessary
                    if source_q_table.shape == (self.observation_space.n, self.action_space.n):
                        if table_name == "q_table_a":
                            self.q_table_a = source_q_table.copy()
                        else:
                            self.q_table_b = source_q_table.copy()
                    else:
                        self._adapt_discrete_q_table(source_q_table, table_name)
                else:  # Dictionary-based
                    if table_name == "q_table_a":
                        self.q_table_a = {}
                    else:
                        self.q_table_b = {}
                        
                    for state_key, q_values in source_q_table.items():
                        if isinstance(state_key, tuple) and len(state_key) == self.observation_space.shape[0]:
                            if table_name == "q_table_a":
                                self.q_table_a[state_key] = q_values[:self.action_space.n].copy()
                            else:
                                self.q_table_b[state_key] = q_values[:self.action_space.n].copy()
        
        # Set other parameters if provided
        for param in ["learning_rate", "discount_factor", "exploration_rate", 
                      "exploration_decay", "min_exploration_rate"]:
            if param in knowledge:
                setattr(self, param, knowledge[param])
    
    def _adapt_discrete_q_table(self, source_q_table, target="both"):
        """Adapt a Q-table from a source environment with different dimensions."""
        source_states, source_actions = source_q_table.shape
        target_states, target_actions = self.observation_space.n, self.action_space.n
        
        # Initialize target Q-table(s)
        if target == "q_table_a" or target == "both":
            self.q_table_a = np.zeros((target_states, target_actions))
        if target == "q_table_b" or target == "both":
            self.q_table_b = np.zeros((target_states, target_actions))
        
        # Simple case: fewer target states and actions (just copy the overlapping part)
        if target_states <= source_states and target_actions <= source_actions:
            if target == "q_table_a" or target == "both":
                self.q_table_a = source_q_table[:target_states, :target_actions].copy()
            if target == "q_table_b" or target == "both":
                self.q_table_b = source_q_table[:target_states, :target_actions].copy()
        
        # More complex case: more target states/actions (need to interpolate)
        else:
            # Interpolate for matching dimensions
            min_states = min(target_states, source_states)
            min_actions = min(target_actions, source_actions)
            
            # Copy what we can directly
            if target == "q_table_a" or target == "both":
                self.q_table_a[:min_states, :min_actions] = source_q_table[:min_states, :min_actions].copy()
            if target == "q_table_b" or target == "both":
                self.q_table_b[:min_states, :min_actions] = source_q_table[:min_states, :min_actions].copy()
            
            # For any extra states/actions, initialize with zeros or small random values
            if self.config.get("initialize_new_states_randomly", False):
                if target_states > min_states:
                    random_values = np.random.uniform(
                        0, 0.1, size=(target_states - min_states, target_actions)
                    )
                    if target == "q_table_a" or target == "both":
                        self.q_table_a[min_states:, :] = random_values
                    if target == "q_table_b" or target == "both":
                        self.q_table_b[min_states:, :] = random_values
                
                if target_actions > min_actions:
                    random_values = np.random.uniform(
                        0, 0.1, size=(target_states, target_actions - min_actions)
                    )
                    if target == "q_table_a" or target == "both":
                        self.q_table_a[:, min_actions:] = random_values
                    if target == "q_table_b" or target == "both":
                        self.q_table_b[:, min_actions:] = random_values