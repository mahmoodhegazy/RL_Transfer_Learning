import gymnasium as gym
import numpy as np
from gymnasium import spaces
from environments.base_env import BaseEnvironment

class TaxiEnv(BaseEnvironment):
    """
    Full Taxi environment with customizable complexity levels.
    
    Complexity can be adjusted through:
    - Grid size
    - Number of passengers
    - Number of destinations
    - Fuel constraints (optional)
    - Stochastic transitions (optional)
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Extract configuration parameters
        self.grid_size = config.get("grid_size", 5)
        self.num_passengers = config.get("num_passengers", 1)
        self.max_fuel = config.get("max_fuel", float('inf'))  # Infinite by default
        self.stochasticity = config.get("stochasticity", 0.0)  # Deterministic by default
        
        # Define the grid world
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Generate passenger locations and destinations
        self.passenger_locations = None
        self.destination_locations = None
        
        # Define action space (North, South, East, West, Pickup, Dropoff)
        self.action_space = spaces.Discrete(6)
        
        # Define observation space
        # State: (taxi_row, taxi_col, passenger1_status, passenger1_destination, passenger2_status, ...)
        # passenger_status: -1 = not picked up, 0 = in taxi, 1 = delivered
        obs_size = 2 + 2 * self.num_passengers  # taxi position + passenger statuses and destinations
        self.observation_space = spaces.MultiDiscrete([
            self.grid_size,                      # taxi row
            self.grid_size,                      # taxi col
            *([3] * self.num_passengers),        # passenger status
            *([self.grid_size * self.grid_size] * self.num_passengers) , # destination locations
            self.grid_size,                      # passenger row
            self.grid_size,                      # passenger col
        ])
        
        # Initialize state
        self.state = None
        self.remaining_fuel = None
        self.done = False
        self.reward = 0
        
        # Rewards
        self.step_penalty = config.get("step_penalty", -1)
        self.dropoff_reward = config.get("dropoff_reward", 20)
        self.fuel_penalty = config.get("fuel_penalty", -10)  # Penalty for running out of fuel
        self.illegal_action_penalty = config.get("illegal_action_penalty", -10)

    # Add this as a class variable
    FIXED_LOCATIONS = [
        (0, 0),  # R(ed)    - Top left
        (0, 4),  # G(reen)  - Top right
        (1, 2),  # Y(ellow) - Bottom left
        (4, 4)   # B(lue)   - Bottom right If this is enabled, learning is very noisy
    ]

    def _generate_locations(self):
        """Generate passenger pickup and destination locations from fixed locations."""
        # Clear previous locations
        self.passenger_locations = None
        self.destination_locations = None
        
        # For each passenger
        for _ in range(self.num_passengers):
            # Pick random pickup and dropoff locations from fixed points
            pickup_idx = np.random.randint(len(self.FIXED_LOCATIONS))
            dropoff_idx = np.random.randint(len(self.FIXED_LOCATIONS))
            
            self.passenger_locations = self.FIXED_LOCATIONS[pickup_idx]
            self.destination_locations = self.FIXED_LOCATIONS[dropoff_idx]     

    def reset(self):
        """Reset the environment to initial state."""
        # Reset taxi to random position
        taxi_row = np.random.randint(0, self.grid_size)
        taxi_col = np.random.randint(0, self.grid_size)
        
        self._generate_locations()
        
        # Initialize passenger status (not picked up)
        passenger_status = -1
        
        # Convert destination location to flattened index, 
        dest_row, dest_col = self.destination_locations
        dest_idx = dest_row * self.grid_size + dest_col
        
        # Correct state format: [taxi_row, taxi_col, passenger_status, dest_idx]
        self.state = [taxi_row, taxi_col, passenger_status, dest_idx, self.passenger_locations[0], self.passenger_locations[1]]
        
        # Reset done flag
        self.done = False
        
        return self._get_obs()
    
    def step(self, action):
        """Take a step in the environment with the given action."""
        taxi_row, taxi_col = self.state[0], self.state[1]
        passenger_status = self.state[2]
        dest_idx = self.state[3]
        
        # Convert destination index back to coordinates
        dest_row, dest_col = dest_idx // self.grid_size, dest_idx % self.grid_size
 
        
        # Apply stochasticity if configured
        if np.random.random() < self.stochasticity:
            action = np.random.randint(0, 6)  # Random action
        
        # Default step penalty
        reward = self.step_penalty
        
        # Apply action
        if action == 0:  # North
            taxi_row = max(0, taxi_row - 1)
        elif action == 1:  # South
            taxi_row = min(self.grid_size - 1, taxi_row + 1)
        elif action == 2:  # East
            taxi_col = min(self.grid_size - 1, taxi_col + 1)
        elif action == 3:  # West
            taxi_col = max(0, taxi_col - 1)
        elif action == 4:  # Pickup
            if (taxi_row, taxi_col) == tuple(self.state[-2:]) and passenger_status == -1:
                self.state[2] = 0
            else:
                reward = self.illegal_action_penalty
        elif action == 5:  # Dropoff
            if passenger_status == 0 and taxi_row == dest_row and taxi_col == dest_col:
                self.state[2] = 1  # Delivered
                reward = self.dropoff_reward
            else:
                reward = self.illegal_action_penalty
  
        # Update taxi position
        self.state[0], self.state[1] = taxi_row, taxi_col
        
        # Reduce fuel if using fuel constraints
        if self.max_fuel != float('inf'):
            self.remaining_fuel -= 1
            if self.remaining_fuel <= 0:
                reward = self.fuel_penalty
                self.done = True
        
        # Check if all passengers delivered
        all_delivered = all(self.state[2 + i] == 1 for i in range(self.num_passengers))
        if all_delivered:
            self.done = True
        
        self.reward = reward
        
        return self._get_obs(), reward, self.done, False, {}
    
    def _get_obs(self):
        """Convert internal state to observation."""
        return np.array(self.state)
        
    def get_state_representation(self):
        """Get state representation suitable for transfer learning."""
        # Normalize positions
        taxi_row_norm = self.state[0] / (self.grid_size - 1)
        taxi_col_norm = self.state[1] / (self.grid_size - 1)
        
        # Passenger status (one-hot encoded)
        status = self.state[2]
        status_vector = [0, 0, 0]  # [not picked, in taxi, delivered]
        status_vector[status + 1] = 1
        
        # Destination location (normalized)
        dest_idx = self.state[3]
        dest_row, dest_col = dest_idx // self.grid_size, dest_idx % self.grid_size
        dest_row_norm = dest_row / (self.grid_size - 1)
        dest_col_norm = dest_col / (self.grid_size - 1)
        
        # Passenger location (normalized)
        passenger_row_norm = self.state[-2] / (self.grid_size - 1)
        passenger_col_norm = self.state[-1] / (self.grid_size - 1)
        
        # Combine all features
        return np.array([
            taxi_row_norm, taxi_col_norm,          # Taxi position
            *status_vector,                        # Passenger status
            dest_row_norm, dest_col_norm,          # Destination location
            passenger_row_norm, passenger_col_norm  # Passenger location
        ])

    def _passenger_at_location(self, row, col):
        """Check if any passenger is at the given location."""
        for i, loc in enumerate([self.passenger_locations]):
            if loc[0] == row and loc[1] == col and self.state[2 + i] == -1:
                return i
        return None
    
    def increase_complexity(self, increment=0.1):
        """Increase the environment complexity."""
        current = self.complexity_level
        target = min(1.0, current + increment)
        
        # Update complexity parameters based on new level
        if current < 0.3 and target >= 0.3:
            # Increase grid size
            self.grid_size += 1
            self.observation_space = spaces.MultiDiscrete([
                self.grid_size,                      # taxi row
                self.grid_size,                      # taxi col
                *([3] * self.num_passengers),        # passenger status
                *([self.grid_size * self.grid_size] * self.num_passengers)  # destination locations
            ])
        
        if current < 0.5 and target >= 0.5:
            # Add stochasticity
            self.stochasticity = 0.1
        
        if current < 0.7 and target >= 0.7:
            # Add fuel constraints
            self.max_fuel = self.grid_size * self.grid_size * 3
        
        if current < 0.9 and target >= 0.9:
            # Add a passenger
            self.num_passengers += 1
            # Update observation space
            self.observation_space = spaces.MultiDiscrete([
                self.grid_size,                      # taxi row
                self.grid_size,                      # taxi col
                *([3] * self.num_passengers),        # passenger status
                *([self.grid_size * self.grid_size] * self.num_passengers)  # destination locations
            ])
        
        # Update complexity level in config
        self.config["complexity_level"] = target