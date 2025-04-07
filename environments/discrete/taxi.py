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
        self.passenger_locations = []
        self.destination_locations = []
        self._generate_locations()
        
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
            *([self.grid_size * self.grid_size] * self.num_passengers)  # destination locations
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
        
    def _generate_locations(self):
        """Generate random passenger pickup and destination locations."""
        all_locations = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        sampled_locations = np.random.choice(
            len(all_locations), 
            size=2 * self.num_passengers, 
            replace=False
        )
        
        for i in range(self.num_passengers):
            self.passenger_locations.append(all_locations[sampled_locations[i]])
            self.destination_locations.append(all_locations[sampled_locations[i + self.num_passengers]])
    
    def reset(self):
        """Reset the environment to initial state."""
        # Reset passenger locations and destinations
        self._generate_locations()
        
        # Reset taxi to random position
        taxi_row = np.random.randint(0, self.grid_size)
        taxi_col = np.random.randint(0, self.grid_size)
        
        # Initialize passenger statuses (not picked up)
        passenger_statuses = [-1] * self.num_passengers
        
        # Flatten destination locations
        flat_destinations = [loc[0] * self.grid_size + loc[1] for loc in self.destination_locations]
        
        # Combine into state
        self.state = [taxi_row, taxi_col] + passenger_statuses + flat_destinations
        
        # Reset fuel if using fuel constraints
        self.remaining_fuel = self.max_fuel
        
        # Reset done flag
        self.done = False
        
        return self._get_obs()
    
    def step(self, action):
        """Take a step in the environment with the given action."""
        taxi_row, taxi_col = self.state[0], self.state[1]
        
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
            passenger_idx = self._passenger_at_location(taxi_row, taxi_col)
            if passenger_idx is not None and self.state[2 + passenger_idx] == -1:
                # Passenger found and not yet picked up
                self.state[2 + passenger_idx] = 0  # In taxi
            else:
                reward = self.illegal_action_penalty  # Illegal pickup
        elif action == 5:  # Dropoff
            for i in range(self.num_passengers):
                passenger_status = self.state[2 + i]
                dest_idx = self.state[2 + self.num_passengers + i]
                dest_row, dest_col = dest_idx // self.grid_size, dest_idx % self.grid_size
                
                if passenger_status == 0 and taxi_row == dest_row and taxi_col == dest_col:
                    # Passenger in taxi and at destination
                    self.state[2 + i] = 1  # Delivered
                    reward = self.dropoff_reward
                    break
            else:
                # No valid dropoff occurred
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
        # Normalize positions to [0,1] range
        taxi_row_norm = self.state[0] / (self.grid_size - 1)
        taxi_col_norm = self.state[1] / (self.grid_size - 1)
        
        # Passenger statuses (one-hot encoded)
        passenger_status_vectors = []
        for i in range(self.num_passengers):
            status = self.state[2 + i]
            status_vector = [0, 0, 0]  # [not picked, in taxi, delivered]
            status_vector[status + 1] = 1
            passenger_status_vectors.extend(status_vector)
        
        # Destination locations (normalized)
        destination_vectors = []
        for i in range(self.num_passengers):
            dest_idx = self.state[2 + self.num_passengers + i]
            dest_row, dest_col = dest_idx // self.grid_size, dest_idx % self.grid_size
            dest_row_norm = dest_row / (self.grid_size - 1)
            dest_col_norm = dest_col / (self.grid_size - 1)
            destination_vectors.extend([dest_row_norm, dest_col_norm])
        
        # Combine all features
        return np.array([taxi_row_norm, taxi_col_norm] + passenger_status_vectors + destination_vectors)
    
    def _passenger_at_location(self, row, col):
        """Check if any passenger is at the given location."""
        for i, loc in enumerate(self.passenger_locations):
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