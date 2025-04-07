import numpy as np
from environments.discrete.taxi import TaxiEnv

class FixedRouteTaxiEnv(TaxiEnv):
    """
    Taxi environment with predetermined routes/paths.
    
    The taxi follows a fixed path network (like a road system) rather than
    being able to move freely in any direction.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Generate road network
        self.roads = self._generate_road_network()
    
    def _generate_road_network(self):
        """Generate a road network as a graph of connected cells."""
        # Initialize with no connections
        roads = set()
        
        # Create a grid of roads
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                # Horizontal roads
                roads.add(((i, j), (i, j + 1)))
                roads.add(((i, j + 1), (i, j)))
                
                # Vertical roads
                roads.add(((j, i), (j + 1, i)))
                roads.add(((j + 1, i), (j, i)))
        
        return roads
    
    def step(self, action):
        """Override step to enforce road network constraints."""
        taxi_row, taxi_col = self.state[0], self.state[1]
        next_row, next_col = taxi_row, taxi_col
        
        # Determine next position based on action
        if action == 0:  # North
            next_row = taxi_row - 1
        elif action == 1:  # South
            next_row = taxi_row + 1
        elif action == 2:  # East
            next_col = taxi_col + 1
        elif action == 3:  # West
            next_col = taxi_col - 1
        
        # Check if movement is valid on road network
        valid_move = (
            action >= 4 or  # Pickup/dropoff actions
            ((taxi_row, taxi_col), (next_row, next_col)) in self.roads
        )
        
        # If not a valid move, stay in place
        if not valid_move:
            next_row, next_col = taxi_row, taxi_col
            
        # Update state temporarily for parent step method
        self.state[0], self.state[1] = next_row, next_col
        
        # Call parent step method with modified state
        return super().step(action)