from environments.discrete.taxi import TaxiEnv

class SimplifiedTaxiEnv(TaxiEnv):
    """
    Simplified Taxi environment with reduced complexity.
    
    Simplifications include:
    - Smaller grid size
    - Single passenger
    - Fixed or limited pickup/dropoff locations
    - No fuel constraints
    - Deterministic transitions
    """
    
    def __init__(self, config):
        # Set default simplifications
        simplified_config = config.copy()
        simplified_config.setdefault("grid_size", 3)  # Smaller grid
        simplified_config.setdefault("num_passengers", 1)  # Single passenger
        simplified_config.setdefault("max_fuel", float('inf'))  # No fuel constraint
        simplified_config.setdefault("stochasticity", 0.0)  # Deterministic
        
        # Fixed pickup/dropoff locations if specified
        self.use_fixed_locations = simplified_config.get("use_fixed_locations", False)
        self.fixed_passenger_locations = simplified_config.get("fixed_passenger_locations", (0, 0))
        self.fixed_destination_locations = simplified_config.get("fixed_destination_locations", (2, 2))
        
        super().__init__(simplified_config)
    
    def _generate_locations(self):
        """Generate passenger pickup and destination locations."""
        if self.use_fixed_locations:
            # Use fixed locations
            self.passenger_locations = self.fixed_passenger_locations
            self.destination_locations = self.fixed_destination_locations
        else:
            # Use random but simplified locations (e.g., corners only)
            super()._generate_locations()