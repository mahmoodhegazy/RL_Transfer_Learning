from environments.discrete.taxi import TaxiEnv

class SinglePassengerTaxiEnv(TaxiEnv):
    """
    Taxi environment with exactly one passenger, but potentially higher complexity
    in other dimensions (grid size, stochasticity, etc.)
    """
    
    def __init__(self, config):
        # Force single passenger
        single_passenger_config = config.copy()
        single_passenger_config["num_passengers"] = 1
        
        super().__init__(single_passenger_config)