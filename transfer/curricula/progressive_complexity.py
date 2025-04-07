# transfer/curricula/progressive_complexity.py
class ProgressiveComplexityCurriculum:
    """Curriculum that progressively increases environment complexity."""
    
    def __init__(self, env_configs, agent_factory, transfer_mechanism, config):
        self.env_configs = sorted(env_configs, key=lambda x: x.get("complexity_level", 0))
        self.agent_factory = agent_factory
        self.transfer_mechanism = transfer_mechanism
        self.config = config
        
        # Performance thresholds for progression
        self.performance_threshold = config.get("performance_threshold", 0.9)
        self.min_episodes = config.get("min_episodes", 10)
        self.patience = config.get("patience", 5)
        
        # Environment creation function
        self.create_env_func = config.get("create_env_func", self._default_create_environment)
        
        # Logging and visualization
        self.log_progress = config.get("log_progress", True)
        self.visualize = config.get("visualize", False)
        self.visualization_frequency = config.get("visualization_frequency", 10)
        
    def train(self):
        """Execute curriculum training through all complexity levels."""
        current_agent = None
        performance_history = []
        complexity_levels = []
        
        for idx, env_config in enumerate(self.env_configs):
            if self.log_progress:
                print(f"Training on complexity level {idx+1}/{len(self.env_configs)}")
                print(f"Complexity: {env_config.get('complexity_level', 0)}")
                complexity_levels.append(env_config.get('complexity_level', 0))
            
            # Create environment for this complexity level
            env = self.create_env_func(env_config)
            
            # Create new agent or transfer from previous
            if current_agent is None:
                current_agent = self.agent_factory(env, self.config)
            else:
                # Create a new agent for this environment
                new_agent = self.agent_factory(env, self.config)
                
                # Transfer knowledge from previous agent
                current_agent = self.transfer_mechanism.transfer(current_agent, new_agent)
            
            # Train until performance threshold is met
            performance = self._train_until_threshold(current_agent, env)
            performance_history.append(performance)
            
        return current_agent, {"performance_history": performance_history, "complexity_levels": complexity_levels}
    
    def _default_create_environment(self, env_config):
        """Create an environment based on its type."""
        env_type = env_config.get("type", "")
        
        # Discrete environments
        if env_type == "taxi":
            from environments.discrete.taxi import TaxiEnv
            return TaxiEnv(env_config)
        elif env_type == "simplified_taxi":
            from environments.discrete.simplified_taxi import SimplifiedTaxiEnv
            return SimplifiedTaxiEnv(env_config)
        elif env_type == "fixed_route_taxi":
            from environments.discrete.fixed_route_taxi import FixedRouteTaxiEnv
            return FixedRouteTaxiEnv(env_config)
        elif env_type == "single_passenger_taxi":
            from environments.discrete.single_passenger_taxi import SinglePassengerTaxiEnv
            return SinglePassengerTaxiEnv(env_config)
        
        # Continuous environments
        elif env_type == "ant":
            from environments.continuous.ant import AntEnv
            return AntEnv(env_config)
        elif env_type == "reduced_dof_ant":
            from environments.continuous.reduced_dof_ant import ReducedDOFAntEnv
            return ReducedDOFAntEnv(env_config)
        elif env_type == "half_ant":
            from environments.continuous.half_ant import HalfAntEnv
            return HalfAntEnv(env_config)
        elif env_type == "planar_ant":
            from environments.continuous.planar_ant import PlanarAntEnv
            return PlanarAntEnv(env_config)
        elif env_type == "simplified_physics_ant":
            from environments.continuous.simplified_physics_ant import SimplifiedPhysicsAntEnv
            return SimplifiedPhysicsAntEnv(env_config)
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
    
    def _train_until_threshold(self, agent, env):
        """Train agent until performance threshold is met."""
        episode = 0
        stable_episodes = 0
        performance_history = []
        
        while episode < self.min_episodes or stable_episodes < self.patience:
            # Run an episode
            episode_reward = self._run_episode(agent, env, episode)
            performance_history.append(episode_reward)
            episode += 1
            
            # Check if we've reached performance threshold over a window
            recent_performance = sum(performance_history[-self.patience:]) / min(self.patience, len(performance_history))
            if recent_performance >= self.performance_threshold:
                stable_episodes += 1
            else:
                stable_episodes = 0
            
            # Log progress
            if self.log_progress and episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Avg: {recent_performance:.2f}")
                
        return performance_history
    
    def _run_episode(self, agent, env, episode_num):
        """Run a single episode and return total reward."""
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            action = agent.select_action(state)
            
            # Handle different API versions (OpenAI Gym vs Gymnasium)
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'spec') and env.unwrapped.spec.version.startswith('0.'):
                # OpenAI Gym style (old API)
                next_state, reward, done, info = env.step(action)
                truncated = False
            else:
                # Gymnasium style (new API)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1
            
            # Visualize if requested
            if self.visualize and episode_num % self.visualization_frequency == 0:
                env.render()
            
        return total_reward