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
        
    def train(self):
        """Execute curriculum training through all complexity levels."""
        current_agent = None
        performance_history = []
        
        for idx, env_config in enumerate(self.env_configs):
            print(f"Training on complexity level {idx+1}/{len(self.env_configs)}")
            
            # Create environment for this complexity level
            env = self._create_environment(env_config)
            
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
            
        return current_agent, performance_history
    
    def _create_environment(self, env_config):
        """Create an environment with the given configuration."""
        # Implementation depends on environment type
        raise NotImplementedError
    
    def _train_until_threshold(self, agent, env):
        """Train agent until performance threshold is met."""
        episode = 0
        stable_episodes = 0
        performance_history = []
        
        while episode < self.min_episodes or stable_episodes < self.patience:
            # Run an episode
            episode_reward = self._run_episode(agent, env)
            performance_history.append(episode_reward)
            episode += 1
            
            # Check if we've reached performance threshold over a window
            recent_performance = sum(performance_history[-self.patience:]) / min(self.patience, len(performance_history))
            if recent_performance >= self.performance_threshold:
                stable_episodes += 1
            else:
                stable_episodes = 0
                
        return performance_history
    
    def _run_episode(self, agent, env):
        """Run a single episode and return total reward."""
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
        return total_reward