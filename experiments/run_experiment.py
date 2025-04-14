import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

class ExperimentRunner:
    """Run transfer learning experiments and collect results."""
    
    def __init__(self, experiment_config, output_dir="results"):
        self.config = experiment_config
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def run(self):
        """Run the complete experiment."""
        print(f"Starting experiment: {self.config['name']}")
        
        # Run baseline (learning from scratch)
        print("Running baseline experiment...")
        self.results['baseline'] = self._run_baseline()
        print(f"Ran baseline for: {self.config['name']}")
        # Run transfer experiments
        for transfer_config in self.config['transfer_configs']:
            transfer_name = transfer_config['name']
            print(f"Running transfer experiment: {transfer_name}")
            self.results[transfer_name] = self._run_transfer_experiment(transfer_config)
        
        # Evaluate transfer performance
        self._evaluate_transfer()
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _run_baseline(self):
        """Run baseline learning from scratch."""
        # Create environment and agent
        env = self._create_environment(self.config['target_env_config'])
        agent = self._create_agent(env, self.config['agent_config'])
        
        # Train agent from scratch
        return self._train_agent(agent, env, self.config['num_episodes'])
    
    def _run_transfer_experiment(self, transfer_config):
        """Run a transfer learning experiment."""
        # Create source environment and agent
        source_env = self._create_environment(transfer_config['source_env_config'])
        source_agent = self._create_agent(source_env, transfer_config['source_agent_config'])
        
        # Train source agent
        source_results = self._train_agent(
            source_agent, 
            source_env,
            transfer_config.get('source_episodes', self.config['num_episodes'])
        )
        
        # Create target environment
        target_env = self._create_environment(self.config['target_env_config'])
        
        # Create transfer mechanism
        transfer_mechanism = self._create_transfer_mechanism(transfer_config['mechanism_config'])
        
        # Create new agent for target environment
        target_agent = self._create_agent(target_env, self.config['agent_config'])
        
        # Apply transfer
        transferred_agent = transfer_mechanism.transfer(source_agent, target_agent)
        
        # Train transferred agent
        target_results = self._train_agent(
            transferred_agent,
            target_env,
            self.config['num_episodes']
        )
        
        return {
            'source_results': source_results,
            'target_results': target_results,
            'transfer_config': transfer_config
        }
    
    def _train_agent(self, agent, env, num_episodes):
        """Train an agent for specified number of episodes."""
        results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'evaluation_rewards': []
        }
        
        for episode in range(num_episodes):
            # # Training episode
            # state = env.reset()
            # done = False
            # episode_reward = 0
            # steps = 0
            
            # while not done:
            #     action = agent.select_action(state)
            #     result = env.step(action)
            #     next_state, reward, terminated, truncated, _ = result
            #     agent.update(state, action, reward, next_state, done)
            #     state = next_state
            #     episode_reward += reward
            #     steps += 1

            #     done = terminated or truncated or steps >= 700
            steps = 0
            s = env.reset()
            a = agent.select_action(s)
            done = False
            episode_reward = 0
            max_steps = 600

            while not done and steps < max_steps:
                s_prime, reward, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                a_prime = agent.select_action(s_prime)
                agent.update(s, a, reward, s_prime, done)
                s, a = s_prime, a_prime
                episode_reward += reward
                steps += 1

            results['episode_rewards'].append(episode_reward)
            results['episode_lengths'].append(steps)
            
            # Periodic evaluation (without exploration)
            if episode % self.config.get('eval_frequency', 10) == 0:
                eval_reward = self._evaluate_agent(agent, env, self.config.get('eval_episodes', 5))
                results['evaluation_rewards'].append((episode, eval_reward))
                
            # Log progress
            if episode % 10 == 0:
                # Calculate average reward over last 10 episodes
                last_10_avg = np.mean(results['episode_rewards'][-10:]) if len(results['episode_rewards']) >= 10 else np.mean(results['episode_rewards'])
                print(f"Episode {episode}/{num_episodes}, Average Reward (last 10): {last_10_avg:.2f}")
        # Plot results for this training session
        plt.figure(figsize=(10, 6))
        episodes = range(1, len(results['episode_rewards']) + 1)
        plt.plot(episodes, results['episode_rewards'], label='Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()
        return results
    
    def _evaluate_agent(self, agent, env, num_episodes):
        """Evaluate agent performance without exploration."""
        # Save original exploration parameters
        original_exploration = agent.config.get('exploration_rate', 0.1)
        agent.config['exploration_rate'] = 0
        
        total_rewards = []
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                action = agent.select_action(state)
                result = env.step(action)
                next_state, reward, terminated, truncated, _ = result
                # agent.update(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                steps += 1

                done = terminated or truncated or steps >= 700
                
            total_rewards.append(episode_reward)
        
        # Restore exploration
        agent.config['exploration_rate'] = original_exploration
        
        return np.mean(total_rewards)
    
    def _evaluate_transfer(self):
        """Evaluate transfer learning performance."""
        from evaluation.metrics import TransferMetrics
        
        baseline_rewards = self.results['baseline']['episode_rewards']
        
        for transfer_name, transfer_results in self.results.items():
            if transfer_name == 'baseline':
                continue
                
            transfer_rewards = transfer_results['target_results']['episode_rewards']
            
            # Calculate transfer metrics
            metrics = {
                'jumpstart': TransferMetrics.jumpstart_performance(baseline_rewards, transfer_rewards),
                'asymptotic': TransferMetrics.asymptotic_performance(baseline_rewards, transfer_rewards),
                'transfer_ratio': TransferMetrics.transfer_ratio(baseline_rewards, transfer_rewards),
                'significance': TransferMetrics.statistical_significance(baseline_rewards, transfer_rewards),
                'negative_transfer': TransferMetrics.detect_negative_transfer(baseline_rewards, transfer_rewards)
            }
            
            # Calculate time to threshold if possible
            target_perf = np.max(baseline_rewards) * 0.9  # 90% of max baseline performance
            baseline_time, transfer_time = TransferMetrics.time_to_threshold(
                baseline_rewards, transfer_rewards, target_perf
            )
            metrics['time_to_threshold'] = {
                'baseline': baseline_time,
                'transfer': transfer_time,
                'improvement': baseline_time - transfer_time
            }
            
            # Store metrics
            self.results[transfer_name]['metrics'] = metrics
    
    def _save_results(self):
        """Save experiment results."""
        experiment_dir = os.path.join(
            self.output_dir, 
            f"{self.config['name']}_{self.timestamp}"
        )
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # Save metrics summary
        metrics_summary = {}
        for transfer_name, results in self.results.items():
            if transfer_name == 'baseline':
                continue
            metrics_summary[transfer_name] = {
                k: v for k, v in results.get('metrics', {}).items() 
                if k not in ['significance', 'negative_transfer']
            }
            # Add simplified versions of complex metrics
            if 'significance' in results.get('metrics', {}):
                metrics_summary[transfer_name]['p_value'] = results['metrics']['significance']['p_value']
            if 'negative_transfer' in results.get('metrics', {}):
                metrics_summary[transfer_name]['is_negative'] = results['metrics']['negative_transfer']['negative_transfer_detected']
        
        with open(os.path.join(experiment_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=4)
        
        # Generate learning curve plot
        self._plot_learning_curves(experiment_dir)
        
        # Save full results
        with open(os.path.join(experiment_dir, 'full_results.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_serializable(self.results)
            json.dump(serializable_results, f, indent=4)
    
    def _make_serializable(self, obj):
        """Convert numpy values to serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def _plot_learning_curves(self, output_dir):
        """Plot learning curves from all experiments."""
        plt.figure(figsize=(10, 6))
        
        # Plot baseline
        baseline_rewards = self.results['baseline']['episode_rewards']
        episodes = range(1, len(baseline_rewards) + 1)
        plt.plot(episodes, baseline_rewards, label='Baseline (No Transfer)')
        
        # Plot transfer experiments
        for transfer_name, results in self.results.items():
            if transfer_name == 'baseline':
                continue
                
            transfer_rewards = results['target_results']['episode_rewards']
            plt.plot(episodes[:len(transfer_rewards)], transfer_rewards, label=transfer_name)
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Learning Curves: Baseline vs Transfer Methods')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300)
        plt.close()
    
    def _create_environment(self, env_config):
        """Create environment based on configuration."""
        # This would be implemented based on your environment classes
        env_type = env_config.get('type', 'taxi')
        
        if env_type == 'taxi':
            from environments.discrete.taxi import TaxiEnv
            return TaxiEnv(env_config)
        elif env_type == 'simplified_taxi':
            from environments.discrete.simplified_taxi import SimplifiedTaxiEnv
            return SimplifiedTaxiEnv(env_config)
        elif env_type == 'ant':
            from environments.continuous.ant import AntEnv
            return AntEnv(env_config)
        elif env_type == 'reduced_dof_ant':
            from environments.continuous.reduced_dof_ant import ReducedDOFAntEnv
            return ReducedDOFAntEnv(env_config)
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
    
    def _create_agent(self, env, agent_config):
        """Create agent based on configuration."""
        # This would be implemented based on your agent classes
        agent_type = agent_config.get('type', 'q_learning')
        
        if agent_type == 'q_learning':
            from agents.discrete.q_learning import QLearningAgent
            return QLearningAgent(env, agent_config)
        elif agent_type == 'expected_sarsa':
            from agents.discrete.expected_sarsa import ExpectedSarsaAgent
            return ExpectedSarsaAgent(env, agent_config)
        elif agent_type == 'actor_critic':
            from agents.continuous.actor_critic import ActorCriticAgent
            return ActorCriticAgent(env, agent_config)
        elif agent_type == 'ppo':
            from agents.continuous.ppo import PPOAgent
            return PPOAgent(env, agent_config)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _create_transfer_mechanism(self, mechanism_config):
        """Create transfer mechanism based on configuration."""
        mechanism_type = mechanism_config.get('type', 'parameter_transfer')
        
        if mechanism_type == 'parameter_transfer':
            from transfer.mechanisms.parameter_transfer import ParameterTransfer
            return ParameterTransfer(mechanism_config)
        elif mechanism_type == 'feature_transfer':
            from transfer.mechanisms.feature_transfer import FeatureTransfer
            return FeatureTransfer(mechanism_config)
        # elif mechanism_type == 'policy_distillation':
        #     from transfer.mechanisms.policy_distillation import FeatureTransfer
        #     return FeatureTransfer(mechanism_config)
        elif mechanism_type == 'value_transfer':
            from transfer.mechanisms.value_transfer import ValueTransfer
            return ValueTransfer(mechanism_config)
        elif mechanism_type == 'reward_shaping':
            from transfer.mechanisms.reward_shaping import RewardShaping
            return RewardShaping(mechanism_config)
        else:
            raise ValueError(f"Unknown transfer mechanism: {mechanism_type}")