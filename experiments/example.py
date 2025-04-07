"""
Example script demonstrating how to use different transfer learning mechanisms.
This script shows how to:
1. Train agents in source environments
2. Transfer knowledge to target environments using different mechanisms
3. Compare performance with and without transfer
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Import environment creators
from environments.discrete.simplified_taxi import SimplifiedTaxiEnv
from environments.discrete.taxi import TaxiEnv
from environments.continuous.reduced_dof_ant import ReducedDOFAntEnv
from environments.continuous.ant import AntEnv

# Import agent types
from agents.discrete.q_learning import QLearningAgent
from agents.continuous.actor_critic import ActorCriticAgent

# Import transfer mechanisms
from transfer.mechanisms.parameter_transfer import ParameterTransfer
from transfer.mechanisms.feature_transfer import FeatureTransfer
from transfer.mechanisms.policy_distillation import PolicyDistillation
from transfer.mechanisms.reward_shaping import RewardShaping
from transfer.mechanisms.value_transfer import ValueTransfer

# Import experiment utilities
from experiments.run_experiment import ExperimentRunner

def run_taxi_transfer_experiment():
    """Run transfer learning experiment with Taxi environments."""
    print("\n===== Running Taxi Transfer Experiment =====")
    
    # Define experiment configuration
    experiment_config = {
        'name': 'taxi_transfer_experiment',
        'num_episodes': 300,
        'eval_frequency': 10,
        'eval_episodes': 5,

        # Target environment configuration (full Taxi)
        'target_env_config': {
            'type': 'taxi',
            'grid_size': 5,
            'num_passengers': 1
        },

        # Agent configuration
        'agent_config': {
            'type': 'q_learning',
            'learning_rate': 0.1,
            'discount_factor': 0.99,
            'exploration_rate': 0.1,
            'exploration_decay': 0.995
        },

        # Transfer experiment configurations
        'transfer_configs': [
            {
                'name': 'parameter_transfer',
                'source_env_config': {
                    'type': 'simplified_taxi',
                    'grid_size': 3,
                    'num_passengers': 1
                },
                'source_agent_config': {
                    'type': 'q_learning',
                    'learning_rate': 0.1,
                    'discount_factor': 0.99,
                    'exploration_rate': 0.1,
                    'exploration_decay': 0.995
                },
                'source_episodes': 200,
                'mechanism_config': {
                    'type': 'parameter_transfer',
                    'transfer_weights': True,
                    'transfer_bias': True
                }
            },
            {
                'name': 'value_transfer',
                'source_env_config': {
                    'type': 'simplified_taxi',
                    'grid_size': 3,
                    'num_passengers': 1
                },
                'source_agent_config': {
                    'type': 'q_learning',
                    'learning_rate': 0.1,
                    'discount_factor': 0.99,
                    'exploration_rate': 0.1,
                    'exploration_decay': 0.995
                },
                'source_episodes': 200,
                'mechanism_config': {
                    'type': 'value_transfer',
                    'transfer_type': 'q_values',
                    'adaptation_method': 'normalized'
                }
            },
            {
                'name': 'reward_shaping',
                'source_env_config': {
                    'type': 'simplified_taxi',
                    'grid_size': 3,
                    'num_passengers': 1
                },
                'source_agent_config': {
                    'type': 'q_learning',
                    'learning_rate': 0.1,
                    'discount_factor': 0.99,
                    'exploration_rate': 0.1,
                    'exploration_decay': 0.995
                },
                'source_episodes': 200,
                'mechanism_config': {
                    'type': 'reward_shaping',
                    'shaping_method': 'potential_based',
                    'scaling_factor': 0.5,
                    'gamma': 0.99,
                    'decay_factor': 0.99
                }
            }
        ]
    }

    # Run the experiment
    runner = ExperimentRunner(experiment_config, output_dir="results/taxi")
    results = runner.run()
    
    print("\nTaxi experiment completed. Results saved in 'results/taxi' directory.")
    return results

def run_ant_transfer_experiment():
    """Run transfer learning experiment with Ant environments."""
    print("\n===== Running Ant Transfer Experiment =====")
    
    # Define experiment configuration
    experiment_config = {
        'name': 'ant_transfer_experiment',
        'num_episodes': 200,
        'eval_frequency': 10,
        'eval_episodes': 3,

        # Target environment configuration (full Ant)
        'target_env_config': {
            'type': 'ant',
            'observation_noise': 0.0,
            'action_noise': 0.0
        },

        # Agent configuration
        'agent_config': {
            'type': 'actor_critic',
            'learning_rate': 0.001,
            'discount_factor': 0.99,
            'hidden_dims': [128, 128]
        },

        # Transfer experiment configurations
        'transfer_configs': [
            {
                'name': 'parameter_transfer',
                'source_env_config': {
                    'type': 'reduced_dof_ant',
                    'active_joints': [0, 1, 2, 3]
                },
                'source_agent_config': {
                    'type': 'actor_critic',
                    'learning_rate': 0.001,
                    'discount_factor': 0.99,
                    'hidden_dims': [128, 128]
                },
                'source_episodes': 100,
                'mechanism_config': {
                    'type': 'parameter_transfer',
                    'transfer_weights': True,
                    'transfer_bias': True
                }
            },
            {
                'name': 'feature_transfer',
                'source_env_config': {
                    'type': 'reduced_dof_ant',
                    'active_joints': [0, 1, 2, 3]
                },
                'source_agent_config': {
                    'type': 'actor_critic',
                    'learning_rate': 0.001,
                    'discount_factor': 0.99,
                    'hidden_dims': [128, 128]
                },
                'source_episodes': 100,
                'mechanism_config': {
                    'type': 'feature_transfer',
                    'layers_to_transfer': ["all"],
                    'freeze_transferred': False,
                    'adaptation_method': 'truncate'
                }
            },
            {
                'name': 'policy_distillation',
                'source_env_config': {
                    'type': 'reduced_dof_ant',
                    'active_joints': [0, 1, 2, 3]
                },
                'source_agent_config': {
                    'type': 'actor_critic',
                    'learning_rate': 0.001,
                    'discount_factor': 0.99,
                    'hidden_dims': [128, 128]
                },
                'source_episodes': 100,
                'mechanism_config': {
                    'type': 'policy_distillation',
                    'temperature': 2.0,
                    'iterations': 500,
                    'batch_size': 64,
                    'learning_rate': 0.0001,
                    'loss_type': 'kl'
                }
            }
        ]
    }

    # Run the experiment
    runner = ExperimentRunner(experiment_config, output_dir="results/ant")
    results = runner.run()
    
    print("\nAnt experiment completed. Results saved in 'results/ant' directory.")
    return results

def compare_transfer_mechanisms(taxi_results=None, ant_results=None):
    """Compare different transfer mechanisms and visualize results."""
    print("\n===== Comparing Transfer Mechanisms =====")
    
    # Create results directory if it doesn't exist
    os.makedirs("results/comparison", exist_ok=True)
    
    # Plot transfer metrics for discrete (Taxi) environment
    if taxi_results:
        print("\nTaxi Transfer Metrics:")
        # Extract metrics for each mechanism
        mechanism_names = []
        jumpstart = []
        time_to_threshold = []
        asymptotic = []
        
        for name, results in taxi_results.items():
            if name == 'baseline':
                continue
                
            metrics = results.get('metrics', {})
            mechanism_names.append(name)
            jumpstart.append(metrics.get('jumpstart', 0))
            
            if 'time_to_threshold' in metrics:
                threshold_improvement = metrics['time_to_threshold'].get('improvement', 0)
                time_to_threshold.append(threshold_improvement)
            else:
                time_to_threshold.append(0)
                
            asymptotic.append(metrics.get('asymptotic', 0))
            
            # Print metrics
            print(f"  {name}:")
            print(f"    Jumpstart Improvement: {metrics.get('jumpstart', 0):.2f}")
            if 'time_to_threshold' in metrics:
                print(f"    Time to Threshold Improvement: {metrics['time_to_threshold'].get('improvement', 0):.2f} episodes")
            print(f"    Asymptotic Performance Improvement: {metrics.get('asymptotic', 0):.2f}")
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(mechanism_names))
        width = 0.25
        
        ax.bar(x - width, jumpstart, width, label='Jumpstart')
        ax.bar(x, time_to_threshold, width, label='Time to Threshold Improvement')
        ax.bar(x + width, asymptotic, width, label='Asymptotic')
        
        ax.set_ylabel('Improvement')
        ax.set_title('Taxi Environment: Transfer Mechanism Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(mechanism_names)
        ax.legend()
        
        plt.savefig("results/comparison/taxi_metrics_comparison.png", dpi=300)
    
    # Plot transfer metrics for continuous (Ant) environment
    if ant_results:
        print("\nAnt Transfer Metrics:")
        # Extract metrics for each mechanism
        mechanism_names = []
        jumpstart = []
        time_to_threshold = []
        asymptotic = []
        
        for name, results in ant_results.items():
            if name == 'baseline':
                continue
                
            metrics = results.get('metrics', {})
            mechanism_names.append(name)
            jumpstart.append(metrics.get('jumpstart', 0))
            
            if 'time_to_threshold' in metrics:
                threshold_improvement = metrics['time_to_threshold'].get('improvement', 0)
                time_to_threshold.append(threshold_improvement)
            else:
                time_to_threshold.append(0)
                
            asymptotic.append(metrics.get('asymptotic', 0))
            
            # Print metrics
            print(f"  {name}:")
            print(f"    Jumpstart Improvement: {metrics.get('jumpstart', 0):.2f}")
            if 'time_to_threshold' in metrics:
                print(f"    Time to Threshold Improvement: {metrics['time_to_threshold'].get('improvement', 0):.2f} episodes")
            print(f"    Asymptotic Performance Improvement: {metrics.get('asymptotic', 0):.2f}")
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(mechanism_names))
        width = 0.25
        
        ax.bar(x - width, jumpstart, width, label='Jumpstart')
        ax.bar(x, time_to_threshold, width, label='Time to Threshold Improvement')
        ax.bar(x + width, asymptotic, width, label='Asymptotic')
        
        ax.set_ylabel('Improvement')
        ax.set_title('Ant Environment: Transfer Mechanism Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(mechanism_names)
        ax.legend()
        
        plt.savefig("results/comparison/ant_metrics_comparison.png", dpi=300)
    
    print("\nComparison complete. Visualization saved in 'results/comparison' directory.")

def main():
    """Run all transfer learning experiments."""
    print("Starting Transfer Learning Experiments...")
    
    # Run Taxi experiments
    taxi_results = run_taxi_transfer_experiment()
    
    # Run Ant experiments
    # Note: This might require more computational resources
    ant_results = run_ant_transfer_experiment()
    
    # Compare results
    compare_transfer_mechanisms(taxi_results, ant_results)
    
    print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    main()