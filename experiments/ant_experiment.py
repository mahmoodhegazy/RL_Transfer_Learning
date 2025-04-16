"""
Continuous Transfer Learning Experiment with Ant Environment

This script demonstrates how to:
1. Train agents on simplified Ant environments
2. Transfer knowledge to the full Ant environment
3. Compare performance across different transfer mechanisms
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from run_experiment import ExperimentRunner

# Get the absolute path of the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the project root to the Python path
sys.path.insert(0, project_root)

def run_ant_transfer_experiment():
    """Run transfer learning experiment with Ant environments."""
    print("\n===== Running Ant Transfer Experiment =====")
    
    # Define experiment configuration
    experiment_config = {
        'name': 'ant_transfer_experiment',
        'num_episodes': 200,  # Number of episodes to train in target environment
        'eval_frequency': 10,
        'eval_episodes': 3,

        # Target environment configuration (full Ant)
        'target_env_config': {
            'type': 'ant',
            'observation_noise': 0.0,
            'action_noise': 0.0,
            'control_frequency': 5
        },

        # Agent configuration for target environment
        'agent_config': {
            'type': 'actor_critic',  # Can also try 'ppo' or 'sac'
            'learning_rate': 0.001,
            'discount_factor': 0.99,
            'hidden_dims': [128, 128]
        },

        # Transfer experiment configurations
        'transfer_configs': [
            # 1. Parameter Transfer from Reduced DOF Ant
            {
                'name': 'parameter_transfer_reduced_dof',
                'source_env_config': {
                    'type': 'reduced_dof_ant',
                    'active_joints': [0, 1, 2, 3]  # Only front legs active
                },
                'source_agent_config': {
                    'type': 'actor_critic',
                    'learning_rate': 0.001,
                    'discount_factor': 0.99,
                    'hidden_dims': [128, 128]
                },
                'source_episodes': 100,  # Train source for 100 episodes
                'mechanism_config': {
                    'type': 'parameter_transfer',
                    'transfer_weights': True,
                    'transfer_bias': True
                }
            },
            
            # 2. Feature Transfer from Planar Ant
            {
                'name': 'feature_transfer_planar',
                'source_env_config': {
                    'type': 'planar_ant',
                    'observation_noise': 0.0
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
            
            # 3. Policy Distillation from Simplified Physics Ant
            {
                'name': 'policy_distillation_simple_physics',
                'source_env_config': {
                    'type': 'simplified_physics_ant',
                    'friction': 0.8,
                    'gravity': -9.0,
                    'control_frequency': 10
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
                    'iterations': 300,  # Distillation iterations
                    'batch_size': 64,
                    'learning_rate': 0.0001,
                    'loss_type': 'kl',
                    'collect_states': 2000  # States to collect for distillation
                }
            }
        ]
    }

    # Create results directory if it doesn't exist
    os.makedirs("results/ant", exist_ok=True)
    
    # Run the experiment
    runner = ExperimentRunner(experiment_config, output_dir="results/ant")
    results = runner.run()
    
    print("\nAnt experiment completed. Results saved in 'results/ant' directory.")
    
    # Visualize results
    visualize_results(results, "results/ant")
    
    return results

def visualize_results(results, output_dir):
    """Create additional visualizations of the results."""
    
    # Plot transfer metrics comparison
    mechanism_names = []
    jumpstart = []
    time_to_threshold = []
    asymptotic = []
    
    for name, results_data in results.items():
        if name == 'baseline':
            continue
            
        metrics = results_data.get('metrics', {})
        mechanism_names.append(name)
        jumpstart.append(metrics.get('jumpstart', 0))
        
        if 'time_to_threshold' in metrics:
            threshold_improvement = metrics['time_to_threshold'].get('improvement', 0)
            time_to_threshold.append(threshold_improvement)
        else:
            time_to_threshold.append(0)
            
        asymptotic.append(metrics.get('asymptotic', 0))
    
    # Create bar chart of metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(mechanism_names))
    width = 0.25
    
    ax.bar(x - width, jumpstart, width, label='Jumpstart Performance')
    ax.bar(x, time_to_threshold, width, label='Time to Threshold Improvement')
    ax.bar(x + width, asymptotic, width, label='Asymptotic Performance')
    
    ax.set_ylabel('Improvement over Baseline')
    ax.set_title('Ant Environment: Transfer Learning Mechanism Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(mechanism_names)
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transfer_metrics_comparison.png"), dpi=300)
    plt.close()
    
    # Print summary of metrics
    print("\nTransfer Learning Metrics Summary:")
    for i, name in enumerate(mechanism_names):
        print(f"  {name}:")
        print(f"    Jumpstart Improvement: {jumpstart[i]:.2f}")
        print(f"    Time to Threshold Improvement: {time_to_threshold[i]:.2f} episodes")
        print(f"    Asymptotic Performance Improvement: {asymptotic[i]:.2f}")

if __name__ == "__main__":
    run_ant_transfer_experiment()