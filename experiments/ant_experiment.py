"""
Continuous Transfer Learning Experiment with Ant Environment

This script demonstrates:
1. Training agents on simplified Ant environments with better source training
2. Transferring knowledge to the full Ant environment with improved mechanisms
3. Comparing performance across different transfer methods
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
    print("\n===== Running Improved Ant Transfer Experiment =====")
    
    # Define experiment configuration with improved parameters
    experiment_config = {
        'name': 'improved_ant_transfer_experiment',
        'num_episodes': 300,  # Increased training episodes in target environment
        'eval_frequency': 20,
        'eval_episodes': 5,

        # Target environment configuration (full Ant)
        'target_env_config': {
            'type': 'ant',
            'observation_noise': 0.0,
            'action_noise': 0.0,
            'control_frequency': 5
        },

        # Agent configuration with improved hyperparameters
        'agent_config': {
            'type': 'actor_critic',
            'learning_rate': 0.0005,  # Reduced learning rate for more stable learning
            'discount_factor': 0.99,
            'hidden_dims': [256, 256],  # Larger network for better capacity
            'exploration_rate': 0.1,
            'entropy_coef': 0.01,  # Add entropy for exploration
            'clip_grad': 0.5  # Add gradient clipping for stability
        },

        # Transfer experiment configurations
        'transfer_configs': [
            # 1. Parameter Transfer from Reduced DOF Ant with improved settings
            {
                'name': 'parameter_transfer_reduced_dof',
                'source_env_config': {
                    'type': 'reduced_dof_ant',
                    'active_joints': [0, 1, 2, 3]  # Only front legs active
                },
                'source_agent_config': {
                    'type': 'actor_critic',
                    'learning_rate': 0.0005,
                    'discount_factor': 0.99,
                    'hidden_dims': [256, 256],
                    'exploration_rate': 0.1,
                    'entropy_coef': 0.01,
                    'clip_grad': 0.5
                },
                'source_episodes': 200,  # Increased source training
                'mechanism_config': {
                    'type': 'parameter_transfer',
                    'transfer_weights': True,
                    'transfer_bias': True,
                    'use_state_mapping': True,  # Enable state mapping
                    'weight_scaling': 0.8  # Scale weights to prevent overfitting
                }
            },
            
            # 2. Feature Transfer from Planar Ant with improved settings
            {
                'name': 'feature_transfer_planar',
                'source_env_config': {
                    'type': 'planar_ant',
                    'observation_noise': 0.0,
                    'control_frequency': 5  # Match target control frequency
                },
                'source_agent_config': {
                    'type': 'actor_critic',
                    'learning_rate': 0.0005,
                    'discount_factor': 0.99,
                    'hidden_dims': [256, 256],
                    'exploration_rate': 0.1,
                    'entropy_coef': 0.01,
                    'clip_grad': 0.5
                },
                'source_episodes': 200,  # Increased source training
                'mechanism_config': {
                    'type': 'feature_transfer',
                    'layers_to_transfer': ["0", "2"],  # Transfer only early layers
                    'freeze_transferred': False,
                    'adaptation_method': 'pad',  # Use padding instead of truncation
                    'use_state_mapping': True  # Enable state mapping
                }
            },
            
            # 3. Policy Distillation from Simplified Physics Ant with improved settings
            {
                'name': 'policy_distillation_simple_physics',
                'source_env_config': {
                    'type': 'simplified_physics_ant',
                    'friction': 0.8,
                    'gravity': -9.0,
                    'control_frequency': 5  # Match target control frequency
                },
                'source_agent_config': {
                    'type': 'actor_critic',
                    'learning_rate': 0.0005,
                    'discount_factor': 0.99,
                    'hidden_dims': [256, 256],
                    'exploration_rate': 0.1,
                    'entropy_coef': 0.01,
                    'clip_grad': 0.5
                },
                'source_episodes': 200,  # Increased source training
                'mechanism_config': {
                    'type': 'policy_distillation',
                    'temperature': 1.0,  # Reduced temperature for harder targets
                    'iterations': 500,   # More distillation iterations
                    'batch_size': 128,   # Larger batch size
                    'learning_rate': 0.0001,
                    'loss_type': 'kl',
                    'collect_states': 3000,  # More diverse states
                    'use_state_mapping': True  # Enable state mapping
                }
            },
            
            # 4. New: Progressive Transfer from Simplified to Full Ant
            {
                'name': 'progressive_transfer',
                'source_env_config': {
                    'type': 'simplified_physics_ant',
                    'friction': 0.9,
                    'gravity': -9.5,
                    'control_frequency': 10  # Higher control frequency for easier learning
                },
                'source_agent_config': {
                    'type': 'actor_critic',
                    'learning_rate': 0.0005,
                    'discount_factor': 0.99,
                    'hidden_dims': [256, 256],
                    'exploration_rate': 0.1,
                    'entropy_coef': 0.01,
                    'clip_grad': 0.5
                },
                'source_episodes': 200,  # Increased source training
                'mechanism_config': {
                    'type': 'parameter_transfer',  # Use parameter transfer for each stage
                    'transfer_weights': True,
                    'transfer_bias': True,
                    'use_state_mapping': True,
                    'progressive': True,  # Enable progressive transfer
                    'intermediate_envs': [
                        # Intermediate environment 1: Lower control frequency
                        {
                            'type': 'simplified_physics_ant',
                            'friction': 0.9,
                            'gravity': -9.5,
                            'control_frequency': 7  # Intermediate frequency
                        },
                        # Intermediate environment 2: Standard gravity
                        {
                            'type': 'simplified_physics_ant',
                            'friction': 0.9,
                            'gravity': -9.81,  # Standard gravity
                            'control_frequency': 5  # Target frequency
                        },
                        # Intermediate environment 3: Standard friction
                        {
                            'type': 'simplified_physics_ant',
                            'friction': 1.0,  # Standard friction
                            'gravity': -9.81,
                            'control_frequency': 5
                        }
                    ],
                    'intermediate_episodes': 50  # Train 50 episodes at each intermediate step
                }
            }
        ]
    }

    # Create results directory if it doesn't exist
    os.makedirs("results/improved_ant", exist_ok=True)
    
    # Run the experiment
    runner = ExperimentRunner(experiment_config, output_dir="results/improved_ant")
    results = runner.run()
    
    print("\nImproved Ant experiment completed. Results saved in 'results/improved_ant' directory.")
    
    # Visualize results
    visualize_results(results, "results/improved_ant")
    
    return results

def visualize_results(results, output_dir):
    """Create enhanced visualizations of the results."""
    # Create a more informative plot of learning curves
    if 'baseline' in results:
        plt.figure(figsize=(12, 8))
        
        # Get baseline data
        baseline_rewards = results['baseline']['episode_rewards']
        episodes = range(1, len(baseline_rewards) + 1)
        
        # Apply smoothing for better visualization
        def smooth(data, window=10):
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # Plot baseline with smoothing
        smoothed_baseline = smooth(baseline_rewards)
        plt.plot(episodes[len(episodes)-len(smoothed_baseline):], smoothed_baseline, 
                label='Baseline (No Transfer)', linewidth=2)
        
        # Plot transfer methods
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, (transfer_name, results_data) in enumerate([item for item in results.items() if item[0] != 'baseline']):
            if 'target_results' in results_data:
                transfer_rewards = results_data['target_results']['episode_rewards']
                episodes_transfer = range(1, len(transfer_rewards) + 1)
                
                # Apply smoothing
                smoothed_transfer = smooth(transfer_rewards)
                plt.plot(episodes_transfer[len(episodes_transfer)-len(smoothed_transfer):], 
                        smoothed_transfer, label=transfer_name, linewidth=2, 
                        color=colors[i % len(colors)])
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.title('Learning Curves: Baseline vs Improved Transfer Methods', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=300)
        plt.close()
    
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
    
    # Create enhanced bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(mechanism_names))
    width = 0.25
    
    # Create bars with error patterns for negative values
    jumpstart_bars = ax.bar(x - width, jumpstart, width, label='Jumpstart Performance',
                           color=['#1f77b4' if val >= 0 else '#ff9999' for val in jumpstart])
    
    threshold_bars = ax.bar(x, time_to_threshold, width, label='Time to Threshold Improvement',
                           color=['#ff7f0e' if val >= 0 else '#ffcc99' for val in time_to_threshold])
    
    asymptotic_bars = ax.bar(x + width, asymptotic, width, label='Asymptotic Performance',
                            color=['#2ca02c' if val >= 0 else '#99ff99' for val in asymptotic])
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height < 0:
                height = bar.get_y()
                va = 'top'
            else:
                va = 'bottom'
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va=va,
                        fontsize=9)
    
    add_labels(jumpstart_bars)
    add_labels(threshold_bars)
    add_labels(asymptotic_bars)
    
    ax.set_ylabel('Improvement over Baseline', fontsize=12)
    ax.set_title('Improved Ant Environment: Transfer Learning Mechanism Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(mechanism_names, rotation=15, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
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