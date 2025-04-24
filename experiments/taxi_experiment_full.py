import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from run_experiment import ExperimentRunner

# Get the absolute path of the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the project root to the Python path
sys.path.insert(0, project_root)

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def run_comprehensive_taxi_experiment(num_trials=5, num_episodes=1000, save_dir="results/comprehensive_taxi"):
    """Run comprehensive Ant transfer experiment with multiple trials."""
    
    # Results collection
    all_trial_results = {
        'baseline': [],
        'parameter_transfer': [],
        'value_transfer': [],
        'reward_shaping': []
    }
    
    # Run multiple trials
    for trial in range(num_trials):
        print(f"\n===== Starting Trial {trial+1}/{num_trials} =====")
        
        # Set unique seed for this trial
        trial_seed = 42 + trial
        
        # Configure experiment
        experiment_config = create_experiment_config(
            num_episodes=num_episodes,
            seed=trial_seed
        )
        
        # Create results directory
        trial_dir = os.path.join(save_dir, f"trial_{trial+1}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # Run experiment
        runner = ExperimentRunner(experiment_config, output_dir=trial_dir)
        trial_results = runner.run()
        
        # Store results
        for method_name, result in trial_results.items():
            all_trial_results[method_name].append(result)
        
        # Save trial results
        with open(os.path.join(trial_dir, "trial_results.json"), 'w') as f:
            json.dump(trial_results, f, indent=4, cls=NumpyEncoder)
    
    # Analyze and visualize aggregated results
    aggregate_and_analyze_results(all_trial_results, save_dir)
    
    return all_trial_results

def create_experiment_config(num_episodes=1000, seed=42):
    """Create comprehensive experiment configuration."""
    return {
        'name': 'comprehensive_taxi_transfer_experiment',
        'num_episodes': num_episodes,
        'eval_frequency': 10,
        'eval_episodes': 5,

        # Target environment configuration (full Taxi)
        'target_env_config': {
            'type': 'taxi',
            'grid_size': 5,
            'num_passengers': 1,
            # 'use_fixed_locations': True
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
                'source_episodes': 500,
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
                    'num_passengers': 1,
                    # 'use_fixed_locations': True
                },
                'source_agent_config': {
                    'type': 'q_learning',
                    'learning_rate': 0.1,
                    'discount_factor': 0.99,
                    'exploration_rate': 0.1,
                    'exploration_decay': 0.995
                },
                'source_episodes': num_episodes,
                'mechanism_config': {
                    'type': 'value_transfer',
                    'transfer_type': 'q_values',
                    'use_state_mapping': True,
                    'adaptation_method': 'normalized',
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
                'source_episodes': num_episodes,
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

def aggregate_and_analyze_results(all_trial_results, save_dir):
    """Aggregate results across trials and perform statistical analysis."""
    # Create directory for aggregated results
    agg_dir = os.path.join(save_dir, "aggregated_results")
    os.makedirs(agg_dir, exist_ok=True)
    
    # Aggregated metrics storage
    aggregated_metrics = {}
    
    # Process each transfer method
    for method_name, trial_results in all_trial_results.items():
        if method_name == 'baseline':
            continue
            
        # Extract metrics from all trials
        jumpstart_values = []
        asymptotic_values = []
        transfer_ratio_values = []
        time_to_threshold_values = []
        
        for result in trial_results:
            metrics = result.get('metrics', {})
            jumpstart_values.append(metrics.get('jumpstart', 0))
            asymptotic_values.append(metrics.get('asymptotic', 0))
            transfer_ratio_values.append(metrics.get('transfer_ratio', 1.0))
            
            if 'time_to_threshold' in metrics:
                improvement = metrics['time_to_threshold'].get('improvement', 0)
                time_to_threshold_values.append(improvement)
        
        # Calculate aggregate statistics
        aggregated_metrics[method_name] = {
            'jumpstart': {
                'mean': np.mean(jumpstart_values),
                'std': np.std(jumpstart_values),
                'values': jumpstart_values
            },
            'asymptotic': {
                'mean': np.mean(asymptotic_values),
                'std': np.std(asymptotic_values),
                'values': asymptotic_values
            },
            'transfer_ratio': {
                'mean': np.mean(transfer_ratio_values),
                'std': np.std(transfer_ratio_values),
                'values': transfer_ratio_values
            }
        }
        
        if time_to_threshold_values:
            aggregated_metrics[method_name]['time_to_threshold'] = {
                'mean': np.mean(time_to_threshold_values),
                'std': np.std(time_to_threshold_values),
                'values': time_to_threshold_values
            }
    
    # Save aggregated metrics
    with open(os.path.join(agg_dir, "aggregated_metrics.json"), 'w') as f:
        json.dump(aggregated_metrics, f, indent=4, cls=NumpyEncoder)
    
    # Create comparison visualizations
    visualize_aggregated_results(aggregated_metrics, agg_dir)
    
    # Perform statistical tests
    perform_statistical_tests(aggregated_metrics, agg_dir)

def visualize_aggregated_results(aggregated_metrics, save_dir):
    """Create visualizations for aggregated results."""
    # Bar chart with error bars for key metrics
    plt.figure(figsize=(15, 10))
    
    methods = list(aggregated_metrics.keys())
    x = np.arange(len(methods))
    width = 0.25
    
    # Plot jumpstart performance
    jumpstart_means = [aggregated_metrics[m]['jumpstart']['mean'] for m in methods]
    jumpstart_stds = [aggregated_metrics[m]['jumpstart']['std'] for m in methods]
    plt.bar(x - width, jumpstart_means, width, label='Jumpstart Performance', 
            yerr=jumpstart_stds, capsize=5, color='#1f77b4')
    
    # Plot asymptotic performance
    asymptotic_means = [aggregated_metrics[m]['asymptotic']['mean'] for m in methods]
    asymptotic_stds = [aggregated_metrics[m]['asymptotic']['std'] for m in methods]
    plt.bar(x, asymptotic_means, width, label='Asymptotic Performance', 
            yerr=asymptotic_stds, capsize=5, color='#ff7f0e')
    
    # Plot transfer ratio
    transfer_ratio_means = [aggregated_metrics[m]['transfer_ratio']['mean'] for m in methods]
    transfer_ratio_stds = [aggregated_metrics[m]['transfer_ratio']['std'] for m in methods]
    # Scale transfer ratios to be on similar scale as other metrics
    scaled_ratio_means = [(tr - 1.0) * 10 for tr in transfer_ratio_means]
    scaled_ratio_stds = [std * 10 for std in transfer_ratio_stds]
    plt.bar(x + width, scaled_ratio_means, width, label='Transfer Ratio (scaled)', 
            yerr=scaled_ratio_stds, capsize=5, color='#2ca02c')
    
    # Add labels and legend
    plt.xlabel('Transfer Method', fontsize=14)
    plt.ylabel('Performance Metric', fontsize=14)
    plt.title('Comparison of Transfer Learning Methods (Mean ± Std across trials)', fontsize=16)
    plt.xticks(x, methods, rotation=15, ha='right', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "transfer_methods_comparison.png"), dpi=300)
    plt.close()
    
    # Create boxplot visualization for each metric
    metrics = ['jumpstart', 'asymptotic', 'transfer_ratio']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Extract values for boxplot
        boxplot_data = [aggregated_metrics[m][metric]['values'] for m in methods]
        
        # Create boxplot
        plt.boxplot(boxplot_data, tick_labels=methods, patch_artist=True,
                   boxprops=dict(facecolor='#9999ff', alpha=0.7),
                   medianprops=dict(color='black'))
        
        plt.title(f'{metric.capitalize()} Performance Across Trials', fontsize=16)
        plt.ylabel(f'{metric.capitalize()} Value', fontsize=14)
        plt.xticks(rotation=15, ha='right', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add a reference line
        if metric == 'transfer_ratio':
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, 
                       label='No Transfer Benefit Threshold')
        else:
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7,
                       label='No Transfer Benefit Threshold')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric}_boxplot.png"), dpi=300)
        plt.close()

def perform_statistical_tests(aggregated_metrics, save_dir):
    """Perform statistical tests to compare transfer methods."""
    from scipy import stats
    
    methods = list(aggregated_metrics.keys())
    metrics = ['jumpstart', 'asymptotic', 'transfer_ratio']
    
    # Create a table of statistics
    stat_results = {}
    
    for metric in metrics:
        stat_results[metric] = {}
        
        # Perform one-sample t-test for each method against null hypothesis
        for method in methods:
            values = aggregated_metrics[method][metric]['values']
            if metric == 'transfer_ratio':
                t_stat, p_value = stats.ttest_1samp(values, 1.0)
            else:
                t_stat, p_value = stats.ttest_1samp(values, 0.0)
            
            stat_results[metric][method] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'mean': np.mean(values),
                'std': np.std(values),
                'effect_size': np.mean(values) / np.std(values) if np.std(values) > 0 else float('inf')
            }
        
        # Perform paired t-tests between methods
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                values1 = aggregated_metrics[method1][metric]['values']
                values2 = aggregated_metrics[method2][metric]['values']
                
                # Ensure equal length for paired test
                min_len = min(len(values1), len(values2))
                t_stat, p_value = stats.ttest_rel(values1[:min_len], values2[:min_len])
                
                stat_results[metric][f"{method1}_vs_{method2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
    
    # Save statistical results
    with open(os.path.join(save_dir, "statistical_tests.json"), 'w') as f:
        json.dump(stat_results, f, indent=4, cls=NumpyEncoder)
    
    # Create readable summary
    summary_text = "Statistical Analysis Summary\n"
    summary_text += "=========================\n\n"
    
    for metric in metrics:
        summary_text += f"{metric.capitalize()} Performance:\n"
        summary_text += "-" * (len(metric) + 12) + "\n"
        
        # Single method results
        summary_text += "Individual Method Performance:\n"
        for method in methods:
            results = stat_results[metric][method]
            sig_symbol = "*" if results['significant'] else ""
            summary_text += f"  - {method}: mean={results['mean']:.4f}, std={results['std']:.4f}, p={results['p_value']:.4f}{sig_symbol}\n"
        
        # Method comparisons
        summary_text += "\nMethod Comparisons:\n"
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                results = stat_results[metric][f"{method1}_vs_{method2}"]
                sig_symbol = "*" if results['significant'] else ""
                summary_text += f"  - {method1} vs {method2}: t={results['t_statistic']:.4f}, p={results['p_value']:.4f}{sig_symbol}\n"
        
        summary_text += "\n"
    
    summary_text += "Note: * indicates statistical significance at p < 0.05\n"
    
    # Save summary
    with open(os.path.join(save_dir, "statistical_summary.txt"), 'w') as f:
        f.write(summary_text)

def main():
    """Run the comprehensive Ant transfer learning experiment."""
    print("Starting Comprehensive Ant Transfer Learning Experiment")
    
    # Parameters
    num_trials = 50  # Run 5 trials with different seeds
    num_episodes = 500  # Run 1000 episodes per experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/comprehensive_taxi_{timestamp}"
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Save experiment parameters
    with open(os.path.join(save_dir, "experiment_params.json"), 'w') as f:
        json.dump({
            "num_trials": num_trials,
            "num_episodes": num_episodes,
            "timestamp": timestamp
        }, f, indent=4)
    
    # Run experiment
    all_results = run_comprehensive_taxi_experiment(
        num_trials=num_trials,
        num_episodes=num_episodes,
        save_dir=save_dir
    )
    
    print(f"Experiment completed. Results saved to {save_dir}")

if __name__ == "__main__":
    main()