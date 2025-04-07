import numpy as np
from scipy import stats

class TransferMetrics:
    """Metrics for evaluating transfer learning effectiveness."""
    
    @staticmethod
    def jumpstart_performance(baseline_rewards, transfer_rewards):
        """Calculate improvement in initial performance.
        
        Args:
            baseline_rewards: Rewards from learning from scratch (first N episodes)
            transfer_rewards: Rewards from transfer learning (first N episodes)
            
        Returns:
            Float: Jumpstart performance metric
        """
        # Take the first few episodes
        n_episodes = min(10, len(baseline_rewards), len(transfer_rewards))
        baseline_initial = np.mean(baseline_rewards[:n_episodes])
        transfer_initial = np.mean(transfer_rewards[:n_episodes])
        
        return transfer_initial - baseline_initial
    
    @staticmethod
    def asymptotic_performance(baseline_rewards, transfer_rewards):
        """Calculate improvement in final performance.
        
        Args:
            baseline_rewards: Rewards from learning from scratch
            transfer_rewards: Rewards from transfer learning
            
        Returns:
            Float: Asymptotic performance improvement
        """
        # Take the last 10% of episodes
        n_episodes = max(10, int(0.1 * len(baseline_rewards)))
        baseline_final = np.mean(baseline_rewards[-n_episodes:])
        transfer_final = np.mean(transfer_rewards[-n_episodes:])
        
        return transfer_final - baseline_final
    
    @staticmethod
    def time_to_threshold(baseline_rewards, transfer_rewards, threshold):
        """Calculate episodes needed to reach a performance threshold.
        
        Args:
            baseline_rewards: Rewards from learning from scratch
            transfer_rewards: Rewards from transfer learning
            threshold: Performance threshold to reach
            
        Returns:
            Tuple[int, int]: Episodes to threshold for baseline and transfer
        """
        # Apply smoothing for stability
        window = max(1, int(0.05 * len(baseline_rewards)))
        baseline_smoothed = np.convolve(baseline_rewards, np.ones(window)/window, mode='valid')
        transfer_smoothed = np.convolve(transfer_rewards, np.ones(window)/window, mode='valid')
        
        # Find first episode that crosses threshold
        baseline_thresh = next((i for i, r in enumerate(baseline_smoothed) if r >= threshold), len(baseline_rewards))
        transfer_thresh = next((i for i, r in enumerate(transfer_smoothed) if r >= threshold), len(transfer_rewards))
        
        return baseline_thresh, transfer_thresh
    
    @staticmethod
    def transfer_ratio(baseline_rewards, transfer_rewards):
        """Calculate ratio of area under learning curves.
        
        Args:
            baseline_rewards: Rewards from learning from scratch
            transfer_rewards: Rewards from transfer learning
            
        Returns:
            Float: Transfer ratio (>1 means positive transfer)
        """
        # Ensure same length
        min_length = min(len(baseline_rewards), len(transfer_rewards))
        baseline_area = np.sum(baseline_rewards[:min_length])
        transfer_area = np.sum(transfer_rewards[:min_length])
        
        if baseline_area == 0:
            return float('inf') if transfer_area > 0 else 0.0
            
        return transfer_area / baseline_area
    
    @staticmethod
    def statistical_significance(baseline_rewards, transfer_rewards):
        """Calculate statistical significance of transfer learning effect.
        
        Args:
            baseline_rewards: Rewards from learning from scratch
            transfer_rewards: Rewards from transfer learning
            
        Returns:
            Dict: Statistical test results
        """
        # Perform Mann-Whitney U test (non-parametric)
        u_stat, p_value = stats.mannwhitneyu(transfer_rewards, baseline_rewards, alternative='greater')
        
        return {
            'u_statistic': u_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def detect_negative_transfer(baseline_rewards, transfer_rewards):
        """Detect if transfer is harmful.
        
        Args:
            baseline_rewards: Rewards from learning from scratch
            transfer_rewards: Rewards from transfer learning
            
        Returns:
            Dict: Negative transfer detection results
        """
        # Compare AUC for first third of training
        third_point = min(len(baseline_rewards), len(transfer_rewards)) // 3
        early_baseline = np.sum(baseline_rewards[:third_point])
        early_transfer = np.sum(transfer_rewards[:third_point])
        
        # Check if transfer is significantly worse
        is_negative = early_transfer < 0.9 * early_baseline
        
        return {
            'negative_transfer_detected': is_negative,
            'early_baseline_performance': early_baseline / third_point,
            'early_transfer_performance': early_transfer / third_point,
            'relative_performance': early_transfer / early_baseline if early_baseline else float('inf')
        }