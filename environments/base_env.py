import numpy as np
import gymnasium as gym
from gymnasium import spaces
import json
import pickle
import os
import time
from abc import ABC, abstractmethod

class BaseEnvironment(ABC):
    """
    Abstract base class for all environments in the transfer learning framework.
    
    This class defines the standard interface for environments and provides common
    functionality that applies to both discrete and continuous domains.
    """
    
    def __init__(self, config):
        """
        Initialize the base environment with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with environment parameters
        """
        self.config = config
        
        # Extract common configuration parameters
        self.complexity_level = config.get("complexity_level", 1.0)
        self.max_episode_steps = config.get("max_episode_steps", 1000)
        self.reward_scale = config.get("reward_scale", 1.0)
        self.seed = config.get("seed", None)
        
        # Initialize metrics tracking
        self.episode_count = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.start_time = time.time()
        
        # Initialize spaces (to be defined by subclasses)
        self.observation_space = None
        self.action_space = None
        
        # Set random seed if provided
        if self.seed is not None:
            self.set_seed(self.seed)
    
    @abstractmethod
    def reset(self):
        """
        Reset environment to initial state.
        
        Returns:
            observation: Initial observation after reset
        """
        pass
    
    @abstractmethod
    def step(self, action):
        """
        Take a step in environment with given action.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        pass
    
    @abstractmethod
    def get_state_representation(self):
        """
        Get the state representation suitable for transfer learning.
        
        This should extract the most important features from the current state
        that would be useful for transfer between different but related environments.
        
        Returns:
            numpy.ndarray: Feature representation of current state
        """
        pass
    
    @property
    def complexity(self):
        """Return the complexity level of this environment."""
        return self.complexity_level
    
    @abstractmethod
    def increase_complexity(self, increment=0.1):
        """
        Increase the environment complexity.
        
        Args:
            increment (float): Amount to increase complexity by
            
        Returns:
            float: New complexity level
        """
        pass
    
    def set_seed(self, seed):
        """
        Set random seed for reproducibility.
        
        Args:
            seed (int): Random seed
            
        Returns:
            list: List of seeds used in the environment
        """
        np.random.seed(seed)
        # If using gymnasium
        if hasattr(self, 'np_random'):
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def get_env_parameters(self):
        """
        Get environment parameters relevant for transfer learning.
        
        Returns:
            dict: Dictionary of transferable parameters
        """
        return {
            "complexity_level": self.complexity_level,
            "observation_shape": self.observation_space.shape if hasattr(self.observation_space, 'shape') else None,
            "action_shape": self.action_space.shape if hasattr(self.action_space, 'shape') else None,
            "action_space_type": "discrete" if isinstance(self.action_space, spaces.Discrete) else "continuous",
            "config": {k: v for k, v in self.config.items() if isinstance(v, (int, float, str, bool, list, tuple))}
        }
    
    def get_metrics(self):
        """
        Get environment performance metrics.
        
        Returns:
            dict: Dictionary of environment metrics
        """
        return {
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "avg_episode_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "avg_episode_length": np.mean(self.episode_lengths) if self.episode_lengths else 0,
            "runtime_seconds": time.time() - self.start_time
        }
    
    def track_episode(self, episode_reward, episode_length):
        """
        Track metrics for completed episode.
        
        Args:
            episode_reward (float): Total reward for the episode
            episode_length (int): Number of steps in the episode
        """
        self.episode_count += 1
        self.total_steps += episode_length
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
    
    def normalize_observation(self, observation):
        """
        Normalize observation to standard range.
        
        Args:
            observation: Raw observation from environment
            
        Returns:
            numpy.ndarray: Normalized observation
        """
        # Default implementation - override in subclasses for specific normalization
        return observation
    
    def save_state(self, path):
        """
        Save environment state to file.
        
        Args:
            path (str): Path to save state file
            
        Returns:
            bool: Success status
        """
        try:
            # Save configuration and metrics
            state_dict = {
                "config": self.config,
                "metrics": self.get_metrics(),
                "complexity_level": self.complexity_level,
                "episode_count": self.episode_count,
                "total_steps": self.total_steps
            }
            
            # Save state dict
            with open(path, 'wb') as f:
                pickle.dump(state_dict, f)
            
            return True
        except Exception as e:
            print(f"Error saving environment state: {e}")
            return False
    
    def load_state(self, path):
        """
        Load environment state from file.
        
        Args:
            path (str): Path to state file
            
        Returns:
            bool: Success status
        """
        try:
            # Load state dict
            with open(path, 'rb') as f:
                state_dict = pickle.load(f)
            
            # Update configuration and metrics
            self.config = state_dict["config"]
            self.complexity_level = state_dict["complexity_level"]
            self.episode_count = state_dict["episode_count"]
            self.total_steps = state_dict["total_steps"]
            
            return True
        except Exception as e:
            print(f"Error loading environment state: {e}")
            return False
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            Varies: Rendering of the environment
        """
        # Default implementation that subclasses can override
        print("Rendering not implemented for this environment")
        return None
    
    def get_environment_info(self):
        """
        Get detailed information about the environment.
        
        Returns:
            dict: Detailed environment information
        """
        return {
            "name": self.__class__.__name__,
            "complexity_level": self.complexity_level,
            "observation_space": str(self.observation_space),
            "action_space": str(self.action_space),
            "metrics": self.get_metrics(),
            "parameters": self.get_env_parameters()
        }
    
    def calculate_similarity(self, other_env):
        """
        Calculate similarity score between this environment and another.
        
        This is useful for determining how well knowledge might transfer between environments.
        
        Args:
            other_env (BaseEnvironment): Another environment to compare with
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Default implementation using basic environment properties
        # More sophisticated methods could be implemented in subclasses
        
        # Check if environments are of the same type
        same_type = self.__class__.__name__ == other_env.__class__.__name__
        type_score = 1.0 if same_type else 0.5
        
        # Compare complexity levels
        complexity_diff = abs(self.complexity_level - other_env.complexity_level)
        complexity_score = 1.0 - min(complexity_diff, 1.0)
        
        # Compare action spaces
        action_type_match = type(self.action_space) == type(other_env.action_space)
        action_score = 1.0 if action_type_match else 0.0
        
        # Compare observation spaces
        obs_type_match = type(self.observation_space) == type(other_env.observation_space)
        obs_score = 1.0 if obs_type_match else 0.0
        
        # Weighted similarity score
        similarity = (
            0.4 * type_score +
            0.3 * complexity_score +
            0.15 * action_score +
            0.15 * obs_score
        )
        
        return similarity
    
    def create_environment_variants(self, num_variants=3):
        """
        Create variants of this environment with different complexity levels.
        
        This is useful for curriculum learning experiments.
        
        Args:
            num_variants (int): Number of environment variants to create
            
        Returns:
            list: List of environment configurations with varying complexity
        """
        variants = []
        
        # Create variants with evenly spaced complexity levels
        for i in range(num_variants):
            # Calculate complexity level for this variant
            complexity = (i + 1) / num_variants
            
            # Create a copy of the current config
            variant_config = self.config.copy()
            
            # Update complexity level
            variant_config["complexity_level"] = complexity
            
            variants.append(variant_config)
        
        return variants