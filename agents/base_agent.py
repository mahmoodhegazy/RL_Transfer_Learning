import numpy as np
import torch
import pickle
import os
import time
from abc import ABC, abstractmethod
import gymnasium as gym
from typing import Dict, Any, Tuple, List, Union, Optional

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the transfer learning framework.
    
    This class defines the standard interface for agents and provides common
    functionality that applies to both discrete and continuous domains.
    Subclasses must implement the abstract methods.
    """
    
    def __init__(self, env, config):
        """
        Initialize the base agent with environment and configuration.
        
        Args:
            env: The environment the agent will interact with
            config (dict): Configuration dictionary with agent parameters
        """
        self.env = env
        self.config = config
        self.training_steps = 0
        self.episode_counter = 0
        self.total_rewards = []
        self.episode_lengths = []
        self.start_time = time.time()
        
        # Extract common parameters from config
        self.name = config.get("name", self.__class__.__name__)
        self.learning_rate = config.get("learning_rate", 0.01)
        self.discount_factor = config.get("discount_factor", 0.99)
        self.batch_size = config.get("batch_size", 64)
        self.exploration_rate = config.get("exploration_rate", 0.1)
        self.exploration_decay = config.get("exploration_decay", 0.995)
        self.min_exploration_rate = config.get("min_exploration_rate", 0.01)
        self.training = config.get("training", True)
        self.device = self._get_device()
        
        # Initialize training metrics
        self.metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "learning_rates": [],
            "exploration_rates": [],
            "avg_q_values": [],
            "loss_values": []
        }
    
    def _get_device(self) -> torch.device:
        """
        Determine the appropriate compute device.
        
        Returns:
            torch.device: The device to use for tensor operations
        """
        if torch.cuda.is_available() and self.config.get("use_cuda", True):
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and self.config.get("use_mps", True):
            return torch.device("mps")  # For Apple Silicon
        else:
            return torch.device("cpu")
        
    @abstractmethod
    def select_action(self, state):
        """
        Select an action based on current state.
        
        Args:
            state: The current state of the environment
            
        Returns:
            The selected action
        """
        raise NotImplementedError
        
    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """
        Update agent's knowledge after interaction.
        
        Args:
            state: The current state
            action: The action taken
            reward: The reward received
            next_state: The resulting next state
            done: Whether the episode is done
            
        Returns:
            None
        """
        raise NotImplementedError
    
    def train_episode(self) -> Tuple[float, int]:
        """
        Train the agent for a single episode.
        
        Returns:
            Tuple[float, int]: Total episode reward and episode length
        """
        state = self.env.reset()[0] if isinstance(self.env.reset(), tuple) else self.env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        while not (done or truncated):
            # Select and perform action
            action = self.select_action(state)
            
            # Handle both gym and gymnasium API formats
            if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'spec') and \
               hasattr(self.env.unwrapped.spec, 'version') and \
               self.env.unwrapped.spec.version.startswith('0.'):
                # Old OpenAI Gym format
                next_state, reward, done, info = self.env.step(action)
                truncated = False
            else:
                # New Gymnasium format
                next_state, reward, done, truncated, info = self.env.step(action)
            
            # Update agent knowledge
            self.update(state, action, reward, next_state, done or truncated)
            
            # Transition to next state
            state = next_state
            total_reward += reward
            steps += 1
            
            # Optional render for visualization
            if self.config.get("render", False) and steps % self.config.get("render_frequency", 1) == 0:
                self.env.render()
        
        # Update episode metrics
        self.episode_counter += 1
        self.metrics["episode_rewards"].append(total_reward)
        self.metrics["episode_lengths"].append(steps)
        self.metrics["exploration_rates"].append(self.exploration_rate if hasattr(self, 'exploration_rate') else 0)
        self.metrics["learning_rates"].append(self.learning_rate if hasattr(self, 'learning_rate') else 0)
        
        return total_reward, steps
    
    def train(self, num_episodes: int, evaluate_freq: int = 10) -> Dict[str, Any]:
        """
        Train the agent for a specified number of episodes.
        
        Args:
            num_episodes (int): Number of episodes to train
            evaluate_freq (int): Frequency of evaluation during training
            
        Returns:
            Dict[str, Any]: Training metrics
        """
        print(f"Starting training for {num_episodes} episodes...")
        
        evaluation_rewards = []
        
        for episode in range(num_episodes):
            # Train for one episode
            episode_reward, episode_length = self.train_episode()
            
            # Evaluate periodically
            if evaluate_freq > 0 and episode % evaluate_freq == 0:
                eval_reward = self.evaluate(self.config.get("eval_episodes", 5))
                evaluation_rewards.append((episode, eval_reward))
                
                # Log progress
                print(f"Episode {episode}/{num_episodes}, Train reward: {episode_reward:.2f}, " 
                      f"Eval reward: {eval_reward:.2f}, Steps: {episode_length}")
            elif episode % 10 == 0:
                # Log progress without evaluation
                print(f"Episode {episode}/{num_episodes}, Reward: {episode_reward:.2f}, Steps: {episode_length}")
        
        # Compile training results
        training_duration = time.time() - self.start_time
        
        results = {
            "episode_rewards": self.metrics["episode_rewards"],
            "episode_lengths": self.metrics["episode_lengths"],
            "evaluation_rewards": evaluation_rewards,
            "final_exploration_rate": self.exploration_rate if hasattr(self, 'exploration_rate') else None,
            "training_steps": self.training_steps,
            "training_episodes": self.episode_counter,
            "training_duration": training_duration
        }
        
        print(f"Training completed. Total steps: {self.training_steps}, "
              f"Average reward: {np.mean(self.metrics['episode_rewards'][-100:]):.2f}")
        
        return results
    
    def evaluate(self, num_episodes: int = 5) -> float:
        """
        Evaluate agent performance without exploration.
        
        Args:
            num_episodes (int): Number of episodes to evaluate
            
        Returns:
            float: Average evaluation reward
        """
        # Save original exploration settings
        original_exploration = None
        if hasattr(self, 'exploration_rate'):
            original_exploration = self.exploration_rate
            self.exploration_rate = 0
        
        # Set to evaluation mode
        original_training = self.training
        self.training = False
        
        total_rewards = []
        
        for _ in range(num_episodes):
            state = self.env.reset()[0] if isinstance(self.env.reset(), tuple) else self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            while not (done or truncated):
                action = self.select_action(state)
                
                # Handle both gym and gymnasium API formats
                if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'spec') and \
                   hasattr(self.env.unwrapped.spec, 'version') and \
                   self.env.unwrapped.spec.version.startswith('0.'):
                    # Old OpenAI Gym format
                    next_state, reward, done, _ = self.env.step(action)
                    truncated = False
                else:
                    # New Gymnasium format
                    next_state, reward, done, truncated, _ = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                
            total_rewards.append(episode_reward)
        
        # Restore original settings
        if original_exploration is not None:
            self.exploration_rate = original_exploration
        
        self.training = original_training
        
        return np.mean(total_rewards)

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save agent's parameters to file.
        
        Args:
            path (str): Path to save the agent
            
        Returns:
            None
        """
        raise NotImplementedError
        
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load agent's parameters from file.
        
        Args:
            path (str): Path to load the agent from
            
        Returns:
            None
        """
        raise NotImplementedError
    
    def get_state_representation(self, state: Any) -> np.ndarray:
        """
        Get a standardized state representation suitable for transfer learning.
        
        This method converts environment-specific state representations into a
        standardized format that can be used for knowledge transfer between
        similar environments.
        
        Args:
            state: The state to convert
            
        Returns:
            np.ndarray: Standardized state representation
        """
        # Default implementation - should be overridden by subclasses
        # for environment-specific representations
        if hasattr(self.env, 'get_state_representation'):
            return self.env.get_state_representation(state)
        
        # Convert state to numpy array if it isn't already
        if not isinstance(state, np.ndarray):
            if isinstance(state, (list, tuple)):
                state = np.array(state, dtype=np.float32)
            elif isinstance(state, (int, float)):
                state = np.array([state], dtype=np.float32)
            elif isinstance(state, dict):
                # Handle dictionary observations by flattening
                values = []
                for k in sorted(state.keys()):
                    values.append(state[k])
                state = np.concatenate([v.flatten() if isinstance(v, np.ndarray) else np.array([v]) 
                                      for v in values])
        
        # Attempt basic normalization if possible
        if hasattr(self.env, 'observation_space'):
            if hasattr(self.env.observation_space, 'high') and hasattr(self.env.observation_space, 'low'):
                # Normalize to [0, 1] range using observation space bounds
                high = self.env.observation_space.high
                low = self.env.observation_space.low
                
                # Handle infinity bounds
                high_mask = np.isfinite(high)
                low_mask = np.isfinite(low)
                mask = np.logical_and(high_mask, low_mask)
                
                # Only normalize finite dimensions
                if np.any(mask):
                    normalized_state = state.copy()
                    normalized_state[mask] = (state[mask] - low[mask]) / (high[mask] - low[mask])
                    return normalized_state
        
        # If normalization not possible, return as is
        return state
    
    @abstractmethod
    def extract_knowledge(self, knowledge_type: str) -> Dict[str, Any]:
        """
        Extract agent's knowledge for transfer learning.
        
        This method extracts relevant knowledge from the agent that can be 
        transferred to another agent. The type of knowledge to extract
        is specified by the knowledge_type parameter.
        
        Args:
            knowledge_type (str): Type of knowledge to extract 
                                 (e.g., "parameters", "q_values", "policy")
            
        Returns:
            Dict[str, Any]: Extracted knowledge
        """
        raise NotImplementedError
    
    @abstractmethod
    def initialize_from_knowledge(self, knowledge: Dict[str, Any]) -> None:
        """
        Initialize agent using transferred knowledge.
        
        This method initializes the agent using knowledge transferred
        from another agent.
        
        Args:
            knowledge (Dict[str, Any]): Transferred knowledge
            
        Returns:
            None
        """
        raise NotImplementedError
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent's performance metrics.
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        return {
            "training_steps": self.training_steps,
            "episodes": self.episode_counter,
            "avg_reward_last_100": np.mean(self.metrics["episode_rewards"][-100:]) if self.metrics["episode_rewards"] else 0,
            "avg_episode_length": np.mean(self.metrics["episode_lengths"]) if self.metrics["episode_lengths"] else 0,
            "training_time": time.time() - self.start_time
        }
    
    def _decay_exploration_rate(self) -> None:
        """
        Decay the exploration rate according to the set schedule.
        
        Returns:
            None
        """
        if hasattr(self, 'exploration_rate') and hasattr(self, 'exploration_decay'):
            self.exploration_rate = max(
                self.min_exploration_rate,
                self.exploration_rate * self.exploration_decay
            )
    
    def _compute_returns(self, rewards: List[float], gamma: Optional[float] = None) -> np.ndarray:
        """
        Compute discounted returns for a sequence of rewards.
        
        Args:
            rewards (List[float]): List of rewards
            gamma (Optional[float]): Discount factor (uses self.discount_factor if None)
            
        Returns:
            np.ndarray: Array of discounted returns
        """
        if gamma is None:
            gamma = self.discount_factor
            
        returns = []
        G = 0
        
        # Calculate returns in reverse (from last step to first)
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        
        return np.array(returns)
    
    def reset(self) -> None:
        """
        Reset the agent's state (but not its learned parameters).
        
        Returns:
            None
        """
        # Reset episode-specific variables, but keep learned knowledge
        self.training_steps = 0
        self.episode_counter = 0
        self.start_time = time.time()
        
        # Reset metrics
        for key in self.metrics:
            self.metrics[key] = []

    def _save_config(self, path: str) -> None:
        """
        Save agent configuration to file.
        
        Args:
            path (str): Path to save configuration
            
        Returns:
            None
        """
        # Extract serializable config (exclude env and other non-serializable items)
        serializable_config = {k: v for k, v in self.config.items() 
                               if k != 'env' and not callable(v)}
        
        # Save to file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(serializable_config, f)
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        """
        Load agent configuration from file.
        
        Args:
            path (str): Path to load configuration from
            
        Returns:
            Dict[str, Any]: Loaded configuration
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def _adapt_network_dimensions(self, 
                                 source_shape: Tuple[int, ...], 
                                 target_shape: Tuple[int, ...], 
                                 source_params: np.ndarray) -> np.ndarray:
        """
        Adapt neural network parameters between different dimensions.
        
        This is a utility method for transfer learning between networks with
        different input/output dimensions.
        
        Args:
            source_shape (Tuple[int, ...]): Shape of source parameters
            target_shape (Tuple[int, ...]): Shape of target parameters
            source_params (np.ndarray): Source parameters
            
        Returns:
            np.ndarray: Adapted parameters
        """
        if source_shape == target_shape:
            return source_params.copy()
        
        # For 2D weight matrices (common in neural networks)
        if len(source_shape) == 2 and len(target_shape) == 2:
            target_params = np.zeros(target_shape)
            
            # Copy the overlapping part
            min_rows = min(source_shape[0], target_shape[0])
            min_cols = min(source_shape[1], target_shape[1])
            
            target_params[:min_rows, :min_cols] = source_params[:min_rows, :min_cols]
            
            # For any remaining rows/columns, initialize with small random values
            if target_shape[0] > min_rows:
                # Initialize remaining rows
                scale = np.std(source_params) * 0.1  # Small random values based on source scale
                target_params[min_rows:, :min_cols] = np.random.randn(
                    target_shape[0] - min_rows, min_cols) * scale
            
            if target_shape[1] > min_cols:
                # Initialize remaining columns
                scale = np.std(source_params) * 0.1
                target_params[:, min_cols:] = np.random.randn(
                    target_shape[0], target_shape[1] - min_cols) * scale
            
            return target_params
        
        # For 1D bias vectors
        elif len(source_shape) == 1 and len(target_shape) == 1:
            target_params = np.zeros(target_shape)
            min_dim = min(source_shape[0], target_shape[0])
            
            target_params[:min_dim] = source_params[:min_dim]
            return target_params
        
        # For other parameter shapes, just create a new array
        else:
            return np.zeros(target_shape)
    
    def __str__(self) -> str:
        """
        Get string representation of the agent.
        
        Returns:
            str: String representation
        """
        return (f"{self.name} Agent - "
                f"Steps: {self.training_steps}, "
                f"Episodes: {self.episode_counter}, "
                f"Avg reward: {np.mean(self.metrics['episode_rewards'][-100:]) if self.metrics['episode_rewards'] else 0:.2f}")