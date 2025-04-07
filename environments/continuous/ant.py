import numpy as np
import gym
from gymnasium import spaces
from environments.base_env import BaseEnvironment

class AntEnv(BaseEnvironment):
    """
    Full Ant environment with customizable complexity levels.
    
    Complexity can be adjusted through:
    - Physics parameters (friction, gravity)
    - Observation noise
    - Action noise
    - Control frequency
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Import here to avoid dependency issues if MuJoCo is not installed
        try:
            import gymnasium as gym
            import mujoco
        except ImportError:
            raise ImportError("To use AntEnv, please install gymnasium[mujoco]")
        
        # Extract configuration parameters
        self.friction = config.get("friction", 1.0)
        self.gravity = config.get("gravity", -9.81)
        self.observation_noise = config.get("observation_noise", 0.0)
        self.action_noise = config.get("action_noise", 0.0)
        self.control_frequency = config.get("control_frequency", 5)
        
        # Create and modify base environment
        self.base_env = gym.make('Ant-v4')
        self.modify_env_parameters()
        
        # Set spaces to match base environment
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        
        # Additional reward components
        self.energy_penalty_coef = config.get("energy_penalty_coef", 0.5)
        self.forward_reward_weight = config.get("forward_reward_weight", 1.0)
        self.goal_position = config.get("goal_position", None)  # Optional goal-based reward
        
    def modify_env_parameters(self):
        """Modify MuJoCo environment parameters based on configuration."""
        # This requires modifying the MuJoCo model parameters
        # Here we modify the environment's physics parameters
        model = self.base_env.unwrapped.model
        
        # Set friction
        for i in range(model.geom_friction.shape[0]):
            model.geom_friction[i, 0] = self.friction
        
        # Set gravity
        model.opt.gravity[2] = self.gravity
    
    def reset(self):
        """Reset the environment to initial state."""
        observation, info = self.base_env.reset()
        
        # Apply observation noise if configured
        if self.observation_noise > 0:
            noise = np.random.normal(0, self.observation_noise, observation.shape)
            observation += noise
        
        return observation
    
    def step(self, action):
        """Take a step in the environment with the given action."""
        # Apply action noise if configured
        if self.action_noise > 0:
            noise = np.random.normal(0, self.action_noise, action.shape)
            noisy_action = action + noise
            noisy_action = np.clip(noisy_action, self.action_space.low, self.action_space.high)
        else:
            noisy_action = action
        
        # Take multiple substeps for lower control frequency
        cumulative_reward = 0
        for _ in range(self.control_frequency):
            observation, reward, terminated, truncated, info = self.base_env.step(noisy_action)
            cumulative_reward += reward
            
            if terminated or truncated:
                break
        
        # Apply observation noise
        if self.observation_noise > 0:
            noise = np.random.normal(0, self.observation_noise, observation.shape)
            observation += noise
        
        # Modify reward based on custom components
        modified_reward = self._compute_reward(cumulative_reward, observation, info)
        
        return observation, modified_reward, terminated, truncated, info
    
    def _compute_reward(self, base_reward, observation, info):
        """Compute custom reward components."""
        # Extract velocity and position information
        if hasattr(info, 'x_velocity'):
            x_velocity = info['x_velocity']
        else:
            # Extract from observation if not in info
            x_velocity = observation[0]  # This depends on observation structure
        
        # Base forward reward
        forward_reward = self.forward_reward_weight * x_velocity
        
        # Energy penalty (control cost)
        control_cost = self.energy_penalty_coef * np.sum(np.square(info.get('action', 0)))
        
        # Goal reward if applicable
        goal_reward = 0
        if self.goal_position is not None:
            # Extract position from observation or info
            position = observation[0:2]  # Assuming first 2 elements are x,y position
            dist_to_goal = np.linalg.norm(position - self.goal_position)
            goal_reward = -0.1 * dist_to_goal
        
        return forward_reward - control_cost + goal_reward
    
def get_state_representation(self):
    """Get the state representation suitable for transfer learning.
    
    Extracts and normalizes key features from the observation space that
    are most relevant for transfer learning between different Ant variants.
    
    Returns:
        numpy.ndarray: Normalized state representation with key features
    """
    # Get the raw observation from the environment
    raw_obs = self.base_env._get_obs()
    
    # Extract the key components based on the standard Ant observation structure
    # Typically, the Ant observation space includes:
    # - Position [0:3]
    # - Orientation (quaternion) [3:7]
    # - Joint positions [7:15]
    # - Joint velocities [15:23] 
    # - External forces (optional)
    
    # Check if we have a proper observation
    if raw_obs is None or len(raw_obs) < 15:
        # Fallback to a minimal representation if observation is incomplete
        return np.zeros(10)
    
    # Extract position (x, y, z)
    position = raw_obs[0:3]
    
    # Extract orientation (quaternion)
    orientation = raw_obs[3:7]
    
    # Extract joint positions and velocities
    # Focus on the most important joints for locomotion
    joint_positions = raw_obs[7:15]
    joint_velocities = raw_obs[15:23] if len(raw_obs) >= 23 else np.zeros(8)
    
    # Normalize position
    # Use a reasonable range for the Ant's movement (typically within [-10, 10])
    normalized_position = position / 10.0
    
    # Quaternions are already normalized by definition
    normalized_orientation = orientation
    
    # Normalize joint positions and velocities
    # Joint positions in the Ant are typically in the range [-1, 1]
    normalized_joint_positions = joint_positions / 1.0
    
    # Joint velocities can vary more, typically in range [-10, 10]
    normalized_joint_velocities = joint_velocities / 10.0
    
    # Calculate additional features that may help with transfer
    forward_velocity = np.array([raw_obs[13]])  # Typically the x-velocity component
    normalized_forward_velocity = forward_velocity / 5.0  # Normalize to reasonable range
    
    # Compute body height (z-position) as a separate feature
    # This is often critical for stable locomotion
    body_height = np.array([position[2]])
    normalized_body_height = body_height / 1.0  # Typical height range
    
    # Combine all normalized features
    # Focus on the most transferable aspects, which are typically:
    # - Normalized x-y position (ignore z for planar movement)
    # - Orientation (quaternion)
    # - Joint positions
    # - Forward velocity
    # - Body height
    transferable_state = np.concatenate([
        normalized_position[0:2],  # x-y position (planar movement)
        normalized_orientation,    # orientation
        normalized_joint_positions,  # joint positions
        normalized_forward_velocity,  # forward velocity
        normalized_body_height     # body height
    ])
    
    return transferable_state
    
    def increase_complexity(self, increment=0.1):
        """Increase the environment complexity."""
        current = self.complexity_level
        target = min(1.0, current + increment)
        
        # Update complexity parameters based on new level
        if current < 0.3 and target >= 0.3:
            # Add observation noise
            self.observation_noise = 0.01
        
        if current < 0.5 and target >= 0.5:
            # Add action noise
            self.action_noise = 0.01
        
        if current < 0.7 and target >= 0.7:
            # Decrease control frequency (making control harder)
            self.control_frequency = 3
        
        if current < 0.9 and target >= 0.9:
            # Make physics more challenging
            self.friction = 0.8
            self.modify_env_parameters()
        
        # Update complexity level in config
        self.config["complexity_level"] = target
        
    def render(self):
        """Render the environment."""
        return self.base_env.render()