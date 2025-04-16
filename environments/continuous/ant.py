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
    
    def create_state_mapping(self, source_agent, target_agent):
        """Create mapping between source and target state spaces."""
        # Extract dimensions from both observation spaces
        source_dims = source_agent.obs_dim
        target_dims = target_agent.obs_dim
        
        # Initialize mapping dictionary
        mapping = {}
        
        # For Ant environments specifically, we need to map the observation
        # components based on their semantic meaning
        
        # Define semantic segments of the observation vector
        segments = {
            'position': {'source_range': (0, min(3, source_dims)), 
                        'target_range': (0, min(3, target_dims))},
                        
            'orientation': {'source_range': (3, min(7, source_dims)), 
                        'target_range': (3, min(7, target_dims))},
                        
            'joint_positions': {'source_range': (7, min(15, source_dims)), 
                            'target_range': (7, min(15, target_dims))},
                            
            'velocities': {'source_range': (15, min(23, source_dims)), 
                        'target_range': (15, min(23, target_dims))}
        }
        
        # Create mapping for each segment
        for segment, ranges in segments.items():
            s_start, s_end = ranges['source_range']
            t_start, t_end = ranges['target_range']
            
            # Skip if source doesn't have this segment
            if s_start >= source_dims:
                continue
                
            # Map available dimensions
            shared_length = min(s_end - s_start, t_end - t_start)
            for i in range(shared_length):
                mapping[s_start + i] = t_start + i
        
        return mapping
        
    def get_state_representation(self):
        """Get state representation suitable for transfer learning."""
        # Get the current observation
        if hasattr(self.base_env, 'unwrapped') and hasattr(self.base_env.unwrapped, '_get_obs'):
            raw_obs = self.base_env.unwrapped._get_obs()
        else:
            raw_obs = self.base_env._get_obs()
        
        if raw_obs is None or len(raw_obs) < 15:
            return np.zeros(10)  # Fallback for invalid observations
        
        # Extract components based on standard Ant observation structure
        position = raw_obs[0:3]                         # x, y, z position
        orientation = raw_obs[3:7]                      # quaternion
        joint_positions = raw_obs[7:15]                 # 8 joint angles
        joint_velocities = raw_obs[15:23] if len(raw_obs) >= 23 else np.zeros(8)
        
        # Normalize components to common ranges
        normalized_position = position / 10.0           # Typical Ant movement range
        normalized_orientation = orientation            # Already normalized by definition
        normalized_joint_positions = joint_positions / 1.0  # Joint angle range
        normalized_joint_velocities = joint_velocities / 10.0  # Typical velocity range
        
        # Calculate additional features helpful for transfer
        body_height = np.array([position[2]])  # z-coordinate (height)
        normalized_body_height = body_height / 1.0
        
        # Forward velocity, important for locomotion tasks
        forward_velocity = np.array([raw_obs[13]]) if len(raw_obs) > 13 else np.array([0.0])
        normalized_forward_velocity = forward_velocity / 5.0
        
        # Combine normalized features into transferable state
        transferable_state = np.concatenate([
            normalized_position,                # 3D position 
            normalized_orientation,             # 4D orientation
            normalized_joint_positions,         # 8D joint positions
            normalized_joint_velocities,        # 8D joint velocities
            normalized_body_height,             # 1D body height
            normalized_forward_velocity         # 1D forward velocity
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