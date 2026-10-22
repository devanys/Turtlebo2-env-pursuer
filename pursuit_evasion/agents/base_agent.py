"""
Base class for all agents (pursuers & evaders) — SIMPLIFIED.
"""
import numpy as np


class BaseAgent:
    """Abstract base with minimal utilities."""
    
    def __init__(self, name="agent", lidar_range=5.0, 
                 max_linear_vel=0.65, max_angular_vel=1.5):
        self.name = name
        self.lidar_range = lidar_range
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
    
    def policy(self, obs, agent_idx):
        raise NotImplementedError("Subclass must implement policy()")
    
    @staticmethod
    def angle_diff(target, current):
        """Wrapped angle difference to [-pi, pi]."""
        diff = target - current
        return np.arctan2(np.sin(diff), np.cos(diff))
    
    @staticmethod
    def normalize_angle(angle):
        """Wrap angle to [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def clamp_action(self, v, omega):
        """Clamp velocities to robot limits."""
        v = float(np.clip(v, -self.max_linear_vel, self.max_linear_vel))
        omega = float(np.clip(omega, -self.max_angular_vel, self.max_angular_vel))
        return v, omega