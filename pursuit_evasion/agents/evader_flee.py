"""
DEMO Evader — SIMPLE FLEE.
NO lidar, NO obstacle avoidance, NO stuck detection.
Just drive away from nearest pursuer (slowly).
"""
import numpy as np
from agents.base_agent import BaseAgent


class FleeingEvader(BaseAgent):
    """Simple flee: drive away from nearest pursuer."""
    
    def __init__(self, name="evader", lidar_range=5.0,
                 max_linear_vel=0.5,
                 max_angular_vel=1.5,
                 flee_speed=0.35,
                 kp_heading=2.5,
                 panic_dist=2.5):
        super().__init__(name, lidar_range, max_linear_vel, max_angular_vel)
        self.flee_speed = flee_speed
        self.kp_heading = kp_heading
        self.panic_dist = panic_dist
    
    def policy(self, obs, agent_idx):
        """Simple flee: drive away from nearest pursuer."""
        if not obs["evader_alive"][agent_idx]:
            return (0.0, 0.0)
        
        my_pose = obs["evader_poses"][agent_idx]
        my_pos = my_pose[:2]
        my_yaw = my_pose[2]
        
        # Find nearest pursuer
        pursuer_poses = obs["pursuer_poses"]
        pursuer_positions = pursuer_poses[:, :2]
        dists = np.linalg.norm(pursuer_positions - my_pos, axis=1)
        nearest_idx = np.argmin(dists)
        nearest_dist = dists[nearest_idx]
        nearest_pur_pos = pursuer_positions[nearest_idx]
        
        # Flee direction (AWAY from pursuer)
        dx = my_pos[0] - nearest_pur_pos[0]
        dy = my_pos[1] - nearest_pur_pos[1]
        flee_angle_world = np.arctan2(dy, dx)
        angle_to_flee = self.angle_diff(flee_angle_world, my_yaw)
        
        # Simple P control
        omega = self.kp_heading * angle_to_flee
        omega = np.clip(omega, -self.max_angular_vel, self.max_angular_vel)
        
        # Speed: panic = max, else normal
        if nearest_dist < self.panic_dist:
            v = self.max_linear_vel
        else:
            v = self.flee_speed
        
        return self.clamp_action(v, omega)