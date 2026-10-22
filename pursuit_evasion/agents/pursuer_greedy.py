"""
DEMO Pursuer — PURE DIRECT RAM VERSION.
NO lidar, NO obstacle avoidance, NO stuck detection.
Just drive straight to nearest evader and ram it.
"""
import numpy as np
from agents.base_agent import BaseAgent


class GreedyPursuer(BaseAgent):
    """Super simple: drive straight to nearest evader, ram it."""
    
    def __init__(self, name="pursuer", lidar_range=5.0,
                 max_linear_vel=1.5,
                 max_angular_vel=2.5,
                 chase_speed=1.2,
                 kp_heading=4.0,
                 target_idx=0):
        super().__init__(name, lidar_range, max_linear_vel, max_angular_vel)
        self.chase_speed = chase_speed
        self.kp_heading = kp_heading
        self.target_idx = target_idx
    
    def policy(self, obs, agent_idx):
        """Direct chase: drive straight to nearest evader."""
        my_pose = obs["pursuer_poses"][agent_idx]
        my_pos = my_pose[:2]
        my_yaw = my_pose[2]
        
        # Find nearest alive evader
        alive_mask = obs["evader_alive"]
        if not alive_mask.any():
            return (0.0, 0.0)
        
        alive_indices = np.where(alive_mask)[0]
        evader_positions = obs["evader_poses"][alive_indices][:, :2]
        dists = np.linalg.norm(evader_positions - my_pos, axis=1)
        nearest_local_idx = np.argmin(dists)
        self.target_idx = alive_indices[nearest_local_idx]
        target_pos = evader_positions[nearest_local_idx]
        nearest_dist = dists[nearest_local_idx]
        
        # Angle to target
        dx = target_pos[0] - my_pos[0]
        dy = target_pos[1] - my_pos[1]
        target_angle_world = np.arctan2(dy, dx)
        angle_to_target = self.angle_diff(target_angle_world, my_yaw)
        
        # Simple P control on heading
        omega = self.kp_heading * angle_to_target
        omega = np.clip(omega, -self.max_angular_vel, self.max_angular_vel)
        
        # Full speed ahead
        v = self.chase_speed
        
        # Slow down only when very close (avoid orbit)
        if nearest_dist < 0.4:
            v *= 0.5
        
        return self.clamp_action(v, omega)
    
    @classmethod
    def reset_shared_state(cls):
        pass