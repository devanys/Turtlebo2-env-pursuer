"""HJB-based Pursuer with stuck recovery and direct chase override."""
import numpy as np
from agents.base_agent import BaseAgent


class HJBPursuer(BaseAgent):
    """Pursuer using HJB reachability for optimal capture strategy."""
    
    def __init__(self, name="pursuer", lidar_range=5.0,
                 max_linear_vel=1.5, max_angular_vel=2.5,
                 chase_speed=1.2, kp_heading=4.0,
                 hjb_solver=None, target_idx=0):
        super().__init__(name, lidar_range, max_linear_vel, max_angular_vel)
        self.chase_speed = chase_speed
        self.kp_heading = kp_heading
        self.hjb_solver = hjb_solver
        self.target_idx = target_idx
        self.current_value = 0.0
        self.current_ttc = 0.0
        
        # Stuck detection
        self.stuck_counter = 0
        self.last_pos = None
    
    def policy(self, obs, agent_idx):
        """HJB-guided pursuit with direct chase override when close."""
        my_pose = obs["pursuer_poses"][agent_idx]
        my_pos = my_pose[:2]
        my_yaw = my_pose[2]
        
        # === STUCK DETECTION ===
        if self.last_pos is None:
            self.last_pos = my_pos.copy()
        else:
            moved = np.linalg.norm(my_pos - self.last_pos)
            if moved < 0.005:
                self.stuck_counter += 1
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)
            self.last_pos = my_pos.copy()
        
        # Hard recovery if stuck > 10 steps
        if self.stuck_counter > 10:
            self.stuck_counter = 0
            return self.clamp_action(-0.3, 1.8)
        
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
        
        rel_x = target_pos[0] - my_pos[0]
        rel_y = target_pos[1] - my_pos[1]
        
        # === DIRECT CHASE OVERRIDE when very close (< 1.5m) ===
        # HJB gradient may be unreliable near capture boundary
        if nearest_dist < 1.5:
            desired_angle_world = np.arctan2(rel_y, rel_x)
            if self.hjb_solver is not None:
                self.current_value = self.hjb_solver.query_value(rel_x, rel_y)
                self.current_ttc = self.hjb_solver.predict_capture_time(rel_x, rel_y)
        elif self.hjb_solver is not None:
            # Use HJB gradient for strategic positioning
            V = self.hjb_solver.query_value(rel_x, rel_y)
            grad = self.hjb_solver.query_gradient(rel_x, rel_y)
            self.current_value = V
            self.current_ttc = self.hjb_solver.predict_capture_time(rel_x, rel_y)
            
            grad_mag = np.linalg.norm(grad)
            if grad_mag > 1e-6:
                # Gradient points TOWARD evader (since rel points from pursuer to evader)
                desired_angle_world = np.arctan2(grad[1], grad[0])
            else:
                desired_angle_world = np.arctan2(rel_y, rel_x)
        else:
            desired_angle_world = np.arctan2(rel_y, rel_x)
            self.current_ttc = nearest_dist / max(self.chase_speed, 0.1)
        
        # === HEADING CONTROL ===
        angle_to_target = self.angle_diff(desired_angle_world, my_yaw)
        omega = np.clip(self.kp_heading * angle_to_target,
                       -self.max_angular_vel, self.max_angular_vel)
        
        # === SPEED CONTROL ===
        if abs(angle_to_target) > 1.5:
            v = self.chase_speed * 0.4
        elif abs(angle_to_target) > 0.5:
            v = self.chase_speed * 0.8
        else:
            v = self.chase_speed
        
        # Don't slow down when close — RAM the evader!
        if nearest_dist < 0.8:
            v = self.chase_speed  # full speed to ram
        
        return self.clamp_action(v, omega)
    
    @classmethod
    def reset_shared_state(cls):
        pass