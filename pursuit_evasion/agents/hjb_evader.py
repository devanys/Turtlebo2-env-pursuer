"""HJB-based Evader with stuck recovery."""
import numpy as np
from agents.base_agent import BaseAgent


class HJBEvader(BaseAgent):
    """Evader using HJB reachability for optimal escape strategy."""
    
    def __init__(self, name="evader", lidar_range=5.0,
                 max_linear_vel=0.5, max_angular_vel=1.5,
                 flee_speed=0.35, kp_heading=2.5,
                 hjb_solver=None, panic_dist=2.5):
        super().__init__(name, lidar_range, max_linear_vel, max_angular_vel)
        self.flee_speed = flee_speed
        self.kp_heading = kp_heading
        self.hjb_solver = hjb_solver
        self.panic_dist = panic_dist
        self.current_value = 0.0
        self.current_ttc = 0.0
        
        # Stuck detection
        self.stuck_counter = 0
        self.last_pos = None
    
    def policy(self, obs, agent_idx):
        """HJB-guided escape with direct flee override when close."""
        if not obs["evader_alive"][agent_idx]:
            return (0.0, 0.0)
        
        my_pose = obs["evader_poses"][agent_idx]
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
        
        # Find nearest pursuer
        pursuer_positions = obs["pursuer_poses"][:, :2]
        dists = np.linalg.norm(pursuer_positions - my_pos, axis=1)
        nearest_idx = np.argmin(dists)
        nearest_dist = dists[nearest_idx]
        nearest_pur_pos = pursuer_positions[nearest_idx]
        
        # rel = evader - pursuer (direction AWAY from pursuer in world frame)
        rel_x = my_pos[0] - nearest_pur_pos[0]
        rel_y = my_pos[1] - nearest_pur_pos[1]
        
        # === DIRECT FLEE OVERRIDE when very close (< 1.5m) ===
        if nearest_dist < 1.5:
            # Just flee directly away
            desired_angle_world = np.arctan2(rel_y, rel_x)
            if self.hjb_solver is not None:
                self.current_value = self.hjb_solver.query_value(rel_x, rel_y)
                self.current_ttc = self.hjb_solver.predict_capture_time(rel_x, rel_y)
        elif self.hjb_solver is not None:
            # Use HJB gradient for strategic escape
            V = self.hjb_solver.query_value(rel_x, rel_y)
            grad = self.hjb_solver.query_gradient(rel_x, rel_y)
            self.current_value = V
            self.current_ttc = self.hjb_solver.predict_capture_time(rel_x, rel_y)
            
            grad_mag = np.linalg.norm(grad)
            if grad_mag > 1e-6:
                # For evader: gradient points TOWARD pursuer (in world frame from evader)
                # Wait — rel = evader - pursuer, gradient at rel points AWAY from origin in rel space
                # = AWAY from pursuer in world frame from evader's perspective ✓
                desired_angle_world = np.arctan2(grad[1], grad[0])
            else:
                desired_angle_world = np.arctan2(rel_y, rel_x)
        else:
            desired_angle_world = np.arctan2(rel_y, rel_x)
            self.current_ttc = nearest_dist / max(self.flee_speed, 0.1)
        
        # === HEADING CONTROL ===
        angle_to_flee = self.angle_diff(desired_angle_world, my_yaw)
        omega = np.clip(self.kp_heading * angle_to_flee,
                       -self.max_angular_vel, self.max_angular_vel)
        
        # === SPEED: panic if close OR inside BRT ===
        in_brt = (self.hjb_solver is not None and self.current_value < 0)
        if nearest_dist < self.panic_dist or in_brt:
            speed = self.max_linear_vel
        else:
            speed = self.flee_speed
        
        if abs(angle_to_flee) > 1.5:
            v = speed * 0.3
        elif abs(angle_to_flee) > 0.5:
            v = speed * 0.7
        else:
            v = speed
        
        return self.clamp_action(v, omega)