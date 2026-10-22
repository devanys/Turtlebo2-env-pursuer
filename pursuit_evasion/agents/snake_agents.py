import numpy as np
from agents.base_agent import BaseAgent


class SnakeHead(BaseAgent):
    """Head: free roam + chase when red circle meets evader's yellow magnet circle."""
    
    def __init__(self, name="head", lidar_range=5.0,
                 max_linear_vel=1.5, max_angular_vel=2.5,
                 chase_speed=1.2, free_roam_speed=1.0,
                 kp_heading=3.0, detection_radius=1.5,
                 arena_size=12.0, approach_distance=1.0):
        super().__init__(name, lidar_range, max_linear_vel, max_angular_vel)
        self.chase_speed = chase_speed
        self.free_roam_speed = free_roam_speed
        self.kp_heading = kp_heading
        self.detection_radius = detection_radius
        self.arena_half = arena_size / 2
        self.approach_distance = approach_distance
        self.wander_heading = np.random.uniform(0, 2 * np.pi)
        self.last_pos = None
        self.stuck_counter = 0
        self.cooldown = 0
        self.realtime = True
    
    def _compute_repulsion(self, my_pos, other_positions, repulsion_radius=1.5, max_force=1.0):
        if len(other_positions) == 0:
            return np.array([0.0, 0.0])
        diffs = my_pos - other_positions
        dists = np.linalg.norm(diffs, axis=1)
        close_mask = dists < repulsion_radius
        if not close_mask.any():
            return np.array([0.0, 0.0])
        close_diffs = diffs[close_mask]
        close_dists = np.maximum(dists[close_mask], 0.1)
        weights = (repulsion_radius - close_dists) / repulsion_radius
        repulsion_vec = np.sum(
            (close_diffs / close_dists[:, None]) * weights[:, None], axis=0
        )
        mag = np.linalg.norm(repulsion_vec)
        if mag > 0:
            repulsion_vec = repulsion_vec / mag * min(mag, max_force)
        return repulsion_vec
    
    def policy(self, obs, agent_idx):
        my_pose = obs["head_pose"]
        my_pos = my_pose[:2]
        my_yaw = my_pose[2]
        
        if self.last_pos is None:
            self.last_pos = my_pos.copy()
        else:
            moved = np.linalg.norm(my_pos - self.last_pos)
            if moved < 0.005:
                self.stuck_counter += 1
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)
            self.last_pos = my_pos.copy()
        
        if self.stuck_counter > 15:
            self.stuck_counter = 0
            self.wander_heading += np.random.uniform(-np.pi/2, np.pi/2)
            self.wander_heading = self.normalize_angle(self.wander_heading)
        
        in_cooldown = self.cooldown > 0
        if in_cooldown:
            self.cooldown -= 1
        
        alive_indices = np.where(obs["evader_alive"])[0]
        if len(alive_indices) == 0:
            return (0.0, 0.0)
        
        evader_positions = obs["evader_poses"][alive_indices]
        dists = np.linalg.norm(evader_positions[:, :2] - my_pos, axis=1)
        nearest_local_idx = np.argmin(dists)
        nearest_dist = dists[nearest_local_idx]
        nearest_evader_pos = evader_positions[nearest_local_idx]
        nearest_evader_yaw = nearest_evader_pos[2]
        
        target_pos = nearest_evader_pos[:2]
        target_dist = nearest_dist
        
        if not in_cooldown and "follower_alerts" in obs and obs["follower_alerts"]:
            alert_positions = np.array(obs["follower_alerts"])
            alert_dists = np.linalg.norm(alert_positions - my_pos, axis=1)
            nearest_alert_idx = np.argmin(alert_dists)
            nearest_alert_dist = alert_dists[nearest_alert_idx]
            if nearest_alert_dist < target_dist:
                target_pos = alert_positions[nearest_alert_idx]
                target_dist = nearest_alert_dist
        
        # === CIRCLE-MEET-CIRCLE TRIGGER ===
        # Chase triggers when red circle (head) meets yellow circle (evader magnet)
        # Distance = detection_radius (head) + detection_radius (evader magnet) = 2 * detection_radius
        chase_trigger_dist = self.detection_radius * 2
        
        if not in_cooldown and target_dist < chase_trigger_dist:
            target_x = nearest_evader_pos[0] - self.approach_distance * np.cos(nearest_evader_yaw)
            target_y = nearest_evader_pos[1] - self.approach_distance * np.sin(nearest_evader_yaw)
            arena_half = self.arena_half - 0.5
            target_x = np.clip(target_x, -arena_half, arena_half)
            target_y = np.clip(target_y, -arena_half, arena_half)
            
            dx = target_x - my_pos[0]
            dy = target_y - my_pos[1]
            target_angle = np.arctan2(dy, dx)
            angle_to_target = self.angle_diff(target_angle, my_yaw)
            
            omega = np.clip(self.kp_heading * angle_to_target,
                          -self.max_angular_vel, self.max_angular_vel)
            
            if abs(angle_to_target) > 1.0:
                v = self.chase_speed * 0.5
            else:
                v = self.chase_speed
            
            return self.clamp_action(v, omega)
        else:
            self.wander_heading += np.random.normal(0, 0.1)
            self.wander_heading = self.normalize_angle(self.wander_heading)
            desired_angle = self._wall_avoidance(my_pos, self.wander_heading)
            
            other_evader_positions = []
            for j in alive_indices:
                if j != nearest_local_idx:
                    other_evader_positions.append(obs["evader_poses"][j][:2])
            other_evader_positions = np.array(other_evader_positions) if other_evader_positions else np.empty((0, 2))
            repulsion = self._compute_repulsion(my_pos, other_evader_positions,
                                                repulsion_radius=1.5, max_force=1.0)
            
            if np.linalg.norm(repulsion) > 0.01:
                repulsion_angle = np.arctan2(repulsion[1], repulsion[0])
                repulsion_strength = min(np.linalg.norm(repulsion), 0.8)
                wander_x = (1 - repulsion_strength) * np.cos(desired_angle)
                wander_y = (1 - repulsion_strength) * np.sin(desired_angle)
                rep_x = repulsion_strength * np.cos(repulsion_angle)
                rep_y = repulsion_strength * np.sin(repulsion_angle)
                desired_angle = np.arctan2(wander_y + rep_y, wander_x + rep_x)
                self.wander_heading = desired_angle
            
            angle_to_target = self.angle_diff(desired_angle, my_yaw)
            omega = np.clip(self.kp_heading * angle_to_target,
                          -self.max_angular_vel, self.max_angular_vel)
            v = self.free_roam_speed
            
            return self.clamp_action(v, omega)
    
    def on_capture(self, new_head_yaw):
        self.wander_heading = self.normalize_angle(new_head_yaw)
        self.last_pos = None
        self.stuck_counter = 0
        self.cooldown = 30
    
    def _wall_avoidance(self, my_pos, wander_heading):
        margin = 1.5
        dist_to_wall_x = self.arena_half - abs(my_pos[0])
        dist_to_wall_y = self.arena_half - abs(my_pos[1])
        desired_angle = wander_heading
        if dist_to_wall_x < margin or dist_to_wall_y < margin:
            center_angle = np.arctan2(-my_pos[1], -my_pos[0])
            blend = 1.0 - min(dist_to_wall_x, dist_to_wall_y) / margin
            blend = min(blend, 0.8)
            desired_x = (1 - blend) * np.cos(wander_heading) + blend * np.cos(center_angle)
            desired_y = (1 - blend) * np.sin(wander_heading) + blend * np.sin(center_angle)
            desired_angle = np.arctan2(desired_y, desired_x)
            self.wander_heading = desired_angle
        return desired_angle


class SnakeEvader(BaseAgent):
    """Evader: stationary (no movement)."""
    
    def __init__(self, name="evader", lidar_range=5.0,
                 max_linear_vel=0.5, max_angular_vel=1.5,
                 flee_speed=0.4, kp_heading=2.5,
                 arena_size=12.0):
        super().__init__(name, lidar_range, max_linear_vel, max_angular_vel)
        self.flee_speed = flee_speed
        self.kp_heading = kp_heading
        self.arena_half = arena_size / 2
        self.wander_heading = np.random.uniform(0, 2 * np.pi)
    
    def policy(self, obs, agent_idx):
        if not obs["evader_alive"][agent_idx]:
            return (0.0, 0.0)
        return (0.0, 0.0)