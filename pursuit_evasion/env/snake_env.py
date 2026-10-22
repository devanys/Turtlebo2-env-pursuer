import numpy as np
import pybullet as p
import pybullet_data
from collections import deque

from env.arena import Arena
from env.turtlebot2 import TurtleBot2

HEAD_COLOR = [0.05, 0.28, 0.63, 1.0]
FOLLOWER_COLOR = [0.13, 0.58, 0.84, 1.0]
EVADER_COLOR = [0.72, 0.11, 0.11, 1.0]


class SnakeEnv:
    def __init__(self, config, mode="gui"):
        self.config = config
        self.mode = mode
        
        self.sim_cfg = config["simulation"]
        self.arena_cfg = config["arena"]
        self.robots_cfg = config["robots"]
        self.robot_params = config["robot_params"]
        self.lidar_params = config["lidar"]
        self.game_cfg = config["game"]
        self.snake_cfg = config["snake"]
        
        self.num_evaders = int(self.robots_cfg["num_evaders"])
        self.capture_radius = float(self.game_cfg["capture_radius"])
        self.detection_radius = float(self.game_cfg["detection_radius"])
        self.formation_spacing = float(self.snake_cfg["formation_spacing"])
        self.max_episode_time = float(self.sim_cfg["max_episode_time"])
        self.timestep = float(self.sim_cfg["timestep"])
        self.control_freq = int(self.sim_cfg["control_freq"])
        self.steps_per_control = int(round(1.0 / (self.timestep * self.control_freq)))
        
        self.client = None
        self.arena = None
        self.pursuers = []
        self.evaders = []
        self.evader_alive = np.zeros(self.num_evaders, dtype=bool)
        self.head_path = deque(maxlen=10000)
        self.head_trail_viz = deque(maxlen=500)
        self.time = 0.0
        self.step_count = 0
        self.episode_count = 0
        self.capture_events = []
        
        self._init_physics()
    
    def _init_physics(self):
        if self.mode == "gui":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.timestep, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        for bot in self.pursuers + self.evaders:
            try:
                p.removeBody(bot.robot_id, physicsClientId=self.client)
            except:
                pass
        self.pursuers = []
        self.evaders = []
        
        if self.arena is not None:
            self.arena.clear()
        self.arena = Arena(self.client, self.config)
        self.arena.build(seed=seed)
        
        self.evader_alive = np.zeros(self.num_evaders, dtype=bool)
        self.head_path.clear()
        self.head_trail_viz.clear()
        self.time = 0.0
        self.step_count = 0
        self.capture_events = []
        
        head_pos = (0.0, 0.0)
        head_yaw = np.random.uniform(0, 2 * np.pi)
        
        pursuer_params = dict(self.robot_params)
        pursuer_params["max_linear_vel"] = 1.5
        pursuer_params["max_angular_vel"] = 2.5
        pursuer_params["wheel_force"] = 20.0
        
        head_bot = TurtleBot2(
            client=self.client,
            start_pos=[head_pos[0], head_pos[1], 0.05],
            start_yaw=head_yaw,
            name="head",
            urdf_path=self.robots_cfg["urdf_path"],
            robot_params=pursuer_params,
            lidar_params=self.lidar_params,
            verbose=True,
            color=HEAD_COLOR,
        )
        self.pursuers.append(head_bot)
        
        self.head_path.append((head_pos[0], head_pos[1], head_yaw))
        self.head_trail_viz.append((head_pos[0], head_pos[1]))
        
        arena_half = self.arena.size[0] / 2
        margin = 1.5
        existing_positions = [np.array(head_pos)]
        
        evader_params = dict(self.robot_params)
        evader_params["max_linear_vel"] = 0.5
        evader_params["max_angular_vel"] = 1.5
        evader_params["wheel_force"] = 10.0
        
        for i in range(self.num_evaders):
            pos = None
            for _ in range(200):
                x = np.random.uniform(-arena_half + margin, arena_half - margin)
                y = np.random.uniform(-arena_half + margin, arena_half - margin)
                ok = True
                for ex_pos in existing_positions:
                    if np.linalg.norm(np.array([x, y]) - ex_pos) < 1.5:
                        ok = False
                        break
                if ok:
                    pos = (x, y)
                    break
            if pos is None:
                pos = (np.random.uniform(-arena_half + margin, arena_half - margin),
                       np.random.uniform(-arena_half + margin, arena_half - margin))
            
            yaw = np.random.uniform(0, 2 * np.pi)
            existing_positions.append(np.array(pos))
            
            evader_bot = TurtleBot2(
                client=self.client,
                start_pos=[pos[0], pos[1], 0.05],
                start_yaw=yaw,
                name=f"evader_{i}",
                urdf_path=self.robots_cfg["urdf_path"],
                robot_params=evader_params,
                lidar_params=self.lidar_params,
                verbose=False,
                color=EVADER_COLOR,
            )
            self.evaders.append(evader_bot)
        
        self.evader_alive[:] = True
        
        self.episode_count += 1
        print(f"\n[SNAKE] Episode {self.episode_count} reset: 1 head, {self.num_evaders} evaders")
        
        return self._get_observation()
    
    def step(self, actions):
        head_v, head_omega = actions["head"]
        self.pursuers[0].set_velocity(head_v, head_omega)
        
        for i, (v, omega) in enumerate(actions["evaders"]):
            if not self.evader_alive[i]:
                continue
            self.evaders[i].set_velocity(v, omega)
        
        for _ in range(self.steps_per_control):
            if not p.isConnected(self.client):
                self.time += self.timestep
                continue
            try:
                p.stepSimulation(physicsClientId=self.client)
            except p.error:
                continue
            self.time += self.timestep
            
            head_pose = self.pursuers[0].get_pose()
            self.head_path.append((head_pose[0], head_pose[1], head_pose[2]))
        
        self.step_count += 1
        
        head_pose = self.pursuers[0].get_pose()
        self.head_trail_viz.append((head_pose[0], head_pose[1]))
        
        self._update_followers()
        
        new_captures = self._check_captures()
        for cap_pur_idx, cap_ev_idx in new_captures:
            self._handle_capture(cap_pur_idx, cap_ev_idx)
            self.capture_events.append((self.step_count, cap_pur_idx, cap_ev_idx))
        
        done = bool(
            (not self.evader_alive.any())
            or (self.time >= self.max_episode_time)
        )
        
        info = {
            "step": self.step_count,
            "time": self.time,
            "evader_alive": self.evader_alive.copy(),
            "num_caught": int((~self.evader_alive).sum()),
            "num_remaining": int(self.evader_alive.sum()),
            "snake_length": len(self.pursuers),
            "capture_events": self.capture_events.copy(),
            "max_time": self.max_episode_time,
        }
        
        rewards = {"head": 1.0 if new_captures else 0.0, "evaders": np.zeros(self.num_evaders)}
        
        return self._get_observation(), rewards, done, info
    
    def _get_follower_target(self, follower_index):
        if len(self.head_path) < 2:
            return None
        
        target_distance = follower_index * self.formation_spacing
        
        path = list(self.head_path)
        accumulated = 0.0
        
        for i in range(len(path) - 2, -1, -1):
            x1, y1, yaw1 = path[i + 1]
            x2, y2, yaw2 = path[i]
            seg_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if accumulated + seg_len >= target_distance:
                remaining = target_distance - accumulated
                if seg_len > 1e-6:
                    t = remaining / seg_len
                else:
                    t = 0.0
                tx = x1 + t * (x2 - x1)
                ty = y1 + t * (y2 - y1)
                tyaw = yaw2 if t > 0.5 else yaw1
                return (tx, ty, tyaw)
            
            accumulated += seg_len
        
        return path[0]
    
    def _update_followers(self):
        """Update followers with STRONG anti-clumping + path-following."""
        if len(self.pursuers) < 2:
            return
        
        all_positions = np.array([bot.get_pose()[:2] for bot in self.pursuers])
        
        for i in range(1, len(self.pursuers)):
            target = self._get_follower_target(i)
            if target is None:
                continue
            
            target_x, target_y, target_yaw = target
            follower = self.pursuers[i]
            follower_pose = follower.get_pose()
            fx, fy, fyaw = follower_pose
            my_pos = np.array([fx, fy])
            
            other_positions = np.delete(all_positions, i, axis=0)
            repulsion = self._compute_follower_repulsion(
                my_pos, other_positions, repulsion_radius=0.7, max_force=2.5
            )
            
            repulsion_mag = np.linalg.norm(repulsion)
            
            if repulsion_mag > 0.01:
                repulsion_angle = np.arctan2(repulsion[1], repulsion[0])
                repulsion_strength = min(repulsion_mag / 2.5, 0.8)
                path_angle = np.arctan2(target_y - fy, target_x - fx)
                blended_x = ((1 - repulsion_strength) * np.cos(path_angle) 
                             + repulsion_strength * np.cos(repulsion_angle))
                blended_y = ((1 - repulsion_strength) * np.sin(path_angle) 
                             + repulsion_strength * np.sin(repulsion_angle))
                blended_angle = np.arctan2(blended_y, blended_x)
                target_x = fx + 1.5 * np.cos(blended_angle)
                target_y = fy + 1.5 * np.sin(blended_angle)
            
            dx = target_x - fx
            dy = target_y - fy
            dist = np.sqrt(dx * dx + dy * dy)
            
            if dist > 0.05:
                target_angle = np.arctan2(dy, dx)
                angle_diff_val = target_angle - fyaw
                angle_diff_val = np.arctan2(np.sin(angle_diff_val), np.cos(angle_diff_val))
                
                omega = np.clip(4.0 * angle_diff_val, -2.5, 2.5)
                v = min(1.5, dist * 3.0)
            else:
                angle_diff_val = target_yaw - fyaw
                angle_diff_val = np.arctan2(np.sin(angle_diff_val), np.cos(angle_diff_val))
                omega = np.clip(3.0 * angle_diff_val, -2.5, 2.5)
                v = 0.0
            
            follower.set_velocity(v, omega)
    
    def _compute_follower_repulsion(self, my_pos, other_positions, 
                                     repulsion_radius=0.7, max_force=2.5):
        """STRONG repulsion to prevent clumping (quadratic weighting)."""
        if len(other_positions) == 0:
            return np.array([0.0, 0.0])
        diffs = my_pos - other_positions
        dists = np.linalg.norm(diffs, axis=1)
        close_mask = dists < repulsion_radius
        if not close_mask.any():
            return np.array([0.0, 0.0])
        close_diffs = diffs[close_mask]
        close_dists = np.maximum(dists[close_mask], 0.1)
        weights = ((repulsion_radius - close_dists) / repulsion_radius) ** 2
        repulsion_vec = np.sum(
            (close_diffs / close_dists[:, None]) * weights[:, None], axis=0
        )
        mag = np.linalg.norm(repulsion_vec)
        if mag > 0:
            repulsion_vec = repulsion_vec / mag * min(mag, max_force)
        return repulsion_vec
    
    def _check_captures(self):
        captures = []
        for pur_idx, pursuer in enumerate(self.pursuers):
            pur_pos = pursuer.get_pose()[:2]
            for ev_idx in range(self.num_evaders):
                if not self.evader_alive[ev_idx]:
                    continue
                ev_pos = self.evaders[ev_idx].get_pose()[:2]
                dist = np.linalg.norm(pur_pos - ev_pos)
                if dist < self.capture_radius:
                    captures.append((pur_idx, ev_idx))
                    break
        return captures
    
    def _handle_capture(self, cap_pur_idx, cap_ev_idx):
        """Role swap (NATURAL - NO TELEPORT)."""
        captured_evader = self.evaders[cap_ev_idx]
        capturing_pursuer = self.pursuers[cap_pur_idx]
        
        self.evader_alive[cap_ev_idx] = False
        
        captured_evader.max_linear_vel = 1.5
        captured_evader.max_angular_vel = 2.5
        captured_evader.wheel_force = 20.0
        
        captured_evader._recolor(HEAD_COLOR)
        for pur in self.pursuers:
            pur._recolor(FOLLOWER_COLOR)
        
        self.pursuers.pop(cap_pur_idx)
        self.pursuers.insert(0, captured_evader)
        self.pursuers.insert(1, capturing_pursuer)
        
        new_head_pose = captured_evader.get_pose()
        nh_x, nh_y, nh_yaw = new_head_pose
        self.head_path.append((nh_x, nh_y, nh_yaw))
        self.head_trail_viz.clear()
        self.head_trail_viz.append((nh_x, nh_y))
        
        self._disable_all_pursuer_collisions()
        
        print(f"[SNAKE] Capture! Evader {cap_ev_idx} -> NEW HEAD. "
              f"Old pursuer_{cap_pur_idx} -> follower #1. "
              f"Snake length: {len(self.pursuers)}, "
              f"Evaders remaining: {int(self.evader_alive.sum())}")
    
    def _disable_all_pursuer_collisions(self):
        n_pursuers = len(self.pursuers)
        for i in range(n_pursuers):
            for j in range(i + 1, n_pursuers):
                id_i = self.pursuers[i].robot_id
                id_j = self.pursuers[j].robot_id
                try:
                    p.setCollisionFilterPair(id_i, id_j, -1, -1, 0)
                    n_joints_i = p.getNumJoints(id_i, physicsClientId=self.client)
                    n_joints_j = p.getNumJoints(id_j, physicsClientId=self.client)
                    for li in range(n_joints_i):
                        for lj in range(n_joints_j):
                            try:
                                p.setCollisionFilterPair(id_i, id_j, li, lj, 0)
                            except:
                                pass
                except:
                    pass
    
    def _get_observation(self):
        head_pose = self.pursuers[0].get_pose()
        
        evader_poses = np.full((self.num_evaders, 3), np.nan)
        for i, bot in enumerate(self.evaders):
            if self.evader_alive[i]:
                evader_poses[i] = bot.get_pose()
        
        all_pursuer_poses = [bot.get_pose() for bot in self.pursuers]
        follower_poses = all_pursuer_poses[1:] if len(all_pursuer_poses) > 1 else []
        
        follower_alerts = []
        for i in range(1, len(self.pursuers)):
            pur_pos = self.pursuers[i].get_pose()[:2]
            for j in range(self.num_evaders):
                if not self.evader_alive[j]:
                    continue
                ev_pos = self.evaders[j].get_pose()[:2]
                dist = np.linalg.norm(ev_pos - pur_pos)
                if dist < self.detection_radius:
                    follower_alerts.append(ev_pos.tolist())
        
        return {
            "head_pose": head_pose,
            "follower_poses": follower_poses,
            "all_pursuer_poses": all_pursuer_poses,
            "evader_poses": evader_poses,
            "evader_alive": self.evader_alive.copy(),
            "follower_alerts": follower_alerts,
            "snake_length": len(self.pursuers),
            "time": self.time,
            "step": self.step_count,
            "num_alive_evaders": int(self.evader_alive.sum()),
        }
    
    def close(self):
        if self.client is not None and p.isConnected(self.client):
            p.disconnect(self.client)
            self.client = None
    
    def get_obstacle_list(self):
        return self.arena.get_obstacle_list() if self.arena else []