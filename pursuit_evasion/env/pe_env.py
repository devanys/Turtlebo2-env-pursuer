"""
Pursuit-Evasion Environment for multi-robot simulation in PyBullet.
DEMO VERSION: Pursuer FAST (1.5 m/s) + BLUE, Evader SLOW (0.5 m/s) + MAROON.
"""
import numpy as np
import pybullet as p
import pybullet_data

from env.arena import Arena
from env.turtlebot2 import TurtleBot2


# === COLOR DEFINITIONS (RGBA, 0-1) ===
PURSUER_COLOR = [0.05, 0.28, 0.63, 1.0]  # Biru Tua (#0D47A1)
EVADER_COLOR = [0.72, 0.11, 0.11, 1.0]   # Merah Maroon (#B71C1C)


class PursuitEvasionEnv:
    """Pursuit-Evasion environment with full observability for pursuers."""
    
    def __init__(self, config, mode="gui"):
        self.config = config
        self.mode = mode
        
        self.sim_cfg = config["simulation"]
        self.arena_cfg = config["arena"]
        self.robots_cfg = config["robots"]
        self.robot_params = config["robot_params"]
        self.lidar_params = config["lidar"]
        self.game_cfg = config["game"]
        
        self.num_pursuers = int(self.robots_cfg["num_pursuers"])
        self.num_evaders = int(self.robots_cfg["num_evaders"])
        self.capture_radius = float(self.game_cfg["capture_radius"])
        self.max_episode_time = float(self.sim_cfg["max_episode_time"])
        self.timestep = float(self.sim_cfg["timestep"])
        self.control_freq = int(self.sim_cfg["control_freq"])
        self.steps_per_control = int(round(1.0 / (self.timestep * self.control_freq)))
        
        self.client = None
        self.arena = None
        self.pursuers = []
        self.evaders = []
        self.evader_alive = np.zeros(self.num_evaders, dtype=bool)
        self.time = 0.0
        self.step_count = 0
        self.episode_count = 0
        
        self.collision_counts = np.zeros(self.num_pursuers + self.num_evaders, dtype=int)
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
    
    def reset(self, seed=None, spawn_clearance=None):
        if seed is not None:
            np.random.seed(seed)
        
        if spawn_clearance is None:
            spawn_clearance = float(self.robots_cfg["spawn_clearance"])
        
        self._clear_robots()
        if self.arena is not None:
            self.arena.clear()
        
        self.arena = Arena(self.client, self.config)
        self.arena.build(seed=seed)
        
        self.pursuers = []
        self.evaders = []
        self.evader_alive = np.zeros(self.num_evaders, dtype=bool)
        self.time = 0.0
        self.step_count = 0
        self.collision_counts = np.zeros(self.num_pursuers + self.num_evaders, dtype=int)
        self.capture_events = []
        
        # Spawn pursuers (left, BLUE)
        self._spawn_robots(
            count=self.num_pursuers,
            x_range=(-4.0, -1.5),
            y_range=(-3.0, 3.0),
            clearance=spawn_clearance,
            name_prefix="pursuer",
            robot_list=self.pursuers,
            color=PURSUER_COLOR,
        )
        
        # Spawn evaders (right, MAROON)
        self._spawn_robots(
            count=self.num_evaders,
            x_range=(1.5, 4.0),
            y_range=(-3.0, 3.0),
            clearance=spawn_clearance,
            name_prefix="evader",
            robot_list=self.evaders,
            color=EVADER_COLOR,
        )
        
        self.evader_alive[:] = True
        
        self.episode_count += 1
        print(f"\n[ENV] Episode {self.episode_count} reset: "
              f"{self.num_pursuers} pursuers (BLUE), {self.num_evaders} evaders (MAROON)")
        
        return self._get_observation()
    
    def _spawn_robots(self, count, x_range, y_range, clearance, name_prefix, robot_list, color=None):
        existing_positions = [bot.get_pose()[:2] for bot in self.pursuers + self.evaders]
        
        for i in range(count):
            pos = None
            for _ in range(200):
                x = np.random.uniform(*x_range)
                y = np.random.uniform(*y_range)
                
                if not self._is_free_from_obstacles(x, y, clearance):
                    continue
                
                ok = True
                for ex_pos in existing_positions:
                    if np.linalg.norm(np.array([x, y]) - ex_pos) < clearance:
                        ok = False
                        break
                if ok:
                    pos = (x, y)
                    break
            
            if pos is None:
                pos = self.arena.get_free_position(min_clearance=clearance)
                if pos is None:
                    raise RuntimeError(f"Could not find spawn position for {name_prefix}_{i}")
            
            yaw = np.random.uniform(0, 2 * np.pi)
            existing_positions.append(np.array(pos))
            
            # === OVERRIDE ROBOT PARAMS PER TYPE ===
            robot_params = dict(self.robot_params)
            if name_prefix == "pursuer":
                robot_params["max_linear_vel"] = 1.5
                robot_params["max_angular_vel"] = 2.5
                robot_params["wheel_force"] = 20.0
            else:  # evader
                robot_params["max_linear_vel"] = 0.5
                robot_params["max_angular_vel"] = 1.5
                robot_params["wheel_force"] = 10.0
            
            total_existing = len(self.pursuers) + len(self.evaders)
            bot = TurtleBot2(
                client=self.client,
                start_pos=[pos[0], pos[1], 0.05],
                start_yaw=yaw,
                name=f"{name_prefix}_{i}",
                urdf_path=self.robots_cfg["urdf_path"],
                robot_params=robot_params,
                lidar_params=self.lidar_params,
                verbose=(total_existing == 0),
                color=color,  # === SET COLOR HERE ===
            )
            robot_list.append(bot)
    
    def _is_free_from_obstacles(self, x, y, clearance):
        for obs_center, obs_half in self.arena.obstacles:
            closest_x = np.clip(x, obs_center[0] - obs_half[0], obs_center[0] + obs_half[0])
            closest_y = np.clip(y, obs_center[1] - obs_half[1], obs_center[1] + obs_half[1])
            dx = x - closest_x
            dy = y - closest_y
            if np.sqrt(dx * dx + dy * dy) < clearance:
                return False
        return True
    
    def _clear_robots(self):
        for bot in self.pursuers + self.evaders:
            try:
                p.removeBody(bot.robot_id, physicsClientId=self.client)
            except:
                pass
        self.pursuers = []
        self.evaders = []
    
    def step(self, actions):
        for i, (v, omega) in enumerate(actions["pursuers"]):
            self.pursuers[i].set_velocity(v, omega)
        for i, (v, omega) in enumerate(actions["evaders"]):
            if self.evader_alive[i]:
                self.evaders[i].set_velocity(v, omega)
            else:
                self.evaders[i].stop()
        
        for _ in range(self.steps_per_control):
            if not p.isConnected(self.client):
                self.time += self.timestep
                continue
            try:
                p.stepSimulation(physicsClientId=self.client)
            except p.error:
                continue
            self.time += self.timestep
        
        self.step_count += 1
        
        self._update_collision_counts()
        
        new_captures = self._check_captures()
        for ev_idx, pur_idx in new_captures:
            self.evader_alive[ev_idx] = False
            self.evaders[ev_idx].stop()
            self.capture_events.append((self.step_count, pur_idx, ev_idx))
            print(f"[ENV] Step {self.step_count}: pursuer_{pur_idx} caught evader_{ev_idx}")
        
        rewards = self._compute_rewards(new_captures)
        
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
            "capture_events": self.capture_events.copy(),
            "collision_counts": self.collision_counts.copy(),
            "current_collisions": int(self._current_collision.sum()) if hasattr(self, "_current_collision") else 0,
            "pursuer_winner": bool(not self.evader_alive.any()),
            "evader_winner": bool(self.time >= self.max_episode_time and self.evader_alive.any()),
        }
        
        return self._get_observation(), rewards, done, info
    
    def _check_captures(self):
        new_captures = []
        for ev_idx, evader in enumerate(self.evaders):
            if not self.evader_alive[ev_idx]:
                continue
            ev_pos = evader.get_pose()[:2]
            for pur_idx, pursuer in enumerate(self.pursuers):
                pur_pos = pursuer.get_pose()[:2]
                dist = np.linalg.norm(ev_pos - pur_pos)
                if dist < self.capture_radius:
                    new_captures.append((ev_idx, pur_idx))
                    break
        return new_captures
    
    def _update_collision_counts(self):
        all_bots = self.pursuers + self.evaders
        
        if not hasattr(self, "_prev_collision_state"):
            self._prev_collision_state = np.zeros(len(all_bots), dtype=bool)
        
        current_collision = np.zeros(len(all_bots), dtype=bool)
        
        for i, bot in enumerate(all_bots):
            contacts = p.getContactPoints(bodyA=bot.robot_id, physicsClientId=self.client)
            in_collision = False
            for c in contacts:
                other_id = c[2]
                if other_id == 0:
                    continue
                if c[9] < 1.0:
                    continue
                in_collision = True
                break
            
            current_collision[i] = in_collision
            
            if in_collision and not self._prev_collision_state[i]:
                self.collision_counts[i] += 1
        
        self._prev_collision_state = current_collision.copy()
        self._current_collision = current_collision
    
    def _compute_rewards(self, new_captures):
        capture_bonus = float(self.game_cfg["evader_caught_reward"])
        collision_penalty = float(self.game_cfg["pursuer_collision_penalty"])
        
        pursuer_rewards = np.zeros(self.num_pursuers)
        for i in range(self.num_pursuers):
            if self.collision_counts[i] > 0:
                pursuer_rewards[i] += collision_penalty * 0.1
            
            if self.evader_alive.any():
                pur_pos = self.pursuers[i].get_pose()[:2]
                ev_positions = np.array([
                    self.evaders[j].get_pose()[:2]
                    for j in range(self.num_evaders)
                    if self.evader_alive[j]
                ])
                dists = np.linalg.norm(ev_positions - pur_pos, axis=1)
                min_dist = dists.min()
                pursuer_rewards[i] += 0.01 * max(0, 1.0 - min_dist / 10.0)
        
        for ev_idx, pur_idx in new_captures:
            pursuer_rewards[pur_idx] += capture_bonus
        
        evader_rewards = np.zeros(self.num_evaders)
        for i in range(self.num_evaders):
            if not self.evader_alive[i]:
                continue
            
            evader_rewards[i] += 0.01
            
            if self.collision_counts[self.num_pursuers + i] > 0:
                evader_rewards[i] += collision_penalty * 0.1
            
            ev_pos = self.evaders[i].get_pose()[:2]
            pur_positions = np.array([p.get_pose()[:2] for p in self.pursuers])
            dists = np.linalg.norm(pur_positions - ev_pos, axis=1)
            min_dist = dists.min()
            evader_rewards[i] += 0.01 * min(min_dist / 10.0, 1.0)
        
        for ev_idx, _ in new_captures:
            evader_rewards[ev_idx] -= 1.0
        
        return {"pursuers": pursuer_rewards, "evaders": evader_rewards}
    
    def _get_observation(self):
        pursuer_poses = np.array([bot.get_pose() for bot in self.pursuers])
        
        evader_poses = np.full((self.num_evaders, 3), np.nan)
        for i, bot in enumerate(self.evaders):
            if self.evader_alive[i]:
                evader_poses[i] = bot.get_pose()
        
        num_rays = self.lidar_params["num_rays"]
        pursuer_lidars = np.array([bot.get_lidar() for bot in self.pursuers])
        
        evader_lidars = np.full((self.num_evaders, num_rays), np.nan)
        for i, bot in enumerate(self.evaders):
            if self.evader_alive[i]:
                evader_lidars[i] = bot.get_lidar()
        
        return {
            "pursuer_poses": pursuer_poses,
            "evader_poses": evader_poses,
            "evader_alive": self.evader_alive.copy(),
            "pursuer_lidars": pursuer_lidars,
            "evader_lidars": evader_lidars,
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
    
    def get_all_robot_positions(self):
        positions = {}
        for i, bot in enumerate(self.pursuers):
            positions[f"pursuer_{i}"] = bot.get_pose().tolist()
        for i, bot in enumerate(self.evaders):
            if self.evader_alive[i]:
                positions[f"evader_{i}"] = bot.get_pose().tolist()
        return positions