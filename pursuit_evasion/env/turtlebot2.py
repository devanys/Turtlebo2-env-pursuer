"""
TurtleBot2 class for Pursuit-Evasion simulation.

Wraps a single TurtleBot2 URDF in PyBullet with:
- Differential drive control (v, omega) -> wheel velocities
- Virtual 360° LiDAR via batch ray casting
- Pose & velocity getter
- Bumper contact detection
- Auto-detection of wheel joints
- Custom color support (for visualizing pursuer vs evader)
"""

import os
import numpy as np
import pybullet as p


class TurtleBot2:
    """
    Single TurtleBot2 instance in PyBullet.
    """

    WHEEL_RADIUS = 0.033
    WHEEL_SEPARATION = 0.23
    MAX_LINEAR_VEL = 0.65
    MAX_ANGULAR_VEL = 1.5
    WHEEL_FORCE = 10.0

    LIDAR_NUM_RAYS = 180
    LIDAR_RANGE_MAX = 5.0
    LIDAR_RAY_HEIGHT = 0.1

    def __init__(
        self,
        client,
        start_pos,
        start_yaw=0.0,
        name="tb2",
        urdf_path=None,
        robot_params=None,
        lidar_params=None,
        verbose=False,
        color=None,
    ):
        """
        Args:
            ...
            color: optional [r, g, b, a] in 0-1 range to recolor the robot
                   If None, use URDF default color (white).
        """
        self.client = client
        self.name = name
        self.verbose = verbose

        rp = robot_params or {}
        self.wheel_radius = rp.get("wheel_radius", self.WHEEL_RADIUS)
        self.wheel_separation = rp.get("wheel_separation", self.WHEEL_SEPARATION)
        self.max_linear_vel = rp.get("max_linear_vel", self.MAX_LINEAR_VEL)
        self.max_angular_vel = rp.get("max_angular_vel", self.MAX_ANGULAR_VEL)
        self.wheel_force = rp.get("wheel_force", self.WHEEL_FORCE)

        lp = lidar_params or {}
        self.lidar_num_rays = lp.get("num_rays", self.LIDAR_NUM_RAYS)
        self.lidar_range_max = lp.get("range_max", self.LIDAR_RANGE_MAX)
        self.lidar_ray_height = lp.get("ray_height", self.LIDAR_RAY_HEIGHT)

        if urdf_path is None:
            raise ValueError("urdf_path must be provided")
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        self.urdf_path = urdf_path

        start_orn = p.getQuaternionFromEuler([0, 0, start_yaw])
        self.robot_id = p.loadURDF(
            self.urdf_path,
            start_pos,
            start_orn,
            useMaximalCoordinates=False,
            physicsClientId=self.client,
        )

        self._detect_wheels()
        self.set_velocity(0.0, 0.0)
        self.last_cmd = (0.0, 0.0)

        # === RECOLOR ROBOT IF COLOR PROVIDED ===
        if color is not None:
            self._recolor(color)

    def _recolor(self, color):
        """Recolor all visual shapes of the robot.
        
        Args:
            color: [r, g, b, a] in 0-1 range
        """
        # Recolor base link (linkIndex = -1)
        try:
            p.changeVisualShape(
                self.robot_id,
                linkIndex=-1,
                shapeIndex=-1,
                rgbaColor=color,
                physicsClientId=self.client,
            )
        except Exception as e:
            if self.verbose:
                print(f"[{self.name}] Could not recolor base link: {e}")

        # Recolor all child links
        n_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        for i in range(n_joints):
            try:
                p.changeVisualShape(
                    self.robot_id,
                    linkIndex=i,
                    shapeIndex=-1,
                    rgbaColor=color,
                    physicsClientId=self.client,
                )
            except Exception as e:
                if self.verbose and i == 0:
                    print(f"[{self.name}] Could not recolor link {i}: {e}")

    def _detect_wheels(self):
        """Find the two drive wheels automatically."""
        n_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)

        if self.verbose:
            print(f"[{self.name}] === Joint inspection ({n_joints} joints) ===")
            print(f"{'Idx':<5} {'Name':<35} {'Type':<12} {'ChildLink':<30}")
            print("-" * 90)

        wheel_candidates = []
        type_map = {
            p.JOINT_REVOLUTE: "REVOLUTE",
            p.JOINT_PRISMATIC: "PRISMATIC",
            p.JOINT_FIXED: "FIXED",
            p.JOINT_SPHERICAL: "SPHERICAL",
            p.JOINT_PLANAR: "PLANAR",
            p.JOINT_GEAR: "GEAR",
        }

        for i in range(n_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            idx = info[0]
            name = info[1].decode("utf-8")
            jtype = info[2]
            child_link = info[12].decode("utf-8") if isinstance(info[12], bytes) else info[12]
            type_str = type_map.get(jtype, f"UNKNOWN({jtype})")

            if self.verbose:
                print(f"{idx:<5} {name:<35} {type_str:<12} {child_link:<30}")

            if jtype not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                continue

            name_lower = name.lower()
            child_lower = child_link.lower()
            skip_keywords = ["caster", "gyro", "cliff", "wheel_drop", "button", "motor"]
            if any(kw in name_lower or kw in child_lower for kw in skip_keywords):
                continue

            if "wheel" not in name_lower and "wheel" not in child_lower:
                continue

            wheel_candidates.append({
                "idx": idx,
                "name": name,
                "child_link": child_link,
            })

        left_candidates = []
        right_candidates = []
        for wc in wheel_candidates:
            combined = (wc["name"] + " " + wc["child_link"]).lower()
            if "left" in combined:
                left_candidates.append(wc)
            elif "right" in combined:
                right_candidates.append(wc)

        if left_candidates and right_candidates:
            self.left_wheel_idx = left_candidates[0]["idx"]
            self.right_wheel_idx = right_candidates[0]["idx"]
            self.left_wheel_name = left_candidates[0]["name"]
            self.right_wheel_name = right_candidates[0]["name"]
        elif len(wheel_candidates) >= 2:
            if self.verbose:
                print(f"[{self.name}] WARNING: L/R not identified by name. Using first 2 wheel joints.")
            wheel_candidates.sort(key=lambda c: c["idx"])
            self.left_wheel_idx = wheel_candidates[0]["idx"]
            self.right_wheel_idx = wheel_candidates[1]["idx"]
            self.left_wheel_name = wheel_candidates[0]["name"]
            self.right_wheel_name = wheel_candidates[1]["name"]
        else:
            raise RuntimeError(
                f"[{self.name}] Could not find 2 drive wheels. "
                f"Candidates: {[c['name'] for c in wheel_candidates]}"
            )

        if not self.verbose:
            print(f"[{self.name}] wheels: L='{self.left_wheel_name}' (idx={self.left_wheel_idx}), "
                  f"R='{self.right_wheel_name}' (idx={self.right_wheel_idx})")

    def set_velocity(self, v, omega):
        """Command differential drive velocity."""
        v = float(np.clip(v, -self.max_linear_vel, self.max_linear_vel))
        omega = float(np.clip(omega, -self.max_angular_vel, self.max_angular_vel))
        self.last_cmd = (v, omega)

        v_left = (v - omega * self.wheel_separation / 2.0) / self.wheel_radius
        v_right = (v + omega * self.wheel_separation / 2.0) / self.wheel_radius

        for wheel_idx, target_vel in [
            (self.left_wheel_idx, v_left),
            (self.right_wheel_idx, v_right),
        ]:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=wheel_idx,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=target_vel,
                force=self.wheel_force,
                physicsClientId=self.client,
            )

    def stop(self):
        self.set_velocity(0.0, 0.0)

    def get_pose(self):
        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.client
        )
        euler = p.getEulerFromQuaternion(orn)
        return np.array([pos[0], pos[1], euler[2]], dtype=float)

    def get_velocity(self):
        v_left_state = p.getJointState(
            self.robot_id, self.left_wheel_idx, physicsClientId=self.client
        )
        v_right_state = p.getJointState(
            self.robot_id, self.right_wheel_idx, physicsClientId=self.client
        )
        omega_left = v_left_state[1]
        omega_right = v_right_state[1]

        v = self.wheel_radius * (omega_left + omega_right) / 2.0
        omega = self.wheel_radius * (omega_right - omega_left) / self.wheel_separation
        return np.array([v, omega], dtype=float)

    def get_lidar(self, num_rays=None, range_max=None):
        """Virtual 360° LiDAR via batch ray casting."""
        if num_rays is None:
            num_rays = self.lidar_num_rays
        if range_max is None:
            range_max = self.lidar_range_max

        if not p.isConnected(self.client):
            return np.full(num_rays, range_max, dtype=float)

        try:
            pos = p.getBasePositionAndOrientation(
                self.robot_id, physicsClientId=self.client
            )[0]
        except p.error:
            return np.full(num_rays, range_max, dtype=float)

        x, y, z = pos
        z_ray = z + self.lidar_ray_height

        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        ray_from = np.tile([x, y, z_ray], (num_rays, 1))
        ray_to = np.column_stack([
            x + range_max * np.cos(angles),
            y + range_max * np.sin(angles),
            np.full(num_rays, z_ray),
        ])

        try:
            results = p.rayTestBatch(
                ray_from.tolist(),
                ray_to.tolist(),
                physicsClientId=self.client,
            )
        except p.error:
            return np.full(num_rays, range_max, dtype=float)

        if results is None:
            return np.full(num_rays, range_max, dtype=float)

        ranges = np.full(num_rays, range_max, dtype=float)
        for i, hit in enumerate(results):
            obj_id = hit[0]
            hit_fraction = hit[2]
            if obj_id >= 0:
                ranges[i] = hit_fraction * range_max

        return ranges

    def get_lidar_with_ids(self, num_rays=None, range_max=None):
        """Like get_lidar, but also returns object IDs hit by each ray."""
        if num_rays is None:
            num_rays = self.lidar_num_rays
        if range_max is None:
            range_max = self.lidar_range_max

        pos = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.client
        )[0]
        x, y, z = pos
        z_ray = z + self.lidar_ray_height

        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        ray_from = np.tile([x, y, z_ray], (num_rays, 1))
        ray_to = np.column_stack([
            x + range_max * np.cos(angles),
            y + range_max * np.sin(angles),
            np.full(num_rays, z_ray),
        ])

        results = p.rayTestBatch(
            ray_from.tolist(),
            ray_to.tolist(),
            physicsClientId=self.client,
        )

        ranges = np.full(num_rays, range_max, dtype=float)
        obj_ids = np.full(num_rays, -1, dtype=int)
        for i, hit in enumerate(results):
            obj_id = hit[0]
            hit_fraction = hit[2]
            if obj_id >= 0:
                ranges[i] = hit_fraction * range_max
                obj_ids[i] = obj_id

        return ranges, obj_ids

    def get_contacts(self):
        contacts = p.getContactPoints(
            bodyA=self.robot_id, physicsClientId=self.client
        )
        return contacts

    def is_collided(self):
        contacts = self.get_contacts()
        for c in contacts:
            other_id = c[2] if c[0] == self.robot_id else c[1]
            if other_id != -1:
                return True
        return False

    def step(self, num_steps=1):
        for _ in range(num_steps):
            p.stepSimulation(physicsClientId=self.client)

    def reset(self, pos, yaw=0.0):
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(
            self.robot_id, pos, orn, physicsClientId=self.client
        )
        self.set_velocity(0.0, 0.0)