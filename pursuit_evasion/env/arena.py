"""
Arena for Pursuit-Evasion simulation.
Creates boundary walls + random obstacles in PyBullet.
"""
import numpy as np
import pybullet as p


class Arena:
    """
    PyBullet arena with boundary walls and random box obstacles.
    
    Usage:
        arena = Arena(client, size=[20,20], num_obstacles=10)
        arena.build()        # spawn walls + obstacles
        arena.reset()        # remove & re-randomize obstacles
        free_pos = arena.get_free_position(min_clearance=1.0)  # for robot spawning
    """

    def __init__(self, client, config):
        """
        Args:
            client: PyBullet client ID
            config: dict with keys:
                - arena.size: [width, height]
                - arena.wall_height: float
                - arena.num_obstacles: int
                - arena.obstacle_min_size: float
                - arena.obstacle_max_size: float
        """
        self.client = client
        self.size = np.array(config["arena"]["size"], dtype=float)
        self.wall_height = float(config["arena"]["wall_height"])
        self.num_obstacles = int(config["arena"]["num_obstacles"])
        self.obs_min_size = float(config["arena"]["obstacle_min_size"])
        self.obs_max_size = float(config["arena"]["obstacle_max_size"])

        # Will be populated by build()
        self.wall_ids = []
        self.obstacle_ids = []
        self.obstacles = []  # list of (center, half_extents) for collision checks
        self._built = False

    def build(self, seed=None):
        """Build arena: spawn ground, walls, and random obstacles."""
        if self._built:
            self.clear()
        
        if seed is not None:
            np.random.seed(seed)
        
        self._build_walls()
        self._build_obstacles()
        self._built = True

    def clear(self):
        """Remove all walls and obstacles from simulation."""
        for obj_id in self.wall_ids + self.obstacle_ids:
            try:
                p.removeBody(obj_id, physicsClientId=self.client)
            except:
                pass
        self.wall_ids = []
        self.obstacle_ids = []
        self.obstacles = []
        self._built = False

    def _build_walls(self):
        """Create 4 boundary walls around the arena."""
        w, h = self.size
        wall_thickness = 0.2
        wall_height = self.wall_height
        half_w = w / 2.0
        half_h = h / 2.0

        # Wall definitions: (center_pos, half_extents)
        walls = [
            # Bottom wall
            ([0, -half_h, wall_height / 2], [half_w + wall_thickness, wall_thickness / 2, wall_height / 2]),
            # Top wall
            ([0, half_h, wall_height / 2], [half_w + wall_thickness, wall_thickness / 2, wall_height / 2]),
            # Left wall
            ([-half_w, 0, wall_height / 2], [wall_thickness / 2, half_h + wall_thickness, wall_height / 2]),
            # Right wall
            ([half_w, 0, wall_height / 2], [wall_thickness / 2, half_h + wall_thickness, wall_height / 2]),
        ]

        for center, half_extents in walls:
            col_shape = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=self.client,
            )
            viz_shape = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=[0.3, 0.3, 0.3, 1.0],
                physicsClientId=self.client,
            )
            wall_id = p.createMultiBody(
                baseMass=0,  # static
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=viz_shape,
                basePosition=center,
                physicsClientId=self.client,
            )
            self.wall_ids.append(wall_id)
            # Save for collision check
            self.obstacles.append((np.array(center[:2]), np.array(half_extents[:2])))

    def _build_obstacles(self):
        """Spawn random box obstacles inside the arena."""
        w, h = self.size
        margin = 1.0  # keep obstacles away from walls

        for _ in range(self.num_obstacles):
            # Random size
            size = np.random.uniform(self.obs_min_size, self.obs_max_size, size=2)
            half_extents_xy = size / 2.0
            half_extent_z = np.random.uniform(0.3, 0.8)
            
            # Random position (try a few times to avoid overlap)
            placed = False
            for _ in range(50):
                x = np.random.uniform(-w/2 + margin, w/2 - margin)
                y = np.random.uniform(-h/2 + margin, h/2 - margin)
                center = np.array([x, y])
                
                # Check overlap with existing obstacles
                if not self._check_overlap(center, half_extents_xy, clearance=0.5):
                    placed = True
                    break
            
            if not placed:
                continue  # skip if can't place

            half_extents = [half_extents_xy[0], half_extents_xy[1], half_extent_z]
            center_pos = [x, y, half_extent_z]

            col_shape = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=self.client,
            )
            viz_shape = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=[0.7, 0.5, 0.3, 1.0],
                physicsClientId=self.client,
            )
            obs_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=viz_shape,
                basePosition=center_pos,
                physicsClientId=self.client,
            )
            self.obstacle_ids.append(obs_id)
            self.obstacles.append((center, half_extents_xy))

    def _check_overlap(self, center, half_extents, clearance=0.5):
        """Check if a new obstacle overlaps with existing ones."""
        for obs_center, obs_half in self.obstacles:
            # AABB overlap test
            dx = abs(center[0] - obs_center[0])
            dy = abs(center[1] - obs_center[1])
            min_dx = half_extents[0] + obs_half[0] + clearance
            min_dy = half_extents[1] + obs_half[1] + clearance
            if dx < min_dx and dy < min_dy:
                return True
        return False

    def get_free_position(self, min_clearance=1.0, max_attempts=100):
        """
        Find a random position inside the arena that is at least
        `min_clearance` away from all obstacles and walls.
        Returns: (x, y) or None if failed.
        """
        w, h = self.size
        margin = min_clearance
        
        for _ in range(max_attempts):
            x = np.random.uniform(-w/2 + margin, w/2 - margin)
            y = np.random.uniform(-h/2 + margin, h/2 - margin)
            pos = np.array([x, y])
            
            # Check distance to all obstacles
            free = True
            for obs_center, obs_half in self.obstacles:
                # Closest point on AABB to pos
                closest_x = np.clip(pos[0], obs_center[0] - obs_half[0], obs_center[0] + obs_half[0])
                closest_y = np.clip(pos[1], obs_center[1] - obs_half[1], obs_center[1] + obs_half[1])
                dx = pos[0] - closest_x
                dy = pos[1] - closest_y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_clearance:
                    free = False
                    break
            
            if free:
                return (x, y)
        
        return None

    def is_inside(self, x, y, margin=0.0):
        """Check if (x, y) is inside arena bounds (with optional margin)."""
        w, h = self.size
        return (
            -w/2 + margin <= x <= w/2 - margin
            and -h/2 + margin <= y <= h/2 - margin
        )

    def get_obstacle_list(self):
        """Returns list of (center_xy, half_extents_xy) for planning."""
        return self.obstacles.copy()