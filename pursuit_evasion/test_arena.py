"""
Test script for Arena class.
Spawns arena + 3 TurtleBot2 at random free positions, drive around briefly.
"""
import sys
import os
import time
import yaml
import numpy as np
import pybullet as p
import pybullet_data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.turtlebot2 import TurtleBot2
from env.arena import Arena


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    timestep = config["simulation"]["timestep"]
    steps_per_second = int(1.0 / timestep)

    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(timestep)

    # Ground plane
    p.loadURDF("plane.urdf", physicsClientId=client)

    # Build arena
    print("\n=== Building Arena ===")
    arena = Arena(client, config)
    arena.build(seed=42)
    print(f"Walls: {len(arena.wall_ids)}, Obstacles: {len(arena.obstacle_ids)}")
    print(f"Arena size: {arena.size}")
    print(f"Obstacle list (center_xy, half_xy):")
    for i, (c, h) in enumerate(arena.obstacles):
        print(f"  [{i}] center=({c[0]:.2f}, {c[1]:.2f}), half=({h[0]:.2f}, {h[1]:.2f})")

    # Spawn 3 robots at free positions
    print("\n=== Spawning 3 TurtleBot2 ===")
    bots = []
    for i in range(3):
        free_pos = arena.get_free_position(min_clearance=1.5)
        if free_pos is None:
            print(f"  [!] Could not find free position for bot {i}")
            continue
        x, y = free_pos
        yaw = np.random.uniform(0, 2 * np.pi)
        bot = TurtleBot2(
            client=client,
            start_pos=[x, y, 0.05],
            start_yaw=yaw,
            name=f"bot_{i}",
            urdf_path=config["robots"]["urdf_path"],
            robot_params=config["robot_params"],
            lidar_params=config["lidar"],
        )
        bots.append(bot)

    # Test 1: Let robots drive randomly for 5 seconds
    print("\n=== Test 1: Random drive for 5 seconds ===")
    print("(close window anytime to exit gracefully)")
    for bot in bots:
        v = np.random.uniform(0.1, 0.4)
        omega = np.random.uniform(-0.5, 0.5)
        bot.set_velocity(v=v, omega=omega)
    
    for step in range(5 * steps_per_second):
        if not p.isConnected(client):
            print("[!] Window closed. Exiting.")
            return
        try:
            p.stepSimulation(physicsClientId=client)
            time.sleep(timestep)
        except p.error:
            print("[!] Physics error. Exiting.")
            return
        
        # Print status every second
        if step % steps_per_second == 0:
            print(f"  t={step/steps_per_second:.0f}s:")
            for bot in bots:
                pose = bot.get_pose()
                print(f"    {bot.name}: x={pose[0]:.2f}, y={pose[1]:.2f}, yaw={pose[2]:.2f}")

    # Test 2: LiDAR scan each robot
    print("\n=== Test 2: LiDAR scan ===")
    for bot in bots:
        ranges = bot.get_lidar()
        print(f"  {bot.name}: min={ranges.min():.2f}m, mean={ranges.mean():.2f}m, max={ranges.max():.2f}m")

    print("\n=== Arena test PASSED! ===")
    print("Close the PyBullet window to exit.")
    
    try:
        while p.isConnected(client):
            p.stepSimulation(physicsClientId=client)
            time.sleep(timestep)
    except (p.error, KeyboardInterrupt):
        pass
    finally:
        if p.isConnected(client):
            p.disconnect(client)
        print("Disconnected.")


if __name__ == "__main__":
    main()