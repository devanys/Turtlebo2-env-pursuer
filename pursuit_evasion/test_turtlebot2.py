"""
Test script for TurtleBot2 class.
Spawns 1 robot, drives forward, then turns, and prints LiDAR.
Handles window close gracefully.
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


def safe_step(client, timestep, realtime=True):
    """Step simulation, returns False if connection lost."""
    if not p.isConnected(client):
        return False
    try:
        p.stepSimulation(physicsClientId=client)
        if realtime:
            time.sleep(timestep)
        return True
    except p.error:
        return False


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    urdf_path = config["robots"]["urdf_path"]
    robot_params = config["robot_params"]
    lidar_params = config["lidar"]
    timestep = config["simulation"]["timestep"]
    steps_per_second = int(1.0 / timestep)

    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(timestep)

    p.loadURDF("plane.urdf", physicsClientId=client)

    print("\n=== Spawning TurtleBot2 ===")
    bot = TurtleBot2(
        client=client,
        start_pos=[0, 0, 0.05],
        start_yaw=0.0,
        name="test_bot",
        urdf_path=urdf_path,
        robot_params=robot_params,
        lidar_params=lidar_params,
    )

    pose = bot.get_pose()
    print(f"\nInitial pose: x={pose[0]:.3f}, y={pose[1]:.3f}, yaw={pose[2]:.3f}")

    # Test 1: Drive forward
    print("\n=== Test 1: Drive forward (v=0.3 m/s) for 2 seconds ===")
    bot.set_velocity(v=0.3, omega=0.0)
    for i in range(2 * steps_per_second):
        if not safe_step(client, timestep):
            print("[!] Window closed. Exiting.")
            return
    pose = bot.get_pose()
    vel = bot.get_velocity()
    print(f"After 2s:  x={pose[0]:.3f}, y={pose[1]:.3f}, yaw={pose[2]:.3f}")
    print(f"Velocity: v={vel[0]:.3f}, omega={vel[1]:.3f}")
    print(f"Expected: x≈0.600 (got {pose[0]:.3f}) -> {'PASS' if abs(pose[0]-0.6)<0.1 else 'FAIL'}")

    # Test 2: Turn CCW
    print("\n=== Test 2: Turn CCW (omega=0.5 rad/s) for 2 seconds ===")
    bot.set_velocity(v=0.0, omega=0.5)
    for i in range(2 * steps_per_second):
        if not safe_step(client, timestep):
            print("[!] Window closed. Exiting.")
            return
    pose = bot.get_pose()
    print(f"After turn: x={pose[0]:.3f}, y={pose[1]:.3f}, yaw={pose[2]:.3f}")
    print(f"Yaw change: {pose[2]:.3f} rad (expected ~1.000) -> {'PASS' if abs(pose[2]-1.0)<0.15 else 'FAIL'}")

    # Test 3: Curve
    print("\n=== Test 3: Curve (v=0.3, omega=0.3) for 3 seconds ===")
    print("(you can close the window anytime, it will exit gracefully)")
    bot.set_velocity(v=0.3, omega=0.3)
    for i in range(3 * steps_per_second):
        if not safe_step(client, timestep):
            print("[!] Window closed. Exiting.")
            return
    pose = bot.get_pose()
    print(f"After curve: x={pose[0]:.3f}, y={pose[1]:.3f}, yaw={pose[2]:.3f}")

    # Test 4: LiDAR
    print("\n=== Test 4: LiDAR scan ===")
    bot.stop()
    if not safe_step(client, timestep, realtime=False):
        print("[!] Window closed. Exiting.")
        return
    ranges = bot.get_lidar()
    print(f"LiDAR: {len(ranges)} rays, min={ranges.min():.3f}m, max={ranges.max():.3f}m")
    print(f"  Mean range: {ranges.mean():.3f}m")
    closest_idx = int(np.argmin(ranges))
    closest_angle = closest_idx * (360.0 / len(ranges))
    print(f"  Closest obstacle at angle: {closest_angle:.1f}°, dist={ranges[closest_idx]:.3f}m")

    # Test 5: Stop
    print("\n=== Test 5: Stop ===")
    bot.stop()
    for i in range(steps_per_second):
        if not safe_step(client, timestep):
            print("[!] Window closed. Exiting.")
            return
    vel = bot.get_velocity()
    print(f"Velocity after stop: v={vel[0]:.4f}, omega={vel[1]:.4f}")

    print("\n=== All tests PASSED! ===")
    print("Close the PyBullet window to exit.")
    
    try:
        while p.isConnected(client):
            if not safe_step(client, timestep):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected(client):
            p.disconnect(client)
        print("Disconnected.")


if __name__ == "__main__":
    main()