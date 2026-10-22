"""
Test script for PursuitEvasionEnv with proper agents.
Spawns 7 pursuers + 3 evaders, runs GreedyPursuer vs FleeingEvader.
"""
import sys
import os
import time
import yaml
import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.pe_env import PursuitEvasionEnv
from agents.pursuer_greedy import GreedyPursuer
from agents.evader_flee import FleeingEvader


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print(" Pursuit-Evasion with Proper Agents")
    print("=" * 60)
    
    env = PursuitEvasionEnv(config, mode="gui")
    
    # Create agents (one policy object per robot, stateless)
    pursuer_agents = [
        GreedyPursuer(
            name=f"pursuer_{i}",
            lidar_range=config["lidar"]["range_max"],
            max_linear_vel=config["robot_params"]["max_linear_vel"],
            max_angular_vel=config["robot_params"]["max_angular_vel"],
        )
        for i in range(env.num_pursuers)
    ]
    evader_agents = [
        FleeingEvader(
            name=f"evader_{i}",
            lidar_range=config["lidar"]["range_max"],
            max_linear_vel=config["robot_params"]["max_linear_vel"],
            max_angular_vel=config["robot_params"]["max_angular_vel"],
        )
        for i in range(env.num_evaders)
    ]
    
    print(f"\nSpawning {env.num_pursuers} pursuers + {env.num_evaders} evaders...")
    obs = env.reset(seed=42)
    
    print(f"\nAgents:")
    print(f"  Pursuers: GreedyPursuer (chase nearest evader + obstacle avoidance)")
    print(f"  Evaders : FleeingEvader (flee from nearest pursuer + corner escape)")
    
    print(f"\nRunning episode (max time = {env.max_episode_time}s)...")
    print("Close PyBullet window anytime to exit gracefully.")
    print("=" * 60)
    
    total_rewards = {"pursuers": 0.0, "evaders": 0.0}
    step = 0
    done = False
    info = {}
    
    try:
        while not done:
            if not p.isConnected(env.client):
                print("\n[!] Window closed. Exiting gracefully.")
                return
            
            # Compute actions using agents
            pursuer_actions = [
                pursuer_agents[i].policy(obs, i)
                for i in range(env.num_pursuers)
            ]
            evader_actions = [
                evader_agents[i].policy(obs, i)
                for i in range(env.num_evaders)
            ]
            actions = {"pursuers": pursuer_actions, "evaders": evader_actions}
            
            try:
                obs, rewards, done, info = env.step(actions)
            except (p.error, TypeError) as e:
                print(f"\n[!] Window closed mid-step ({type(e).__name__}). Exiting gracefully.")
                return
            
            total_rewards["pursuers"] += rewards["pursuers"].sum()
            total_rewards["evaders"] += rewards["evaders"].sum()
            step += 1
            
            # Print status every 10 control steps (= 1 second)
            if step % 10 == 0 or done:
                print(f"  [t={info['time']:5.1f}s] "
                      f"alive={info['num_remaining']}/{env.num_evaders}, "
                      f"caught={info['num_caught']}, "
                      f"cur_coll={info.get('current_collisions', 0)}, "
                      f"total_events={info['collision_counts'].sum()}")
            
            time.sleep(0.05)
        
        # Episode end
        print("\n" + "=" * 60)
        print(f" Episode ended at t={info['time']:.1f}s ({step} control steps)")
        print("=" * 60)
        print(f"Pursuer winner : {info['pursuer_winner']}")
        print(f"Evader winner  : {info['evader_winner']}")
        print(f"Total captures : {len(info['capture_events'])}")
        for cap_step, pur_idx, ev_idx in info["capture_events"]:
            print(f"  step={cap_step}: pursuer_{pur_idx} caught evader_{ev_idx}")
        print(f"Total collisions: {info['collision_counts'].sum()}")
        print(f"Total rewards: pursuers={total_rewards['pursuers']:.2f}, "
              f"evaders={total_rewards['evaders']:.2f}")
        
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()