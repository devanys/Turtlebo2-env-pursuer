"""
Main entry point for Pursuit-Evasion simulation.

Usage:
    python main.py                         # Greedy policy (default)
    python main.py --policy hjb            # HJB policy
    python main.py --seed 42               # Fixed seed
    python main.py --episodes 5            # Multiple episodes
    python main.py --record                # Save video
    python main.py --fast                  # No real-time delay
"""
import sys
import os
import time
import argparse
import yaml
import numpy as np
import pybullet as p
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.pe_env import PursuitEvasionEnv
from agents.pursuer_greedy import GreedyPursuer
from agents.evader_flee import FleeingEvader
from agents.hjb_pursuer import HJBPursuer
from agents.hjb_evader import HJBEvader
from algorithms.hjb_solver import HJBSolver
from viz.god_view import GodView


def configure_pybullet_gui(client):
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=client)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=client)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=client)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0, physicsClientId=client)
    except:
        pass
    try:
        p.resetDebugVisualizerCamera(
            cameraDistance=15.0, cameraYaw=0, cameraPitch=-60,
            cameraTargetPosition=[0, 0, 0], physicsClientId=client)
    except:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Pursuit-Evasion Multi-Robot Simulation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--policy", choices=["greedy", "hjb"], default="greedy",
                        help="Policy: greedy (default) or hjb")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Disable overlay mode (God View normal size)")
    parser.add_argument("--god-only", action="store_true",
                        help="Only God View window (PyBullet headless)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--video-path", default="episode.mp4")
    parser.add_argument("--realtime", action="store_true", default=True)
    parser.add_argument("--fast", dest="realtime", action="store_false")
    return parser.parse_args()


def create_agents(config, num_pursuers, num_evaders, policy, hjb_solver=None):
    """Create agents based on policy selection."""
    if policy == "hjb":
        print(f"[MAIN] Creating HJB agents...")
        pursuer_agents = []
        for i in range(num_pursuers):
            pursuer_agents.append(HJBPursuer(
                name=f"pursuer_{i}",
                lidar_range=config["lidar"]["range_max"],
                target_idx=i % num_evaders,
                hjb_solver=hjb_solver,
            ))
        evader_agents = []
        for i in range(num_evaders):
            evader_agents.append(HJBEvader(
                name=f"evader_{i}",
                lidar_range=config["lidar"]["range_max"],
                hjb_solver=hjb_solver,
            ))
    else:
        print(f"[MAIN] Creating Greedy agents...")
        pursuer_agents = []
        for i in range(num_pursuers):
            pursuer_agents.append(GreedyPursuer(
                name=f"pursuer_{i}",
                lidar_range=config["lidar"]["range_max"],
                target_idx=i % num_evaders,
            ))
        evader_agents = []
        for i in range(num_evaders):
            evader_agents.append(FleeingEvader(
                name=f"evader_{i}",
                lidar_range=config["lidar"]["range_max"],
            ))
    return pursuer_agents, evader_agents


def run_episode(env, pursuer_agents, evader_agents, god_view,
                seed=42, realtime=True, max_time=None, policy="greedy", hjb_solver=None):
    """Run a single episode with paper data tracking."""
    obs = env.reset(seed=seed)
    
    if env.mode == "gui":
        configure_pybullet_gui(env.client)
    
    # Reset agent states
    for agent in pursuer_agents:
        if hasattr(agent, 'reset_shared_state'):
            agent.reset_shared_state()
    for agent in evader_agents:
        pass  # evaders don't have shared state
    
    # Reset god view
    if god_view is not None:
        god_view.pursuer_trails = [deque(maxlen=god_view.trail_length)
                                    for _ in range(god_view.num_pursuers)]
        god_view.evader_trails = [deque(maxlen=god_view.trail_length)
                                   for _ in range(god_view.num_evaders)]
        god_view.capture_markers = []
        god_view.obstacles = env.arena.get_obstacle_list()
    
    # Paper data tracking
    capture_log = []  # list of {evader, time, predicted, error}
    ttc_history = {i: deque(maxlen=100) for i in range(env.num_evaders)}  # TTC predictions per evader
    captured_set = set()
    
    total_rewards = {"pursuers": 0.0, "evaders": 0.0}
    step = 0
    done = False
    info = {}
    max_t = max_time or env.max_episode_time
    
    print(f"\n--- Episode starting (seed={seed}, policy={policy}) ---")
    print(f"Pursuers: {env.num_pursuers}")
    print(f"Evaders : {env.num_evaders}")
    print(f"Max time: {max_t}s")
    print("-" * 60)
    
    try:
        while not done:
            if not p.isConnected(env.client):
                print("\n[!] PyBullet disconnected. Exiting.")
                break
            
            if god_view is not None:
                try:
                    import matplotlib.pyplot as plt
                    if not plt.get_fignums():
                        print("\n[!] God view window closed. Exiting.")
                        break
                except:
                    pass
            
            # Compute actions
            pursuer_actions = [pursuer_agents[i].policy(obs, i) for i in range(env.num_pursuers)]
            evader_actions = [evader_agents[i].policy(obs, i) for i in range(env.num_evaders)]
            actions = {"pursuers": pursuer_actions, "evaders": evader_actions}
            
            try:
                obs, rewards, done, info = env.step(actions)
            except (p.error, TypeError):
                print("\n[!] Simulation interrupted. Exiting.")
                break
            
            total_rewards["pursuers"] += rewards["pursuers"].sum()
            total_rewards["evaders"] += rewards["evaders"].sum()
            step += 1
            info["max_time"] = max_t
            
            # Track TTC predictions for paper data
            for ev_idx in range(env.num_evaders):
                if obs["evader_alive"][ev_idx] and hasattr(evader_agents[ev_idx], 'current_ttc'):
                    ttc_history[ev_idx].append((step, evader_agents[ev_idx].current_ttc))
            
            # Check for new captures
            for cap_step, pur_idx, ev_idx in info.get("capture_events", []):
                if ev_idx not in captured_set:
                    captured_set.add(ev_idx)
                    cap_time = info['time']
                    
                    # Find prediction from ~30 steps (3s) ago
                    predicted = None
                    error = None
                    hist = ttc_history[ev_idx]
                    if len(hist) >= 30:
                        old_step, old_ttc = hist[-30]
                        # At that step, predicted TTC was old_ttc
                        # Actual time from then to capture = cap_time - (old_step / 10.0)
                        time_elapsed = cap_time - (old_step * 0.1)
                        # If prediction was accurate, old_ttc ≈ time_elapsed
                        predicted = old_ttc
                        error = abs(old_ttc - time_elapsed)
                    
                    capture_log.append({
                        'evader': ev_idx,
                        'time': cap_time,
                        'predicted': predicted,
                        'error': error,
                    })
                    print(f"[PAPER] E{ev_idx} caught at t={cap_time:.1f}s, "
                          f"pred={predicted:.1f}s, err={error:.2f}s" if predicted else
                          f"[PAPER] E{ev_idx} caught at t={cap_time:.1f}s (no prediction)")
            
            # Build paper stats
            active_ttc = []
            for ev_idx in range(env.num_evaders):
                if obs["evader_alive"][ev_idx] and hasattr(evader_agents[ev_idx], 'current_ttc'):
                    active_ttc.append(evader_agents[ev_idx].current_ttc)
                else:
                    active_ttc.append(None)
            
            # Compute avg prediction error
            errors = [c['error'] for c in capture_log if c['error'] is not None]
            avg_error = np.mean(errors) if errors else 0.0
            
            paper_stats = {
                'policy': policy.upper(),
                'time': info['time'],
                'max_time': max_t,
                'captures': info['num_caught'],
                'total_evaders': env.num_evaders,
                'brt_coverage': hjb_solver.get_brt_coverage() if hjb_solver else 0.0,
                'brt_radius': hjb_solver.brt_radius if hjb_solver else 0.0,
                'avg_pred_error': avg_error,
                'collisions': info.get('collision_counts', np.zeros(1)).sum(),
                'capture_log': capture_log,
                'active_ttc': active_ttc,
            }
            
            # Update god view
            if god_view is not None:
                god_view.update(obs, info, paper_stats)
            
            # Print status every 10 steps
            if step % 10 == 0 or done:
                brt_str = f"BRT:{paper_stats['brt_coverage']:.0f}%" if policy == "hjb" else ""
                print(f"  [t={info['time']:5.1f}s] "
                      f"alive={info['num_remaining']}/{env.num_evaders}, "
                      f"caught={info['num_caught']}, "
                      f"{brt_str} "
                      f"coll={info.get('current_collisions', 0)}")
            
            if realtime:
                time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    
    # Episode summary
    print("\n" + "=" * 60)
    print(f" Episode ended at t={info.get('time', 0):.1f}s ({step} steps)")
    print("=" * 60)
    print(f"Policy         : {policy.upper()}")
    print(f"Pursuer winner : {info.get('pursuer_winner', False)}")
    print(f"Evader winner  : {info.get('evader_winner', False)}")
    print(f"Total captures : {len(capture_log)}")
    
    if capture_log:
        print("\nCapture Log (for paper):")
        for c in capture_log:
            if c['predicted'] is not None:
                print(f"  E{c['evader']}: caught at t={c['time']:.2f}s, "
                      f"predicted TTC={c['predicted']:.2f}s, error={c['error']:.3f}s")
            else:
                print(f"  E{c['evader']}: caught at t={c['time']:.2f}s (no prediction)")
        
        errors = [c['error'] for c in capture_log if c['error'] is not None]
        if errors:
            print(f"\n  Avg prediction error: {np.mean(errors):.3f}s")
            print(f"  Max prediction error: {max(errors):.3f}s")
            print(f"  Min prediction error: {min(errors):.3f}s")
    
    if policy == "hjb" and hjb_solver:
        print(f"\n  BRT coverage: {hjb_solver.get_brt_coverage():.1f}%")
        print(f"  BRT radius: {hjb_solver.brt_radius:.2f}m")
    
    print(f"\n  Total collisions: {info.get('collision_counts', np.zeros(1)).sum()}")
    print(f"  Total rewards: pursuers={total_rewards['pursuers']:.2f}, "
          f"evaders={total_rewards['evaders']:.2f}")
    
    return {
        "time": info.get('time', 0),
        "steps": step,
        "captures": len(capture_log),
        "capture_log": capture_log,
        "pursuer_winner": info.get('pursuer_winner', False),
        "evader_winner": info.get('evader_winner', False),
        "total_collisions": info.get('collision_counts', np.zeros(1)).sum(),
        "avg_pred_error": np.mean([c['error'] for c in capture_log if c['error'] is not None]) if capture_log else 0,
        "rewards_pursuers": total_rewards['pursuers'],
        "rewards_evaders": total_rewards['evaders'],
    }


def main():
    args = parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Determine mode
    if args.god_only:
        mode = "direct"
        overlay_mode = False
    elif args.no_overlay:
        mode = "gui"
        overlay_mode = False
    else:
        mode = "gui"
        overlay_mode = True
    
    print("=" * 60)
    print(" Multi-Robot Pursuit-Evasion Simulation")
    print("=" * 60)
    print(f"Config     : {args.config}")
    print(f"Policy     : {args.policy.upper()}")
    print(f"PyBullet   : {'HEADLESS' if mode == 'direct' else 'GUI (dominant)'}")
    print(f"God View   : {'Overlay' if overlay_mode else 'Normal'}")
    print(f"Seed       : {args.seed}")
    print(f"Episodes   : {args.episodes}")
    print("=" * 60)
    
    # === HJB PRECOMPUTE (if hjb policy) ===
    hjb_solver = None
    if args.policy == "hjb":
        print("\n[HJB] Initializing HJB Solver...")
        hjb_solver = HJBSolver(
            arena_size=config["arena"]["size"][0],
            v_pursuer=1.2,  # chase speed
            v_evader=0.35,  # flee speed
            capture_radius=config["game"]["capture_radius"],
            grid_size=80,
            time_horizon=5.0,
            dt=0.1,
        )
        hjb_solver.precompute(verbose=True)
        print()
    
    # Create env
    env = PursuitEvasionEnv(config, mode=mode)
    if mode == "gui":
        configure_pybullet_gui(env.client)
    
    # Create agents
    pursuer_agents, evader_agents = create_agents(
        config, env.num_pursuers, env.num_evaders, args.policy, hjb_solver)
    
    # Create god view
    god_view = GodView(
        arena_size=config["arena"]["size"],
        obstacles=[],
        num_pursuers=config["robots"]["num_pursuers"],
        num_evaders=config["robots"]["num_evaders"],
        save_video=args.record,
        video_path=args.video_path,
        overlay_mode=overlay_mode,
        hjb_solver=hjb_solver,
        policy_name=args.policy.upper(),
    )
    
    # Run episodes
    all_stats = []
    try:
        for ep in range(args.episodes):
            print(f"\n{'#' * 60}")
            print(f"# Episode {ep + 1} / {args.episodes}")
            print(f"{'#' * 60}")
            
            seed = args.seed + ep
            stats = run_episode(
                env, pursuer_agents, evader_agents, god_view,
                seed=seed, realtime=args.realtime,
                max_time=config["simulation"]["max_episode_time"],
                policy=args.policy, hjb_solver=hjb_solver,
            )
            all_stats.append(stats)
    
    finally:
        if god_view is not None:
            god_view.close()
        env.close()
        print("\nEnvironment closed.")
    
    # Multi-episode summary
    if args.episodes > 1:
        print("\n" + "=" * 60)
        print(" Multi-Episode Summary (for Paper)")
        print("=" * 60)
        print(f"{'Ep':<4} {'Time':<8} {'Caps':<6} {'Winner':<12} {'AvgErr':<8} {'Colls':<6}")
        print("-" * 60)
        for i, s in enumerate(all_stats):
            winner = "Pursuers" if s['pursuer_winner'] else ("Evaders" if s['evader_winner'] else "None")
            print(f"{i+1:<4} {s['time']:<8.1f} {s['captures']:<6} "
                  f"{winner:<12} {s.get('avg_pred_error', 0):<8.3f} {s['total_collisions']:<6}")
        
        avg_time = np.mean([s['time'] for s in all_stats])
        avg_caps = np.mean([s['captures'] for s in all_stats])
        avg_err = np.mean([s.get('avg_pred_error', 0) for s in all_stats])
        pursuer_wins = sum(1 for s in all_stats if s['pursuer_winner'])
        print("-" * 60)
        print(f"Avg time: {avg_time:.1f}s, Avg captures: {avg_caps:.1f}")
        print(f"Avg prediction error: {avg_err:.3f}s")
        print(f"Pursuer wins: {pursuer_wins}/{args.episodes} ({100*pursuer_wins/args.episodes:.0f}%)")


if __name__ == "__main__":
    main()