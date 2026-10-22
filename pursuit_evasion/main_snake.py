"""
Main entry point for Capture Pursuit-Evasion simulation.

Usage:
    python main_snake.py                    # default, 10 runs with recording
    python main_snake.py --runs 5           # custom number of runs
    python main_snake.py --no-record        # disable recording
    python main_snake.py --no-overlay       # God View normal size
    python main_snake.py --no-paper         # disable Paper Data Panel
    python main_snake.py --seed 42
    python main_snake.py --fast             # no real-time delay
"""
import sys
import os
import time
import argparse
import yaml
import csv
import numpy as np
import pybullet as p
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.snake_env import SnakeEnv
from agents.snake_agents import SnakeHead, SnakeEvader
from viz.snake_god_view import SnakeGodView
from viz.paper_data_panel import PaperDataPanel


def check_imageio():
    try:
        import imageio
        return True
    except ImportError:
        print("[WARN] imageio not installed. Recording will be disabled.")
        print("       Install with: pip install imageio imageio-ffmpeg")
        return False


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
            cameraDistance=20.0, cameraYaw=0, cameraPitch=-60,
            cameraTargetPosition=[0, 0, 0], physicsClientId=client)
    except:
        pass


def capture_pybullet_frame(client, width=640, height=480):
    try:
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0, 0, 20],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 1, 0],
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=50, aspect=width/height, nearVal=0.1, farVal=100
        )
        _, _, px, _, _ = p.getCameraImage(
            width, height, view_matrix, proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=client
        )
        img = np.array(px, dtype=np.uint8)
        img = np.reshape(img, (height, width, 4))
        img = img[:, :, :3]
        return img
    except:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Capture Pursuit-Evasion Simulation")
    parser.add_argument("--config", default="configs/snake_config.yaml")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of runs (default: 10)")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--no-paper", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--realtime", action="store_true", default=True)
    parser.add_argument("--fast", dest="realtime", action="store_false")
    return parser.parse_args()


def run_single_episode(env, head_agent, evader_agents, god_view, paper_panel,
                       seed, can_record, capture_pb, run_dir, run_idx):
    """Run a single episode. Returns stats dict."""
    
    obs = env.reset(seed=seed)
    
    head_agent.wander_heading = np.random.uniform(0, 2 * np.pi)
    head_agent.last_pos = None
    head_agent.stuck_counter = 0
    head_agent.cooldown = 0
    
    if god_view is not None:
        from collections import deque
        god_view.head_trail.clear()
        god_view.capture_markers = []
        god_view.obstacles = env.arena.get_obstacle_list()
    
    if paper_panel is not None:
        paper_panel.time_hist.clear()
        paper_panel.capture_length_hist.clear()
        paper_panel.alive_hist.clear()
        paper_panel.head_speed_hist.clear()
        paper_panel.head_omega_hist.clear()
        paper_panel.nearest_dist_hist.clear()
        paper_panel.capture_events = []
    
    pb_frames = []
    step = 0
    done = False
    info = {}
    
    print(f"\n[Run {run_idx}] Starting (seed={seed})...")
    
    try:
        while not done:
            if not p.isConnected(env.client):
                print(f"\n[Run {run_idx}] PyBullet disconnected.")
                break
            
            try:
                import matplotlib.pyplot as plt
                if not plt.get_fignums():
                    print(f"\n[Run {run_idx}] Matplotlib windows closed.")
                    break
            except:
                pass
            
            prev_capture_count = len(info.get('capture_events', [])) if step > 0 else 0
            
            head_action = head_agent.policy(obs, 0)
            evader_actions = [evader_agents[i].policy(obs, i) for i in range(env.num_evaders)]
            actions = {"head": head_action, "evaders": evader_actions}
            
            try:
                obs, rewards, done, info = env.step(actions)
            except (p.error, TypeError) as e:
                print(f"\n[Run {run_idx}] Simulation interrupted: {e}")
                break
            
            step += 1
            
            new_capture_count = len(info.get('capture_events', []))
            if new_capture_count > prev_capture_count:
                new_head_yaw = obs["head_pose"][2]
                head_agent.on_capture(new_head_yaw)
            
            if god_view is not None:
                god_view.update(obs, info)
            
            if paper_panel is not None:
                head_vel = env.pursuers[0].get_velocity()
                paper_panel.update(obs, info, head_velocity=head_vel)
            
            if can_record and capture_pb:
                pb_frame = capture_pybullet_frame(env.client)
                if pb_frame is not None:
                    pb_frames.append(pb_frame)
            
            if step % 50 == 0 or done:
                print(f"  [Run {run_idx}] t={info['time']:5.1f}s "
                      f"capture={info['snake_length']} "
                      f"alive={info['num_remaining']}/{env.num_evaders}")
            
            if god_view is not None and god_view.realtime:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print(f"\n[Run {run_idx}] Interrupted by user.")
    
    # === SAVE RECORDINGS ===
    pb_path = None
    gv_path = None
    paper_path = None
    
    if can_record and run_dir:
        try:
            import imageio
            
            if pb_frames and capture_pb:
                pb_path = os.path.join(run_dir, f"run_{run_idx:02d}_pybullet.mp4")
                print(f"[Run {run_idx}] Saving PyBullet video ({len(pb_frames)} frames)...")
                imageio.mimsave(pb_path, pb_frames, fps=10)
            
            if god_view is not None and god_view.video_frames:
                gv_path = os.path.join(run_dir, f"run_{run_idx:02d}_godview.mp4")
                print(f"[Run {run_idx}] Saving God View video ({len(god_view.video_frames)} frames)...")
                imageio.mimsave(gv_path, god_view.video_frames, fps=10)
                god_view.video_frames = []
            
            if paper_panel is not None:
                paper_path = os.path.join(run_dir, f"run_{run_idx:02d}_paper.png")
                print(f"[Run {run_idx}] Saving Paper Panel PNG...")
                paper_panel.fig.savefig(paper_path, dpi=150, bbox_inches='tight')
        except Exception as e:
            print(f"[Run {run_idx}] Error saving recordings: {e}")
    
    # === BUILD STATS ===
    cap_events = info.get('capture_events', [])
    cap_times = [c[0] * 0.1 for c in cap_events]
    intervals = [cap_times[i+1] - cap_times[i] for i in range(len(cap_times)-1)] if len(cap_times) > 1 else []
    
    stats = {
        'run': run_idx,
        'seed': seed,
        'duration': info.get('time', 0),
        'steps': step,
        'final_capture_length': info.get('snake_length', 1),
        'total_captures': info.get('num_caught', 0),
        'remaining': info.get('num_remaining', 0),
        'winner': 'HEAD' if info.get('num_remaining', 0) == 0 else 'TIMEOUT',
        'capture_times': cap_times,
        'intervals': intervals,
        'mean_interval': float(np.mean(intervals)) if intervals else 0,
        'std_interval': float(np.std(intervals)) if intervals else 0,
        'min_interval': float(min(intervals)) if intervals else 0,
        'max_interval': float(max(intervals)) if intervals else 0,
        'pb_video': pb_path,
        'gv_video': gv_path,
        'paper_png': paper_path,
    }
    
    print(f"[Run {run_idx}] DONE — captures={stats['total_captures']}, "
          f"duration={stats['duration']:.1f}s, winner={stats['winner']}")
    
    return stats


def save_summary_csv(all_stats, csv_path):
    """Save all run stats to CSV."""
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'run', 'seed', 'duration_s', 'steps', 'final_capture_length',
                'total_captures', 'remaining', 'winner',
                'mean_interval_s', 'std_interval_s', 'min_interval_s', 'max_interval_s',
                'pb_video', 'gv_video', 'paper_png'
            ])
            for s in all_stats:
                writer.writerow([
                    s['run'], s['seed'], f"{s['duration']:.2f}", s['steps'],
                    s['final_capture_length'], s['total_captures'], s['remaining'],
                    s['winner'],
                    f"{s['mean_interval']:.2f}", f"{s['std_interval']:.2f}",
                    f"{s['min_interval']:.2f}", f"{s['max_interval']:.2f}",
                    os.path.basename(s['pb_video']) if s['pb_video'] else '',
                    os.path.basename(s['gv_video']) if s['gv_video'] else '',
                    os.path.basename(s['paper_png']) if s['paper_png'] else '',
                ])
        print(f"\n[SUMMARY] CSV saved: {os.path.abspath(csv_path)}")
    except Exception as e:
        print(f"[SUMMARY] Error saving CSV: {e}")


def print_final_summary(all_stats):
    """Print summary of all runs."""
    print("\n" + "=" * 70)
    print(" FINAL SUMMARY — All Runs")
    print("=" * 70)
    print(f"{'Run':<5} {'Seed':<6} {'Time':<8} {'Caps':<6} {'Remain':<7} {'Winner':<10} {'MeanInt':<10}")
    print("-" * 70)
    
    for s in all_stats:
        print(f"{s['run']:<5} {s['seed']:<6} {s['duration']:<8.1f} "
              f"{s['total_captures']:<6} {s['remaining']:<7} "
              f"{s['winner']:<10} {s['mean_interval']:<10.2f}")
    
    print("-" * 70)
    
    durations = [s['duration'] for s in all_stats]
    captures = [s['total_captures'] for s in all_stats]
    head_wins = sum(1 for s in all_stats if s['winner'] == 'HEAD')
    all_intervals = [iv for s in all_stats for iv in s['intervals']]
    
    print(f"Total runs       : {len(all_stats)}")
    print(f"Head wins        : {head_wins}/{len(all_stats)} ({100*head_wins/len(all_stats):.0f}%)")
    print(f"Avg duration     : {np.mean(durations):.1f}s")
    print(f"Avg captures     : {np.mean(captures):.1f}")
    if all_intervals:
        print(f"Avg capture interval: {np.mean(all_intervals):.2f}s")
        print(f"Std capture interval: {np.std(all_intervals):.2f}s")
    print("=" * 70)


def main():
    args = parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    record_enabled = not args.no_record
    can_record = False
    if record_enabled:
        can_record = check_imageio()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    run_dir = os.path.join(data_dir, f"session_{timestamp}") if can_record else None
    
    if can_record:
        os.makedirs(run_dir, exist_ok=True)
    
    if args.no_overlay:
        overlay_mode = False
        mode = "gui"
    else:
        overlay_mode = True
        mode = "gui"
    
    print("=" * 70)
    print(f" Capture Pursuit-Evasion Simulation — {args.runs} RUNS")
    print("=" * 70)
    print(f"Config       : {args.config}")
    print(f"Number of runs: {args.runs}")
    print(f"PyBullet     : {mode}")
    print(f"God View     : {'Overlay' if overlay_mode else 'Normal'}")
    print(f"Paper Panel  : {'OFF' if args.no_paper else 'ON'}")
    print(f"Recording    : {'ON -> ' + run_dir if can_record else 'OFF'}")
    print(f"Base seed    : {args.seed}")
    print("=" * 70)
    
    env = SnakeEnv(config, mode=mode)
    if mode == "gui":
        configure_pybullet_gui(env.client)
    
    head_agent = SnakeHead(
        name="head",
        lidar_range=config["lidar"]["range_max"],
        max_linear_vel=config["robot_params"]["max_linear_vel"],
        max_angular_vel=config["robot_params"]["max_angular_vel"],
        detection_radius=config["game"]["detection_radius"],
        arena_size=config["arena"]["size"][0],
        approach_distance=config["game"]["capture_radius"],
    )
    head_agent.realtime = args.realtime
    
    evader_agents = []
    for i in range(env.num_evaders):
        evader_agents.append(SnakeEvader(
            name=f"evader_{i}",
            lidar_range=config["lidar"]["range_max"],
            max_linear_vel=0.5,
            max_angular_vel=1.5,
            arena_size=config["arena"]["size"][0],
        ))
    
    god_view = SnakeGodView(
        arena_size=config["arena"]["size"],
        num_evaders=config["robots"]["num_evaders"],
        detection_radius=config["game"]["detection_radius"],
        save_video=can_record,
        video_path=os.path.join(run_dir, "temp.mp4") if run_dir else "temp.mp4",
        overlay_mode=overlay_mode,
    )
    god_view.realtime = args.realtime
    
    paper_panel = None
    if not args.no_paper:
        paper_panel = PaperDataPanel(
            num_evaders=config["robots"]["num_evaders"],
            max_time=config["simulation"]["max_episode_time"],
        )
    
    all_stats = []
    
    try:
        for run_idx in range(1, args.runs + 1):
            seed = args.seed + (run_idx - 1) * 100
            
            print(f"\n{'#' * 70}")
            print(f"# RUN {run_idx} / {args.runs}  (seed={seed})")
            print(f"{'#' * 70}")
            
            stats = run_single_episode(
                env=env,
                head_agent=head_agent,
                evader_agents=evader_agents,
                god_view=god_view,
                paper_panel=paper_panel,
                seed=seed,
                can_record=can_record,
                capture_pb=(mode == "gui"),
                run_dir=run_dir,
                run_idx=run_idx,
            )
            
            all_stats.append(stats)
            
            print(f"\n[Run {run_idx}] Completed. "
                  f"{args.runs - run_idx} runs remaining.")
            
            if run_idx < args.runs:
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user. Saving what we have...")
    
    print_final_summary(all_stats)
    
    if can_record and run_dir:
        csv_path = os.path.join(run_dir, "summary.csv")
        save_summary_csv(all_stats, csv_path)
        
        print(f"\n[OUTPUT] All recordings saved to: {os.path.abspath(run_dir)}")
        print(f"[OUTPUT] Files in folder:")
        try:
            files = sorted(os.listdir(run_dir))
            for f in files:
                print(f"  - {f}")
        except:
            pass
    
    try:
        god_view.close()
    except:
        pass
    if paper_panel is not None:
        try:
            paper_panel.close()
        except:
            pass
    env.close()
    print("\nAll done. Environment closed.")


if __name__ == "__main__":
    main()