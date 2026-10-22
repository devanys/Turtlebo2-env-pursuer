"""
God View visualizer with BRT heatmap and Paper Data Panel.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from collections import deque


class GodView:
    PURSUER_COLOR = '#2196F3'
    PURSUER_EDGE = '#0D47A1'
    EVADER_COLOR = '#F44336'
    EVADER_EDGE = '#B71C1C'
    EVADER_CAUGHT_COLOR = '#9E9E9E'
    WALL_COLOR = '#424242'
    OBSTACLE_COLOR = '#795548'
    TRAIL_PURSUER = '#64B5F6'
    TRAIL_EVADER = '#EF9A9A'
    CAPTURE_RING = '#FFC107'
    CAPTURE_X = '#D50000'
    BRT_COLOR = '#FF1744'
    BG_COLOR = '#FAFAFA'
    
    def __init__(self, arena_size, obstacles=None, num_pursuers=7, num_evaders=3,
                 trail_length=50, save_video=False, video_path="episode.mp4",
                 overlay_mode=False, hjb_solver=None, policy_name="Greedy"):
        self.arena_size = np.array(arena_size)
        self.obstacles = obstacles or []
        self.num_pursuers = num_pursuers
        self.num_evaders = num_evaders
        self.trail_length = trail_length
        self.save_video = save_video
        self.video_path = video_path
        self.overlay_mode = overlay_mode
        self.hjb_solver = hjb_solver
        self.policy_name = policy_name
        
        self.pursuer_trails = [deque(maxlen=trail_length) for _ in range(num_pursuers)]
        self.evader_trails = [deque(maxlen=trail_length) for _ in range(num_evaders)]
        self.capture_markers = []
        
        plt.ion()
        if overlay_mode:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            self.fig.canvas.manager.set_window_title('God View (Overlay)')
        else:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            self.fig.canvas.manager.set_window_title('Pursuit-Evasion God View')
        
        try:
            if overlay_mode:
                self.fig.canvas.manager.window.wm_geometry("+10+10")
                self.fig.canvas.manager.window.attributes('-topmost', True)
            else:
                self.fig.canvas.manager.window.wm_geometry("+900+50")
        except:
            pass
        
        w, h = self.arena_size
        margin = 1.0
        self.ax.set_xlim(-w/2 - margin, w/2 + margin)
        self.ax.set_ylim(-h/2 - margin, h/2 + margin)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor(self.BG_COLOR)
        
        if overlay_mode:
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_title(f'{policy_name} Pursuit-Evasion', fontsize=9, fontweight='bold', pad=4)
            self.ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        else:
            self.ax.set_title(f'{policy_name} Pursuit-Evasion God View', fontsize=14, fontweight='bold')
            self.ax.grid(True, alpha=0.3, linestyle='--')
        
        self._draw_static()
        
        # BRT disks (translucent red around each pursuer)
        self.brt_disks = []
        if hjb_solver is not None and hjb_solver.brt_radius > 0:
            for i in range(num_pursuers):
                disk = plt.Circle((0, 0), hjb_solver.brt_radius,
                                  facecolor=self.BRT_COLOR, alpha=0.12,
                                  edgecolor=self.BRT_COLOR, linewidth=1, linestyle='--')
                self.ax.add_patch(disk)
                self.brt_disks.append(disk)
        
        # Pursuer dots
        self.pursuer_dots = []
        self.pursuer_rings = []
        self.pursuer_arrows = []
        for i in range(num_pursuers):
            ring = plt.Circle((0, 0), 0.3, fill=False,
                              edgecolor=self.CAPTURE_RING, linewidth=0.8, alpha=0.5)
            self.ax.add_patch(ring)
            self.pursuer_rings.append(ring)
            
            dot_size = 0.18 if overlay_mode else 0.25
            dot = plt.Circle((0, 0), dot_size, facecolor=self.PURSUER_COLOR,
                             edgecolor=self.PURSUER_EDGE, linewidth=1.2, zorder=5)
            self.ax.add_patch(dot)
            self.pursuer_dots.append(dot)
            
            arrow_lw = 1 if overlay_mode else 2
            arrow, = self.ax.plot([0, 0], [0, 0], color=self.PURSUER_EDGE,
                                  linewidth=arrow_lw, zorder=6)
            self.pursuer_arrows.append(arrow)
        
        # Evader dots
        self.evader_dots = []
        self.evader_arrows = []
        self.ttc_badges = []
        for i in range(num_evaders):
            dot_size = 0.18 if overlay_mode else 0.25
            dot = plt.Circle((0, 0), dot_size, facecolor=self.EVADER_COLOR,
                             edgecolor=self.EVADER_EDGE, linewidth=1.2, zorder=5)
            self.ax.add_patch(dot)
            self.evader_dots.append(dot)
            
            arrow_lw = 1 if overlay_mode else 2
            arrow, = self.ax.plot([0, 0], [0, 0], color=self.EVADER_EDGE,
                                  linewidth=arrow_lw, zorder=6)
            self.evader_arrows.append(arrow)
            
            # TTC badge
            font_size = 6 if overlay_mode else 8
            badge = self.ax.text(0, 0, '', fontsize=font_size, fontweight='bold',
                                color='white', ha='center', va='center', zorder=7,
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
            self.ttc_badges.append(badge)
        
        # Trails
        self.pursuer_trail_lines = []
        self.evader_trail_lines = []
        trail_lw = 0.7 if overlay_mode else 1.0
        for i in range(num_pursuers):
            line = LineCollection([], colors=self.TRAIL_PURSUER,
                                  linewidths=trail_lw, alpha=0.5)
            self.ax.add_collection(line)
            self.pursuer_trail_lines.append(line)
        for i in range(num_evaders):
            line = LineCollection([], colors=self.TRAIL_EVADER,
                                  linewidths=trail_lw, alpha=0.5)
            self.ax.add_collection(line)
            self.evader_trail_lines.append(line)
        
        # Capture markers
        capture_size = 50 if overlay_mode else 100
        self.capture_scatter = self.ax.scatter([], [], marker='x', s=capture_size,
                                                c=self.CAPTURE_X, zorder=10,
                                                linewidths=1.5)
        
        # Status text
        font_size = 7 if overlay_mode else 10
        self.status_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            fontsize=font_size, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85,
                      edgecolor='gray', linewidth=0.5)
        )
        
        # Paper data panel (below status)
        panel_font = 6 if overlay_mode else 8
        self.paper_text = self.ax.text(
            0.02, 0.75, '', transform=self.ax.transAxes,
            fontsize=panel_font, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.9,
                      edgecolor='#2E7D32', linewidth=1)
        )
        
        if not overlay_mode:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.PURSUER_COLOR,
                           markeredgecolor=self.PURSUER_EDGE, markersize=12, label='Pursuer'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.EVADER_COLOR,
                           markeredgecolor=self.EVADER_EDGE, markersize=12, label='Evader'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.BRT_COLOR,
                           markerfacecoloralt=self.BRT_COLOR, alpha=0.3, markersize=14, label='BRT'),
                plt.Line2D([0], [0], marker='x', color=self.CAPTURE_X,
                           markersize=10, label='Capture', linewidth=2),
            ]
            self.ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        self.video_frames = []
        plt.show(block=False)
        plt.pause(0.01)
    
    def _draw_static(self):
        w, h = self.arena_size
        wall_thickness = 0.2
        wall_lw = 0.5
        
        self.ax.add_patch(patches.Rectangle(
            (-w/2 - wall_thickness, -h/2 - wall_thickness),
            w + 2*wall_thickness, wall_thickness,
            facecolor=self.WALL_COLOR, edgecolor='black', linewidth=wall_lw))
        self.ax.add_patch(patches.Rectangle(
            (-w/2 - wall_thickness, h/2),
            w + 2*wall_thickness, wall_thickness,
            facecolor=self.WALL_COLOR, edgecolor='black', linewidth=wall_lw))
        self.ax.add_patch(patches.Rectangle(
            (-w/2 - wall_thickness, -h/2 - wall_thickness),
            wall_thickness, h + 2*wall_thickness,
            facecolor=self.WALL_COLOR, edgecolor='black', linewidth=wall_lw))
        self.ax.add_patch(patches.Rectangle(
            (w/2, -h/2 - wall_thickness),
            wall_thickness, h + 2*wall_thickness,
            facecolor=self.WALL_COLOR, edgecolor='black', linewidth=wall_lw))
        
        for center, half_extents in self.obstacles:
            cx, cy = center
            hw, hh = half_extents
            if hw > 5 or hh > 5:
                continue
            self.ax.add_patch(patches.Rectangle(
                (cx - hw, cy - hh), 2*hw, 2*hh,
                facecolor=self.OBSTACLE_COLOR, edgecolor='black',
                linewidth=0.5, alpha=0.8))
    
    def update(self, obs, info=None, paper_stats=None):
        # Update BRT disks
        for i, disk in enumerate(self.brt_disks):
            if i < len(obs["pursuer_poses"]):
                pose = obs["pursuer_poses"][i]
                disk.center = (pose[0], pose[1])
        
        # Update pursuers
        for i in range(self.num_pursuers):
            pose = obs["pursuer_poses"][i]
            x, y, yaw = pose
            self.pursuer_dots[i].center = (x, y)
            self.pursuer_rings[i].center = (x, y)
            arrow_len = 0.4 if self.overlay_mode else 0.5
            self.pursuer_arrows[i].set_data(
                [x, x + arrow_len * np.cos(yaw)],
                [y, y + arrow_len * np.sin(yaw)])
            self.pursuer_trails[i].append((x, y))
            if len(self.pursuer_trails[i]) >= 2:
                trail_arr = np.array(self.pursuer_trails[i])
                segments = np.stack([trail_arr[:-1], trail_arr[1:]], axis=1)
                self.pursuer_trail_lines[i].set_segments(segments)
        
        # Update evaders
        for i in range(self.num_evaders):
            if obs["evader_alive"][i]:
                pose = obs["evader_poses"][i]
                x, y, yaw = pose
                self.evader_dots[i].set_facecolor(self.EVADER_COLOR)
                self.evader_dots[i].set_edgecolor(self.EVADER_EDGE)
                self.evader_dots[i].center = (x, y)
                arrow_len = 0.4 if self.overlay_mode else 0.5
                self.evader_arrows[i].set_data(
                    [x, x + arrow_len * np.cos(yaw)],
                    [y, y + arrow_len * np.sin(yaw)])
                self.evader_trails[i].append((x, y))
                if len(self.evader_trails[i]) >= 2:
                    trail_arr = np.array(self.evader_trails[i])
                    segments = np.stack([trail_arr[:-1], trail_arr[1:]], axis=1)
                    self.evader_trail_lines[i].set_segments(segments)
                
                # TTC badge
                ttc = paper_stats.get('active_ttc', [None]*self.num_evaders)[i] if paper_stats else None
                if ttc is not None and ttc < 99:
                    self.ttc_badges[i].set_position((x, y + 0.5))
                    self.ttc_badges[i].set_text(f'{ttc:.1f}s')
                    self.ttc_badges[i].set_visible(True)
                else:
                    self.ttc_badges[i].set_visible(False)
            else:
                self.evader_dots[i].set_facecolor(self.EVADER_CAUGHT_COLOR)
                self.evader_dots[i].set_edgecolor('gray')
                self.ttc_badges[i].set_visible(False)
        
        # Capture markers
        if info and "capture_events" in info:
            for cap_step, pur_idx, ev_idx in info["capture_events"]:
                if len(self.evader_trails[ev_idx]) > 0:
                    cx, cy = self.evader_trails[ev_idx][-1]
                    if (cx, cy) not in self.capture_markers:
                        self.capture_markers.append((cx, cy))
            if self.capture_markers:
                xs, ys = zip(*self.capture_markers)
                self.capture_scatter.set_offsets(np.column_stack([xs, ys]))
        
        # Status text
        if info:
            status = (
                f"t={info['time']:.1f}s / {info.get('max_time', '?')}s\n"
                f"alive: {info['num_remaining']}/{self.num_evaders}\n"
                f"caught: {info['num_caught']}\n"
                f"coll: {info.get('current_collisions', 0)}"
            )
            self.status_text.set_text(status)
        
        # Paper data panel
        if paper_stats:
            panel = self._format_paper_stats(paper_stats)
            self.paper_text.set_text(panel)
        
        if self.save_video:
            self.video_frames.append(
                np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
                .reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
                .copy()[:, :, 1:])
        
        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except:
            pass
    
    def _format_paper_stats(self, s):
        """Format paper stats dict into display string."""
        lines = []
        lines.append("=" * 38)
        lines.append(f"  POLICY: {s.get('policy', 'Unknown')}")
        lines.append("=" * 38)
        lines.append(f"  Time:      {s.get('time', 0):.1f}s / {s.get('max_time', 60):.0f}s")
        lines.append(f"  Captures:  {s.get('captures', 0)}/{s.get('total_evaders', 3)}")
        
        if s.get('policy') == 'HJB':
            lines.append(f"  BRT cov:   {s.get('brt_coverage', 0):.1f}%")
            lines.append(f"  BRT radius:{s.get('brt_radius', 0):.2f}m")
            lines.append(f"  Avg error: {s.get('avg_pred_error', 0):.2f}s")
        
        lines.append(f"  Collisions:{s.get('collisions', 0)}")
        lines.append("-" * 38)
        lines.append("  CAPTURE LOG:")
        
        cap_log = s.get('capture_log', [])
        if cap_log:
            for c in cap_log:
                if c.get('predicted') is not None:
                    lines.append(f"  E{c['evader']}: t={c['time']:.1f}s (pred:{c['predicted']:.1f}s, err:{c['error']:.2f}s)")
                else:
                    lines.append(f"  E{c['evader']}: t={c['time']:.1f}s")
        else:
            lines.append("  (none yet)")
        
        lines.append("-" * 38)
        active = s.get('active_ttc', [])
        alive_evs = [i for i, t in enumerate(active) if t is not None and t < 99]
        if alive_evs:
            lines.append("  ACTIVE TTC:")
            for i in alive_evs:
                lines.append(f"  E{i}: TTC={active[i]:.1f}s")
        
        return "\n".join(lines)
    
    def save_video_frames(self, path=None, fps=10):
        if not self.video_frames:
            return
        path = path or self.video_path
        try:
            import imageio
            print(f"Saving {len(self.video_frames)} frames to {path}...")
            imageio.mimsave(path, self.video_frames, fps=fps)
            print(f"Video saved: {path}")
        except ImportError:
            print("imageio not installed. Install with: pip install imageio")
    
    def close(self):
        if self.save_video and self.video_frames:
            self.save_video_frames()
        plt.close(self.fig)