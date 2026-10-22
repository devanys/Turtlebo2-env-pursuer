import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque


class SnakeGodView:
    
    HEAD_COLOR = '#0D47A1'
    FOLLOWER_COLOR = '#2196F3'
    EVADER_COLOR = '#F44336'
    EVADER_EDGE = '#B71C1C'
    EVADER_CAUGHT_COLOR = '#9E9E9E'
    WALL_COLOR = '#424242'
    BRT_COLOR = '#FF1744'
    TRAIL_COLOR = '#64B5F6'
    SNAKE_LINE_COLOR = '#0D47A1'
    POV_COLOR = '#000000'
    POV_CONE_COLOR_HEAD = '#0D47A1'
    POV_CONE_COLOR_FOLLOWER = '#2196F3'
    POV_CONE_COLOR_EVADER = '#F44336'
    MAGNET_COLOR = '#FFD600'
    BG_COLOR = '#FAFAFA'
    CAPTURE_X = '#D50000'
    
    def __init__(self, arena_size, num_evaders=9, detection_radius=1.5,
                 save_video=False, video_path="snake_episode.mp4",
                 overlay_mode=False):
        self.arena_size = np.array(arena_size)
        self.num_evaders = num_evaders
        self.detection_radius = detection_radius
        self.save_video = save_video
        self.video_path = video_path
        self.overlay_mode = overlay_mode
        self.realtime = True
        
        self.head_trail = deque(maxlen=500)
        self.capture_markers = []
        
        plt.ion()
        if overlay_mode:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            self.fig.canvas.manager.set_window_title('Captured Pursuit (Overlay)')
            try:
                self.fig.canvas.manager.window.wm_geometry("+10+10")
                self.fig.canvas.manager.window.attributes('-topmost', True)
            except:
                pass
        else:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            self.fig.canvas.manager.set_window_title('Captured Pursuit God View')
            try:
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
            self.ax.set_title('Captured Pursuit', fontsize=9, fontweight='bold', pad=4)
            self.ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        else:
            self.ax.set_title('Captured Pursuit God View', fontsize=14, fontweight='bold')
            self.ax.grid(True, alpha=0.3, linestyle='--')
        
        self._draw_walls()
        
        self.snake_line, = self.ax.plot([], [], color=self.SNAKE_LINE_COLOR,
                                        linewidth=3, alpha=0.6, zorder=4,
                                        solid_capstyle='round')
        
        self.head_pov_cone = patches.Wedge((0, 0), 1.0, 0, 0,
                                           facecolor=self.POV_CONE_COLOR_HEAD,
                                           alpha=0.15, zorder=2)
        self.ax.add_patch(self.head_pov_cone)
        
        self.head_dot = plt.Circle((0, 0), 0.25, facecolor=self.HEAD_COLOR,
                                   edgecolor='black', linewidth=1.5, zorder=5)
        self.ax.add_patch(self.head_dot)
        self.head_arrow, = self.ax.plot([0, 0], [0, 0], color=self.POV_COLOR,
                                        linewidth=2.5, zorder=6)
        self.head_brt = plt.Circle((0, 0), detection_radius,
                                   facecolor=self.BRT_COLOR, alpha=0.12,
                                   edgecolor=self.BRT_COLOR, linewidth=1, linestyle='--')
        self.ax.add_patch(self.head_brt)
        
        self.follower_dots = []
        self.follower_arrows = []
        self.follower_brts = []
        self.follower_pov_cones = []
        
        self.evader_dots = []
        self.evader_arrows = []
        self.evader_pov_cones = []
        self.evader_magnets = []
        for i in range(num_evaders):
            magnet = plt.Circle((0, 0), detection_radius,
                                facecolor=self.MAGNET_COLOR, alpha=0.15,
                                edgecolor=self.MAGNET_COLOR, linewidth=1.5,
                                linestyle='--', zorder=3)
            self.ax.add_patch(magnet)
            self.evader_magnets.append(magnet)
            
            dot = plt.Circle((0, 0), 0.22, facecolor=self.EVADER_COLOR,
                             edgecolor=self.EVADER_EDGE, linewidth=1.2, zorder=5)
            self.ax.add_patch(dot)
            self.evader_dots.append(dot)
            
            arrow, = self.ax.plot([0, 0], [0, 0], color=self.POV_COLOR,
                                  linewidth=2, zorder=6)
            self.evader_arrows.append(arrow)
            
            cone = patches.Wedge((0, 0), 0.8, 0, 0,
                                 facecolor=self.POV_CONE_COLOR_EVADER,
                                 alpha=0.15, zorder=2)
            self.ax.add_patch(cone)
            self.evader_pov_cones.append(cone)
        
        self.trail_line, = self.ax.plot([], [], color=self.TRAIL_COLOR,
                                        linewidth=1, alpha=0.5, zorder=3)
        
        self.capture_scatter = self.ax.scatter([], [], marker='x', s=100,
                                                c=self.CAPTURE_X, zorder=10,
                                                linewidths=2)
        
        font_size = 7 if overlay_mode else 10
        self.status_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            fontsize=font_size, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85,
                      edgecolor='gray', linewidth=0.5)
        )
        
        if not overlay_mode:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.HEAD_COLOR,
                           markeredgecolor='black', markersize=12, label='Head'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.FOLLOWER_COLOR,
                           markeredgecolor='black', markersize=12, label='Follower'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.EVADER_COLOR,
                           markeredgecolor=self.EVADER_EDGE, markersize=12, label='Evader (static)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.MAGNET_COLOR,
                           alpha=0.3, markersize=14, label='Magnet (attraction)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.BRT_COLOR,
                           alpha=0.3, markersize=14, label='BRT (detection)'),
            ]
            self.ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        self.video_frames = []
        plt.show(block=False)
        plt.pause(0.01)
    
    def _draw_walls(self):
        w, h = self.arena_size
        wall_thickness = 0.2
        self.ax.add_patch(patches.Rectangle(
            (-w/2 - wall_thickness, -h/2 - wall_thickness),
            w + 2*wall_thickness, wall_thickness,
            facecolor=self.WALL_COLOR, edgecolor='black', linewidth=0.5))
        self.ax.add_patch(patches.Rectangle(
            (-w/2 - wall_thickness, h/2),
            w + 2*wall_thickness, wall_thickness,
            facecolor=self.WALL_COLOR, edgecolor='black', linewidth=0.5))
        self.ax.add_patch(patches.Rectangle(
            (-w/2 - wall_thickness, -h/2 - wall_thickness),
            wall_thickness, h + 2*wall_thickness,
            facecolor=self.WALL_COLOR, edgecolor='black', linewidth=0.5))
        self.ax.add_patch(patches.Rectangle(
            (w/2, -h/2 - wall_thickness),
            wall_thickness, h + 2*wall_thickness,
            facecolor=self.WALL_COLOR, edgecolor='black', linewidth=0.5))
    
    def _add_follower(self):
        dot = plt.Circle((0, 0), 0.22, facecolor=self.FOLLOWER_COLOR,
                         edgecolor='black', linewidth=1.2, zorder=5)
        self.ax.add_patch(dot)
        self.follower_dots.append(dot)
        
        arrow, = self.ax.plot([0, 0], [0, 0], color=self.POV_COLOR,
                              linewidth=2, zorder=6)
        self.follower_arrows.append(arrow)
        
        follower_brt_radius = self.detection_radius * 0.6
        brt = plt.Circle((0, 0), follower_brt_radius,
                         facecolor=self.BRT_COLOR, alpha=0.06,
                         edgecolor=self.BRT_COLOR, linewidth=0.5, linestyle=':')
        self.ax.add_patch(brt)
        self.follower_brts.append(brt)
        
        cone = patches.Wedge((0, 0), 0.8, 0, 0,
                             facecolor=self.POV_CONE_COLOR_FOLLOWER,
                             alpha=0.12, zorder=2)
        self.ax.add_patch(cone)
        self.follower_pov_cones.append(cone)
    
    def update(self, obs, info=None):
        all_poses = obs.get("all_pursuer_poses", [])
        if not all_poses:
            return
        
        head_pose = all_poses[0]
        hx, hy, hyaw = head_pose
        self.head_dot.center = (hx, hy)
        self.head_brt.center = (hx, hy)
        
        arrow_len = 0.8
        self.head_arrow.set_data(
            [hx, hx + arrow_len * np.cos(hyaw)],
            [hy, hy + arrow_len * np.sin(hyaw)])
        
        yaw_deg = np.degrees(hyaw)
        self.head_pov_cone.center = (hx, hy)
        self.head_pov_cone.set_theta1(yaw_deg - 45)
        self.head_pov_cone.set_theta2(yaw_deg + 45)
        
        self.head_trail.append((hx, hy))
        if len(self.head_trail) >= 2:
            trail_arr = np.array(self.head_trail)
            self.trail_line.set_data(trail_arr[:, 0], trail_arr[:, 1])
        
        all_x = [p[0] for p in all_poses]
        all_y = [p[1] for p in all_poses]
        self.snake_line.set_data(all_x, all_y)
        
        num_followers = len(all_poses) - 1
        while len(self.follower_dots) < num_followers:
            self._add_follower()
        
        for i in range(num_followers):
            pose = all_poses[i + 1]
            fx, fy, fyaw = pose
            self.follower_dots[i].center = (fx, fy)
            self.follower_brts[i].center = (fx, fy)
            
            arrow_len = 0.6
            self.follower_arrows[i].set_data(
                [fx, fx + arrow_len * np.cos(fyaw)],
                [fy, fy + arrow_len * np.sin(fyaw)])
            
            yaw_deg = np.degrees(fyaw)
            self.follower_pov_cones[i].center = (fx, fy)
            self.follower_pov_cones[i].set_theta1(yaw_deg - 45)
            self.follower_pov_cones[i].set_theta2(yaw_deg + 45)
        
        for i in range(num_followers, len(self.follower_dots)):
            self.follower_dots[i].set_visible(False)
            self.follower_brts[i].set_visible(False)
            self.follower_arrows[i].set_visible(False)
            self.follower_pov_cones[i].set_visible(False)
        
        evader_poses = obs["evader_poses"]
        evader_alive = obs["evader_alive"]
        for i in range(self.num_evaders):
            if evader_alive[i]:
                pose = evader_poses[i]
                ex, ey, eyaw = pose
                self.evader_dots[i].set_facecolor(self.EVADER_COLOR)
                self.evader_dots[i].set_edgecolor(self.EVADER_EDGE)
                self.evader_dots[i].center = (ex, ey)
                
                self.evader_magnets[i].center = (ex, ey)
                self.evader_magnets[i].set_visible(True)
                
                arrow_len = 0.6
                self.evader_arrows[i].set_data(
                    [ex, ex + arrow_len * np.cos(eyaw)],
                    [ey, ey + arrow_len * np.sin(eyaw)])
                
                yaw_deg = np.degrees(eyaw)
                self.evader_pov_cones[i].center = (ex, ey)
                self.evader_pov_cones[i].set_theta1(yaw_deg - 45)
                self.evader_pov_cones[i].set_theta2(yaw_deg + 45)
                
                self.evader_arrows[i].set_visible(True)
                self.evader_pov_cones[i].set_visible(True)
            else:
                self.evader_dots[i].set_facecolor(self.EVADER_CAUGHT_COLOR)
                self.evader_dots[i].set_edgecolor('gray')
                self.evader_arrows[i].set_visible(False)
                self.evader_pov_cones[i].set_visible(False)
                self.evader_magnets[i].set_visible(False)
        
        if info and "capture_events" in info:
            for cap_step, pur_idx, ev_idx in info["capture_events"]:
                if not any(c[2] == ev_idx for c in self.capture_markers):
                    if not np.isnan(evader_poses[ev_idx][0]):
                        cx, cy = evader_poses[ev_idx][0], evader_poses[ev_idx][1]
                        self.capture_markers.append((cx, cy, ev_idx))
            if self.capture_markers:
                xs, ys, _ = zip(*self.capture_markers)
                self.capture_scatter.set_offsets(np.column_stack([xs, ys]))
        
        if info:
            status = (
                f"t={info['time']:.1f}s / {info.get('max_time', '?')}s\n"
                f"Captured: {info['snake_length']}\n"
                f"alive: {info['num_remaining']}/{self.num_evaders}\n"
                f"caught: {info['num_caught']}"
            )
            self.status_text.set_text(status)
        
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
    
    def close(self):
        if self.save_video and self.video_frames:
            try:
                import imageio
                print(f"Saving {len(self.video_frames)} frames to {self.video_path}...")
                imageio.mimsave(self.video_path, self.video_frames, fps=10)
                print(f"Video saved: {self.video_path}")
            except ImportError:
                print("imageio not installed. Install with: pip install imageio")
        plt.close(self.fig)