import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class PaperDataPanel:

    def __init__(self, num_evaders=9, max_time=120.0):
        self.num_evaders = num_evaders
        self.max_time = max_time

        self.time_hist = deque(maxlen=5000)
        self.capture_length_hist = deque(maxlen=5000)
        self.alive_hist = deque(maxlen=5000)
        self.head_speed_hist = deque(maxlen=5000)
        self.head_omega_hist = deque(maxlen=5000)
        self.nearest_dist_hist = deque(maxlen=5000)
        self.capture_events = []

        plt.ion()
        self.fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        self.fig.canvas.manager.set_window_title('Paper Data Panel')
        try:
            self.fig.canvas.manager.window.wm_geometry("+700+50")
        except:
            pass

        self.ax1 = axes[0, 0]
        self.ax2 = axes[0, 1]
        self.ax3 = axes[0, 2]
        self.ax4 = axes[1, 0]
        self.ax5 = axes[1, 1]
        self.ax6 = axes[1, 2]

        self._setup_plots()

        self.fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

    def _setup_plots(self):
        self.line_capture, = self.ax1.plot([], [], 'b-', linewidth=2)
        self.ax1.set_title('Capture Length Over Time', fontweight='bold', fontsize=10)
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Capture Length')
        self.ax1.set_xlim(0, self.max_time)
        self.ax1.set_ylim(0, self.num_evaders + 1)
        self.ax1.grid(True, alpha=0.3)

        self.line_alive, = self.ax2.plot([], [], 'r-', linewidth=2)
        self.ax2.set_title('Evaders Alive Over Time', fontweight='bold', fontsize=10)
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Alive Count')
        self.ax2.set_xlim(0, self.max_time)
        self.ax2.set_ylim(0, self.num_evaders + 1)
        self.ax2.grid(True, alpha=0.3)

        self.ax3.set_title('Capture Events Timeline', fontweight='bold', fontsize=10)
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Evader Index')
        self.ax3.set_xlim(0, self.max_time)
        self.ax3.set_ylim(-1, self.num_evaders)
        self.ax3.grid(True, alpha=0.3)

        self.line_speed, = self.ax4.plot([], [], 'g-', linewidth=2, label='Linear')
        self.line_omega, = self.ax4.plot([], [], 'm--', linewidth=1.5, label='Angular')
        self.ax4.set_title('Head Velocity Profile', fontweight='bold', fontsize=10)
        self.ax4.set_xlabel('Time (s)')
        self.ax4.set_ylabel('Velocity')
        self.ax4.set_xlim(0, self.max_time)
        self.ax4.set_ylim(-3, 3)
        self.ax4.grid(True, alpha=0.3)
        self.ax4.legend(loc='upper right', fontsize=8)

        self.line_dist, = self.ax5.plot([], [], 'k-', linewidth=2)
        self.ax5.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Capture radius')
        self.ax5.axhline(y=1.5, color='orange', linestyle=':', alpha=0.5, label='Detection radius')
        self.ax5.set_title('Distance: Head to Nearest Evader', fontweight='bold', fontsize=10)
        self.ax5.set_xlabel('Time (s)')
        self.ax5.set_ylabel('Distance (m)')
        self.ax5.set_xlim(0, self.max_time)
        self.ax5.set_ylim(0, 15)
        self.ax5.grid(True, alpha=0.3)
        self.ax5.legend(loc='upper right', fontsize=8)

        self.ax6.axis('off')
        self.ax6.set_title('Summary Statistics', fontweight='bold', fontsize=11)
        self.summary_text = self.ax6.text(
            0.05, 0.95, '', transform=self.ax6.transAxes,
            fontsize=8.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
                      edgecolor='gray', linewidth=1)
        )

    def update(self, obs, info, head_velocity=None):
        t = info.get('time', 0)
        capture_length = info.get('snake_length', info.get('capture_length', 1))
        alive = info.get('num_remaining', 0)

        for cap_step, pur_idx, ev_idx in info.get('capture_events', []):
            cap_time = cap_step * 0.1
            if not any(c[1] == ev_idx for c in self.capture_events):
                self.capture_events.append((cap_time, ev_idx))

        self.time_hist.append(t)
        self.capture_length_hist.append(capture_length)
        self.alive_hist.append(alive)

        if head_velocity is not None:
            self.head_speed_hist.append(head_velocity[0])
            self.head_omega_hist.append(head_velocity[1])
        elif len(self.time_hist) > len(self.head_speed_hist):
            self.head_speed_hist.append(0)
            self.head_omega_hist.append(0)

        head_pos = obs['head_pose'][:2]
        evader_poses = obs['evader_poses']
        evader_alive = obs['evader_alive']
        if any(evader_alive):
            alive_idx = np.where(evader_alive)[0]
            positions = evader_poses[alive_idx][:, :2]
            dists = np.linalg.norm(positions - head_pos, axis=1)
            self.nearest_dist_hist.append(dists.min())
        else:
            self.nearest_dist_hist.append(0)

        t_arr = np.array(self.time_hist)

        self.line_capture.set_data(t_arr, np.array(self.capture_length_hist))
        self.line_alive.set_data(t_arr, np.array(self.alive_hist))

        speed_arr = np.array(self.head_speed_hist)
        omega_arr = np.array(self.head_omega_hist)
        n_speed = len(speed_arr)
        t_speed = t_arr[-n_speed:] if n_speed > 0 else t_arr
        self.line_speed.set_data(t_speed, speed_arr)
        self.line_omega.set_data(t_speed, omega_arr)

        self.line_dist.set_data(t_arr, np.array(self.nearest_dist_hist))

        self.ax3.clear()
        self.ax3.set_title('Capture Events Timeline', fontweight='bold', fontsize=10)
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Evader Index')
        self.ax3.set_xlim(0, self.max_time)
        self.ax3.set_ylim(-1, self.num_evaders)
        self.ax3.grid(True, alpha=0.3)
        if self.capture_events:
            cap_times = [c[0] for c in self.capture_events]
            cap_evs = [c[1] for c in self.capture_events]
            self.ax3.scatter(cap_times, cap_evs, c='red', s=150,
                             marker='x', linewidths=2, zorder=5)
            for ct, ce in self.capture_events:
                self.ax3.annotate(f'E{ce}@{ct:.1f}s', (ct, ce),
                                  textcoords="offset points", xytext=(5, 5),
                                  fontsize=7, fontweight='bold')

        summary = self._build_summary(info)
        self.summary_text.set_text(summary)

        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except:
            pass

    def _build_summary(self, info):
        lines = []
        lines.append("=" * 42)
        lines.append("  PAPER DATA SUMMARY")
        lines.append("=" * 42)
        lines.append(f"  Time:       {info.get('time', 0):6.2f}s / {self.max_time:.0f}s")
        lines.append(f"  Capture len:{info.get('snake_length', 1)}")
        lines.append(f"  Captured:   {info.get('num_caught', 0)}/{self.num_evaders}")
        lines.append(f"  Remaining:  {info.get('num_remaining', 0)}")
        lines.append("-" * 42)

        if self.capture_events:
            lines.append("  CAPTURE EVENTS:")
            for ct, ce in self.capture_events:
                lines.append(f"    Evader {ce}: t = {ct:6.2f}s")

            if len(self.capture_events) > 1:
                intervals = [self.capture_events[i+1][0] - self.capture_events[i][0]
                             for i in range(len(self.capture_events)-1)]
                lines.append("-" * 42)
                lines.append("  CAPTURE INTERVALS:")
                lines.append(f"    Mean:   {np.mean(intervals):6.2f}s")
                lines.append(f"    Std:    {np.std(intervals):6.2f}s")
                lines.append(f"    Min:    {min(intervals):6.2f}s")
                lines.append(f"    Max:    {max(intervals):6.2f}s")

        if len(self.head_speed_hist) > 10:
            lines.append("-" * 42)
            lines.append("  HEAD VELOCITY STATS:")
            lines.append(f"    Mean v: {np.mean(self.head_speed_hist):6.2f} m/s")
            lines.append(f"    Max  v: {np.max(self.head_speed_hist):6.2f} m/s")
            lines.append(f"    Min  v: {np.min(self.head_speed_hist):6.2f} m/s")
            lines.append(f"    Mean w: {np.mean(self.head_omega_hist):6.2f} rad/s")
            lines.append(f"    Max  w: {np.max(self.head_omega_hist):6.2f} rad/s")

        if len(self.nearest_dist_hist) > 10:
            lines.append("-" * 42)
            lines.append("  DISTANCE STATS:")
            lines.append(f"    Mean:   {np.mean(self.nearest_dist_hist):6.2f} m")
            lines.append(f"    Min:    {np.min(self.nearest_dist_hist):6.2f} m")
            lines.append(f"    Max:    {np.max(self.nearest_dist_hist):6.2f} m")

        lines.append("=" * 42)

        if info.get('num_remaining', 0) == 0:
            lines.append("  RESULT: HEAD WINS! All captured")
            total_time = info.get('time', 0)
            if total_time > 0:
                rate = self.num_evaders / total_time
                lines.append(f"  Capture rate: {rate:.2f} evaders/s")
        elif info.get('time', 0) >= self.max_time:
            lines.append(f"  RESULT: TIMEOUT ({info.get('num_remaining', 0)} survived)")
        else:
            lines.append("  STATUS: IN PROGRESS")

        return "\n".join(lines)

    def close(self):
        plt.close(self.fig)