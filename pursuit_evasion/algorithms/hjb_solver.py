"""
Hamilton-Jacobi-Bellman Solver for Pursuit-Evasion Reachability.

Solves HJI PDE backward in time to compute Backward Reachable Tube (BRT).
V(x, t) < 0: state x is in BRT (capture guaranteed)
V(x, t) > 0: state x is safe (evader can escape)
"""
import numpy as np


class HJBSolver:
    """HJB solver for 2D pursuit-evasion reachability."""
    
    def __init__(self, arena_size, v_pursuer, v_evader, capture_radius,
                 grid_size=80, time_horizon=2.0, dt=0.1):
        self.arena_size = arena_size
        self.v_pursuer = v_pursuer
        self.v_evader = v_evader
        self.capture_radius = capture_radius
        self.grid_size = grid_size
        self.time_horizon = time_horizon
        self.dt = dt
        self.num_steps = int(time_horizon / dt)
        
        # Grid hanya sampai 8m (cukup untuk BRT 2s)
        grid_extent = 8.0
        self.x = np.linspace(-grid_extent, grid_extent, grid_size)
        self.y = np.linspace(-grid_extent, grid_extent, grid_size)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        
        self.V = None
        self.V_history = None
        self.brt_radius = 0.0
    
    def precompute(self, verbose=True):
        """Solve HJI PDE backward in time."""
        if verbose:
            print(f"[HJB] Precomputing BRT...")
            print(f"[HJB] Grid: {self.grid_size}x{self.grid_size} = {self.grid_size**2} cells")
            print(f"[HJB] Extent: [{self.x[0]:.1f}, {self.x[-1]:.1f}]m")
            print(f"[HJB] Time: {self.time_horizon}s, dt={self.dt}s, {self.num_steps} steps")
            print(f"[HJB] v_pursuer={self.v_pursuer}, v_evader={self.v_evader}")
            print(f"[HJB] capture_radius={self.capture_radius}")
        
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        # Target set: V < 0 inside capture circle
        V = X**2 + Y**2 - self.capture_radius**2
        
        V_history = np.zeros((self.num_steps + 1, self.grid_size, self.grid_size))
        V_history[-1] = V.copy()
        
        # Hamiltonian: H = (v_evader - v_pursuer) * |grad V|
        # If v_pursuer > v_evader, H < 0, BRT grows backward in time
        H_coeff = self.v_evader - self.v_pursuer
        
        for step in reversed(range(self.num_steps)):
            # Compute gradient (central difference)
            V_x = np.zeros_like(V)
            V_x[1:-1, :] = (V[2:, :] - V[:-2, :]) / (2 * self.dx)
            V_y = np.zeros_like(V)
            V_y[:, 1:-1] = (V[:, 2:] - V[:, :-2]) / (2 * self.dy)
            
            grad_mag = np.sqrt(V_x**2 + V_y**2)
            H = H_coeff * grad_mag
            
            # Backward Euler: V(t-dt) = V(t) + dt * H
            V = V + self.dt * H
            V_history[step] = V.copy()
            
            if verbose and step % 5 == 0:
                brt_cells = (V < 0).sum()
                print(f"[HJB] Step {self.num_steps - step}/{self.num_steps}, "
                      f"BRT: {brt_cells} cells ({100*brt_cells/(self.grid_size**2):.1f}%)")
        
        self.V = V
        self.V_history = V_history
        
        # Compute BRT radius (where V = 0 along x-axis)
        center = self.grid_size // 2
        V_row = self.V[:, center]
        self.brt_radius = self.capture_radius  # default
        for i in range(center, self.grid_size):
            if V_row[i] > 0:
                self.brt_radius = self.x[i]
                break
        
        if verbose:
            brt_cells = (V < 0).sum()
            total = self.grid_size ** 2
            print(f"[HJB] ✓ BRT computed: {brt_cells}/{total} cells ({100*brt_cells/total:.1f}%)")
            print(f"[HJB] ✓ BRT radius: {self.brt_radius:.2f}m")
            print(f"[HJB] ✓ Capture radius: {self.capture_radius}m")
    
    def query_value(self, rel_x, rel_y):
        """Bilinear interpolation of V at (rel_x, rel_y)."""
        i = (rel_x - self.x[0]) / self.dx
        j = (rel_y - self.y[0]) / self.dy
        i = np.clip(i, 0, self.grid_size - 1.001)
        j = np.clip(j, 0, self.grid_size - 1.001)
        i0, j0 = int(i), int(j)
        i1, j1 = i0 + 1, j0 + 1
        wi, wj = i - i0, j - j0
        return ((1-wi)*(1-wj)*self.V[i0,j0] + wi*(1-wj)*self.V[i1,j0] +
                (1-wi)*wj*self.V[i0,j1] + wi*wj*self.V[i1,j1])
    
    def query_gradient(self, rel_x, rel_y):
        """Bilinear interpolation of gradient at (rel_x, rel_y)."""
        i = (rel_x - self.x[0]) / self.dx
        j = (rel_y - self.y[0]) / self.dy
        i = np.clip(i, 0, self.grid_size - 1.001)
        j = np.clip(j, 0, self.grid_size - 1.001)
        i0, j0 = int(i), int(j)
        i1, j1 = i0 + 1, j0 + 1
        wi, wj = i - i0, j - j0
        
        V_x = np.zeros_like(self.V)
        V_x[1:-1, :] = (self.V[2:, :] - self.V[:-2, :]) / (2 * self.dx)
        V_y = np.zeros_like(self.V)
        V_y[:, 1:-1] = (self.V[:, 2:] - self.V[:, :-2]) / (2 * self.dy)
        
        gx = ((1-wi)*(1-wj)*V_x[i0,j0] + wi*(1-wj)*V_x[i1,j0] +
              (1-wi)*wj*V_x[i0,j1] + wi*wj*V_x[i1,j1])
        gy = ((1-wi)*(1-wj)*V_y[i0,j0] + wi*(1-wj)*V_y[i1,j0] +
              (1-wi)*wj*V_y[i0,j1] + wi*wj*V_y[i1,j1])
        return np.array([gx, gy])
    
    def predict_capture_time(self, rel_x, rel_y):
        """Predict time-to-capture based on HJB value function."""
        V = self.query_value(rel_x, rel_y)
        
        # If already in BRT (V < 0), capture is imminent
        if V < 0:
            dist = np.sqrt(rel_x**2 + rel_y**2)
            # Time to physically reach capture radius
            if dist < self.capture_radius:
                return 0.0
            ttc = max(0, (dist - self.capture_radius) / max(self.v_pursuer, 0.1))
            return min(ttc, 2.0)
        
        # If outside BRT, use value function to estimate
        # Higher V = safer, more time
        # Lower V = closer to capture, less time
        dist = np.sqrt(rel_x**2 + rel_y**2)
        if self.v_pursuer > self.v_evader:
            # Pursuer faster — capture eventually
            ttc = (dist - self.capture_radius) / (self.v_pursuer - self.v_evader)
            return min(max(0, ttc), self.time_horizon * 2)
        else:
            return self.time_horizon * 2  # safe
    
    def get_brt_coverage(self):
        if self.V is None:
            return 0.0
        return 100.0 * (self.V < 0).sum() / (self.grid_size ** 2)