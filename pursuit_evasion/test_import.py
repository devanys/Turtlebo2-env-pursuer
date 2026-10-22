"""Test script: cek semua dependency & konfigurasi terload dengan benar."""
import sys
import os

print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print("-" * 50)

# Test imports
try:
    import pybullet as p
    print(f"[OK] pybullet loaded")
except ImportError as e:
    print(f"[FAIL] pybullet: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"[OK] numpy {np.__version__}")
except ImportError as e:
    print(f"[FAIL] numpy: {e}")
    sys.exit(1)

try:
    import matplotlib
    print(f"[OK] matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"[FAIL] matplotlib: {e}")
    sys.exit(1)

try:
    import yaml
    print(f"[OK] pyyaml loaded")
except ImportError as e:
    print(f"[FAIL] pyyaml: {e}")
    sys.exit(1)

# Test config loading
print("-" * 50)
config_path = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"[OK] configs/default.yaml loaded")
    print(f"     simulation.mode        : {config['simulation']['mode']}")
    print(f"     arena.size             : {config['arena']['size']}")
    print(f"     robots.num_pursuers    : {config['robots']['num_pursuers']}")
    print(f"     robots.num_evaders     : {config['robots']['num_evaders']}")
    print(f"     robot_params.wheel_rad : {config['robot_params']['wheel_radius']}")
    print(f"     game.capture_radius    : {config['game']['capture_radius']}")
except Exception as e:
    print(f"[FAIL] config load: {e}")
    sys.exit(1)

# Test URDF path exists
print("-" * 50)
urdf_path = config['robots']['urdf_path']
if os.path.exists(urdf_path):
    print(f"[OK] URDF exists: {urdf_path}")
else:
    print(f"[FAIL] URDF NOT FOUND: {urdf_path}")
    print("       -> Update robots.urdf_path in configs/default.yaml")

print("-" * 50)
print("All checks passed! Ready for Step 6.")