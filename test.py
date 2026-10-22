import pybullet as p
import pybullet_data

cid = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

robot = p.loadURDF(r"C:/Users/Devan/Downloads/pybullet_robots-master/pybullet_robots-master/data/turtlebot.urdf",
                   [0, 0, 0.05], useFixedBase=False)

print("=== Joints ===")
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    print(f"[{i}] {info[1].decode()} | type={info[2]} | parent={info[2]}")

p.disconnect()