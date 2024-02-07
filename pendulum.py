import mujoco
import mujoco_viewer
import numpy as np
import time

m = mujoco.MjModel.from_xml_path("pendulum.xml")
d = mujoco.MjData(m)

viewer = mujoco_viewer.MujocoViewer(m, d)

viewer.data.qpos[0] = np.pi/2

# Set camera configuration
viewer.cam.azimuth = 90.0
viewer.cam.distance = 5.0
viewer.cam.elevation = -5
viewer.cam.lookat = np.array([0.012768, -0.000000, 1.254336])
 
for i in range(10000):
        if viewer.is_alive:
                mujoco.mj_step(m, d)
                viewer.render()
        else:
                break
viewer.close()
