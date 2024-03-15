import mujoco
import mujoco_viewer
import numpy as np
import time

m = mujoco.MjModel.from_xml_path("pendulum.xml")
d = mujoco.MjData(m)

viewer = mujoco_viewer.MujocoViewer(m, d)

viewer.data.qpos[0] = np.pi/2

# Set camera configuration
viewer.cam.azimuth = 90.0 #deg left right tilt
viewer.cam.distance = 5.0 #dist to origin
viewer.cam.elevation = -20 # deg up down tile
                             #forward     #right   #up
viewer.cam.lookat = np.array([0, 0, 0.5])
 
for i in range(400):
        if viewer.is_alive:
                mujoco.mj_step(m, d)
                viewer.render()
        else:
                break
viewer.close()
