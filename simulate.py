import dm_control.mujoco
import mujoco.viewer
import time

m = dm_control.mujoco.MjModel.from_xml_path("example.xml")
d = dm_control.mujoco.MjData(m)
with mujoco.viewer.launch_passive(m, d) as viewer:
	for x in range(1000):
		dm_control.mujoco.mj_step(m, d)
		viewer.sync()
		time.sleep(1/100)
	viewer.close()
