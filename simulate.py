#necessary imports
import mujoco
import mujoco_viewer
import numpy as np
from dm_control import mjcf
import time

parent = "blank_slate.xml"


#model generation stage
# Reference: https://github.com/google-deepmind/dm_control/blob/main/dm_control/mjcf/README.md

mjcf_model = mjcf.from_path("blank_slate.xml")

core_body = mjcf_model.worldbody.add('body', name='core_body', pos=[0, 0, 1])
core_geom = core_body.add('geom', name='core_geom', type='box', size='.5 .5 .5', rgba='0 .9 0 1')
core_body.add('joint', name='first', type='free')

#GENOTYPE ENCODING

# get parent object (core)
#core = m.find('geom', 'core')
#print(core.pos)


found_geom = mjcf_model.find('body', 'core_body')
found_geom.pos = [1, 2, 3]


mjcf.export_with_assets(mjcf_model, ".", out_file_name="model.xml")

#simulation and rendering stage
DISPLAY = False


m = mujoco.MjModel.from_xml_path("model.xml")
d = mujoco.MjData(m)

if DISPLAY:
	viewer = mujoco_viewer.MujocoViewer(m, d)
	viewer.cam.azimuth = 20.0 #left right tilt
	viewer.cam.distance = 7.0 #dist to origin
	viewer.cam.elevation = -10 #up down tilt
	viewer.cam.lookat = np.array([1, 0, 1]) #forward, right, up

for i in range(4000):
		if DISPLAY is False:
				mujoco.mj_step(m, d)

		elif DISPLAY is True and viewer.is_alive:
				mujoco.mj_step(m, d)
				viewer.render()

		else:
				break
		
if DISPLAY:
	viewer.close()