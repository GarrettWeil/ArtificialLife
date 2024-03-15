#necessary imports
import mujoco
import mujoco_viewer
import numpy as np
from dm_control import mjcf
import time

name_index = 0

def add_limb(parent_body, parent_geom, attaching_to, thickness, length): #function to add a body part
	global name_index

	limb_pos = None
	joint_pos = None
	parent_pos = parent_body.pos
	parent_size = parent_geom.size

	name_index = name_index + 1

	if attaching_to == 'front':
		pass
	elif attaching_to == 'right':
		joint_pos = (parent_size[0], 0, 0)
		limb_pos = (parent_size[0]+length, 0, 0)
		limb_size = (length, thickness, thickness)

	elif attaching_to == 'back':
		pass
	elif attaching_to == 'left':
		pass


	# get parent position
	

	# add limb to parent
	limb = parent_body.add('body', name=f'limb{name_index}', pos=limb_pos)

	# add joint to limb
	limb.add('joint', name=f'shoulder{name_index}', type='ball', pos=joint_pos)

	# add geom to limb
	
	limb.add('geom', type='box', name=f'arm{name_index}', size=limb_size)



parent_xml = "blank_slate.xml"


#model generation stage
# Reference: https://github.com/google-deepmind/dm_control/blob/main/dm_control/mjcf/README.md

mjcf_model = mjcf.from_path(parent_xml)

core_body = mjcf_model.worldbody.add('body', name='core_body', pos=[0, 0, 2.5])
core_geom = core_body.add('geom', name='core_geom', type='box', size='.5 .5 2.5', rgba='0 .9 0 1')
core_body.add('joint', name='first', type='free')

#GENOTYPE ENCODING

add_limb(parent_body=core_body, parent_geom=core_geom, attaching_to='right', thickness=0.2, length=2)

mjcf.export_with_assets(mjcf_model, ".", out_file_name="model.xml")

#simulation and rendering stage
DISPLAY = True


m = mujoco.MjModel.from_xml_path("model.xml")
d = mujoco.MjData(m)

if DISPLAY:
	viewer = mujoco_viewer.MujocoViewer(m, d)
	#viewer.cam.azimuth = 20.0 #left right tilt
	viewer.cam.distance = 20.0 #dist to origin
	viewer.cam.elevation = -10 #up down tilt
	#viewer.cam.lookat = np.array([1, 0, 1]) #forward, right, up

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