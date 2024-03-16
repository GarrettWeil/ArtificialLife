#necessary imports
import mujoco
import mujoco_viewer
import numpy as np
from dm_control import mjcf
import random

name_index = 0

def add_limb(model, parent_body, parent_geom, face, thickness, length, recursive=0): #function to add a body part
	global name_index # ensure unique naming
	name_index = name_index + 1


	limb_pos = None # body pos is relative to parent
	joint_pos = None # joint pos is relative to its geom
	agroup = None # actuator group
	parent_size = parent_geom.size


	if face == 'front': # define limb/joint sites per face
		agroup = 0
		joint_pos = (0, length, 0)
		limb_pos = (0, -(parent_size[1]+length), 0)
		limb_size = (thickness, length, thickness)

	elif face == 'right':
		agroup = 1
		joint_pos = (-length, 0, 0)
		limb_pos = ((parent_size[0]+length), 0, 0)
		limb_size = (length, thickness, thickness)

	elif face == 'back':
		agroup = 2
		joint_pos = (0, -length, 0)
		limb_pos = (0, parent_size[1]+length, 0)
		limb_size = (thickness, length, thickness)

	elif face == 'left':
		agroup = 3
		joint_pos = (length, 0, 0)
		limb_pos = (-(parent_size[0]+length), 0, 0)
		limb_size = (length, thickness, thickness)
	

	# add limb to parent
	limb = parent_body.add('body', name=f'{face}_limb_{name_index}', pos=limb_pos)

	# add joint to limb
	shoulder = limb.add('joint', name=f'{face}_shoulder_{name_index}', type='ball', pos=joint_pos)

	# add actuator for joint
	model.actuator.add('position', joint=shoulder, group=agroup, name=f'{face}xy_{recursive}', kp=5, gear='0 0 1')
	model.actuator.add('position', joint=shoulder, group=agroup, name=f'{face}xz_{recursive}', kp=5, gear='0 1 0')
	model.actuator.add('position', joint=shoulder, group=agroup, name=f'{face}yz_{recursive}', kp=5, gear='1 0 0')
	

	# add geom to limb
	geom = limb.add('geom', type='box', name=f'{face}_arm_{name_index}', size=limb_size, mass=1, rgba=f'0 0 {1 - recursive*0.1} 1')
	
	if recursive:
		add_limb(model, limb, geom, face, thickness/2, length/2, recursive-1)

	return limb, geom


parent_xml = "blank_slate.xml"

#model generation stage
# Reference: https://github.com/google-deepmind/dm_control/blob/main/dm_control/mjcf/README.md

mjcf_model = mjcf.from_path(parent_xml)

#GENOTYPE ENCODING
body_height = random.uniform(0.2, 3)
body_length = random.uniform(0.2, 3)
body_width = random.uniform(0.2, 3)


# one body
core_body = mjcf_model.worldbody.add('body', name='core_body', pos=[0, 0, body_height])
core_geom = core_body.add('geom', name='core_geom', type='box', size=f'{body_width} {body_length} {body_height}', rgba='0 .9 0 1')
core_body.add('joint', name='first', type='free')

# four, three-segmented legs for each side
for face in ['front', 'right', 'left', 'back']:
	chance_to_skip_face = 0.05

	if random.random() < chance_to_skip_face:
		continue

	recursion_count = random.randrange(0, 4) #random recursions
	leg_thickness = random.uniform(0.2, 0.7) #random leg thickness
	leg_length = random.uniform(0.2, 5) #random leg length


	add_limb(mjcf_model, core_body, core_geom, face, leg_thickness, leg_length, recursion_count)

# save generated file
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

# kp = 100
# kv = 10
# for actuator in range(len(d.ctrl)):
# 	m.actuator_gainprm[actuator, 0] = 0
# 	m.actuator_gainprm[actuator, 1] = -kp
# 	m.actuator_gainprm[actuator, 2] = -kv
	
	

for i in range(4000):
		if DISPLAY is False:
				mujoco.mj_step(m, d)

		elif DISPLAY is True and viewer.is_alive:
				
				# d.ctrl[1] = np.pi

				mujoco.mj_step(m, d)
				viewer.render()

		else:
				break
		
if DISPLAY:
	viewer.close()