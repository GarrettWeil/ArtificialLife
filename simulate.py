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


mjcf_model = mjcf.from_path("blank_slate.xml")

#INITIAL GENOTYPE ENCODING
body_height = random.uniform(0.5, 3)
body_length = random.uniform(0.5, 3)
body_width = random.uniform(0.5, 3)
chance_to_skip_face = 0.05
chance_to_recurse_more = 0.1
chance_to_recurse_less = 0.1


recursion_counts = []
leg_thicknesses = []
leg_lengths = []
face_flags = []
num_generations = 5

for generation in range(num_generations):

	# indirect encoding is saved, so we can just rebuild from a blank slate each time
	mjcf_model = mjcf.from_path("blank_slate.xml")

	# MUTATION

	body_height = max(1, body_height + random.uniform(-0.2, 0.2))
	body_length = max(1, body_length + random.uniform(-0.2, 0.2))
	body_width = max(1, body_width + random.uniform(-0.2, 0.2))
	
	


	# one body
	core_body = mjcf_model.worldbody.add('body', name='core_body', pos=[0, 0, body_height])
	core_geom = core_body.add('geom', name='core_geom', type='box', size=f'{body_width} {body_length} {body_height}', rgba='0 .9 0 1')
	core_body.add('joint', name='first', type='free')

	# four, three-segmented legs for each side
	for index, face in enumerate(['front', 'right', 'left', 'back']):
			if generation == 0:
				recursion_counts.append(random.randint(0, 4))
				leg_thicknesses.append(random.uniform(0.2, 0.5)) #random leg thickness
				leg_lengths.append(random.uniform(0.2, 5)) #random leg length

				if random.random() < chance_to_skip_face:
					face_flags.append(True)
					continue
				else:
					face_flags.append(False)

			elif face_flags[index]:
				continue
			
			if random.random() < chance_to_recurse_more:
				recursion_counts[index] = min(4, 1 + recursion_counts[index])
			elif random.random() < chance_to_recurse_less:
				recursion_counts[index] += 1

			leg_thicknesses[index] = max(0.2, leg_thicknesses[index] + random.uniform(-0.5, 0.5))
			leg_lengths[index] = max(0.2, leg_lengths[index] + random.uniform(-0.5, 0.5))
	
			add_limb(mjcf_model, core_body, core_geom, face, leg_thicknesses[index], leg_lengths[index], recursion_counts[index])

	# save generated file
	mjcf.export_with_assets(mjcf_model, ".", out_file_name=f"child_{generation}.xml")
	print(recursion_counts)






	#simulation and rendering stage
	DISPLAY = True


	m = mujoco.MjModel.from_xml_path(f"child_{generation}.xml")
	d = mujoco.MjData(m)

	if DISPLAY:
		viewer = mujoco_viewer.MujocoViewer(m, d)
		viewer.cam.azimuth = 20 #left right tilt
		viewer.cam.distance = 30.0 #dist to origin
		viewer.cam.elevation = -40 #up down tilt
		#viewer.cam.lookat = np.array([1, 0, 1]) #forward, right, up

	# kp = 100
	# kv = 10
	# for actuator in range(len(d.ctrl)):
	# 	m.actuator_gainprm[actuator, 0] = 0
	# 	m.actuator_gainprm[actuator, 1] = -kp
	# 	m.actuator_gainprm[actuator, 2] = -kv
		
		

	for i in range(400):
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