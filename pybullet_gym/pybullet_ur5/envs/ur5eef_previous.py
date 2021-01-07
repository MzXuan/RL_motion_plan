import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet
from pybullet_envs import env_bases, scene_stadium, scene_abstract

import numpy as np
import robot_bases
import assets
from scenes.stadium import StadiumScene
from ur5 import UR5Robot
import random

import ikfast_ur5


def normalize_conf(start, end):
    circle = 6.2831854
    normal_e = []
    for s, e in zip(start, end):
        if e > s:
            test = e - circle
        else:
            test = e + circle
        normal_e.append(test if np.linalg.norm(test-s)<np.linalg.norm(e-s) else e)
    return normal_e

class UR5EefPreviousRobot(UR5Robot):
	TARG_LIMIT = 0.27
	def __init__(self, dt, action_dim=3, obs_dim=14):
		# self.select_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",\
		# 					"wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
		# self.select_links = ["shoulder_link", "upper_arm_link","forearm_link","ee_link"]

		self.select_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", \
							  "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

		self.select_links = ["upper_arm_link", "wrist_3_link"]

		self.last_position = [0,0,0]
		self.ee_link = "ee_link"
		self.lower_limits = [-3.5, -3.5, -3.5, -3.5, -3.5, -6]
		self.upper_limits = [3.5, 3.5, 3.5, 3.5, 3.5, 6]
		# self.orientation = [0, 0.7071068, 0, 0.7071068]
		# self.orientation = [0.707, 0, 0.707, 0]
		self.orientation = [0, 0.841471, 0, 0.5403023]
		self.dt = dt
		self.n_dofs = 6

		super(UR5EefPreviousRobot, self).__init__(dt=dt, action_dim=action_dim, obs_dim=obs_dim)


	def select_ik_solution(self, solutions):
		feasible_solution = []

		contact_free_solution = []
		for jointPoses in solutions:
			self.jdict['shoulder_pan_joint'].reset_position(jointPoses[0], 0)
			self.jdict['shoulder_lift_joint'].reset_position(jointPoses[1], 0)
			self.jdict['elbow_joint'].reset_position(jointPoses[2], 0)
			self.jdict['wrist_1_joint'].reset_position(jointPoses[3], 0)
			self.jdict['wrist_2_joint'].reset_position(jointPoses[4], 0)
			self.jdict['wrist_3_joint'].reset_position(jointPoses[5], 0)
			self._p.stepSimulation()

			if self._contact_detection() is False:
				contact_free_solution.append(jointPoses)

		if contact_free_solution == []:
			print("can not find contact free solution")
		else:
			for jp in contact_free_solution:
				if self._is_in_range(jp):
					feasible_solution.append(jp)
		if feasible_solution == []:
			print("can not find in range solution")

		return feasible_solution


	def _is_in_range(self, jointPoses):
		for i in range(0, 6):
			if jointPoses[i] < self.upper_limits[i] and jointPoses[i] > self.lower_limits[i]:
				continue
			else:
				return False
		return True


	def _contact_detection(self):
		# collision detection
		collisions = self._p.getContactPoints()
		collision_bodies = []
		for c in collisions:
			bodyinfo1 = self._p.getBodyInfo(c[1])
			bodyinfo2 = self._p.getBodyInfo(c[2])

			if c[3] ==3 and c[4] ==5:
				continue
			if c[3] ==0 or c[4] ==0:
				continue

			# print("bodyinfo1: ", bodyinfo1, "bodyinfo2: ", bodyinfo2)
			# print("collisions", collisions)
			# print("linkid 1 ", c[3])
			# print("linkid 2", c[4])
			# #
			# print("robot parts", self.agents[0].parts)
			# p = self._p.getLinkState(c[1], c[3])[0]
			collision_bodies.append(bodyinfo1[1].decode("utf-8"))
			collision_bodies.append(bodyinfo2[1].decode("utf-8"))

		if len(collision_bodies) != 0:
			if "ur5" in collision_bodies:  # robot collision
				return True
			else:
				return False

		return False

	def robot_specific_reset(self, bullet_client, base_position, base_rotation,
							 eef_position=None, eef_orienration=[0, 0.841471, 0, 0.5403023 ],
							 joint_angle=None):

		if joint_angle is not None:
			print("'reset joint angle", joint_angle)

			self.jdict['shoulder_pan_joint'].reset_position(joint_angle[0], 0)
			self.jdict['shoulder_lift_joint'].reset_position(joint_angle[1], 0)
			self.jdict['elbow_joint'].reset_position(joint_angle[2], 0)
			self.jdict['wrist_1_joint'].reset_position(joint_angle[3], 0)
			self.jdict['wrist_2_joint'].reset_position(joint_angle[4], 0)
			self.jdict['wrist_3_joint'].reset_position(joint_angle[5], 0)

			self.last_position = list(
			self._p.getLinkState(self.robot_body.bodies[0], self.parts['ee_link'].bodyPartIndex)[0])
			return


		self.robot_body.reset_pose(base_position, base_rotation)

		# ik_fn = ikfast_ur5.get_ik
		#
		# pose = self._p.multiplyTransforms(positionA=[0, 0, 0], orientationA=[0, 0, -1, 0],
		# 								  positionB=eef_position, orientationB=eef_orienration)
		#
		#
		# position = np.asarray(pose[0])
		# rotation = np.array(self._p.getMatrixFromQuaternion(pose[1])).reshape(3, 3)
		# solutions = ik_fn(position, rotation, [1])
		#
		# n_conf_list = [normalize_conf(np.asarray([0,0,0,0,0,0]), conf) for conf in solutions]
		#
		# feasible_solutions = self.select_ik_solution(n_conf_list)


		#test pybullet ik
		conf = self._p.calculateInverseKinematics(self.robot_body.bodies[0], self.parts['ee_link'].bodyPartIndex,
														eef_position,eef_orienration
														)
		n_conf_list = [normalize_conf(np.asarray([0, 0, 0, 0, 0, 0]), conf)]
		feasible_solutions = self.select_ik_solution(n_conf_list)





		if feasible_solutions == []:
			print("can not find feasible soltion for robot eef: ", eef_position)
			return False
		# print("feasible solutions", feasible_solutions)
		jointPoses = feasible_solutions[0]


		self.jdict['shoulder_pan_joint'].reset_position(jointPoses[0], 0)
		self.jdict['shoulder_lift_joint'].reset_position(jointPoses[1], 0)
		self.jdict['elbow_joint'].reset_position(jointPoses[2], 0)
		self.jdict['wrist_1_joint'].reset_position(jointPoses[3], 0)
		self.jdict['wrist_2_joint'].reset_position(jointPoses[4], 0)
		self.jdict['wrist_3_joint'].reset_position(jointPoses[5], 0)

		self.last_position = list(self._p.getLinkState(self.robot_body.bodies[0], self.parts['ee_link'].bodyPartIndex)[0])




	def reset(self, bullet_client, client_id, base_position=[0, 0, -1.2], base_rotation=[0, 0, 0, 1],
			  eef_pose=None, joint_angle=None):

		self._p = bullet_client
		self.client_id = client_id
		self.ordered_joints = []

		# print(os.path.join(os.path.dirname(__file__), "data", self.model_urdf))

		if self.jdict is None:
			if self.self_collision:
				self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
					self._p,
					self._p.loadURDF(os.path.join(assets.getDataPath(), self.model_urdf),
									 basePosition=self.basePosition,
									 baseOrientation=self.baseOrientation,
									 useFixedBase=self.fixed_base,
									 flags=pybullet.URDF_USE_SELF_COLLISION))

				self.controlled_joints_id = [self.jdict[joint_name].jointIndex for joint_name in self.select_joints]

			else:
				self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
					self._p,
					self._p.loadURDF(os.path.join(assets.getDataPath(), self.model_urdf),
									 basePosition=self.basePosition,
									 baseOrientation=self.baseOrientation,
									 useFixedBase=self.fixed_base))

		if self.robot_specific_reset(self._p, base_position, base_rotation, eef_pose, joint_angle=joint_angle) is False:
			return False
		self.jdict['shoulder_pan_joint'].max_velocity = 3.15
		self.jdict['shoulder_lift_joint'].max_velocity = 3.15
		self.jdict['elbow_joint'].max_velocity = 3.15
		self.jdict['wrist_1_joint'].max_velocity = 3.2
		self.jdict['wrist_2_joint'].max_velocity = 3.2
		self.jdict['wrist_3_joint'].max_velocity = 3.2

		# self.last_eef_position = self.parts['ee_link'].get_position()
		self.last_eef_position, _, self.last_ee_vel, _ = self.getCurrentEEPos()
		s = self.calc_state(
		)  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
		self.potential = self.calc_potential()

		return s

	def stop(self):
		return 0


	def apply_action(self,a):
		#set position without real dynamic
		assert (np.isfinite(a).all())

		#scale
		max_eef_velocity = 2
		step_max_velocity = max_eef_velocity*self.dt



		ee_lin_pos, ee_lin_ori, _, _ = self.getCurrentEEPos()
		target_position = ee_lin_pos + np.asarray(a) * step_max_velocity



		# # ------------------ik fast ----------------------#
		# ik_fn = ikfast_ur5.get_ik
		#
		#
		# pose = self._p.multiplyTransforms(positionA=[0, 0, 0], orientationA=[0, 0, -1, 0],
		# 								  positionB=target_position, orientationB=self.orientation)
		#
		# position = np.asarray(pose[0])
		# rotation = np.array(self._p.getMatrixFromQuaternion(pose[1])).reshape(3, 3)
		# solutions = ik_fn(position, rotation, [1])
		# n_conf_list = [normalize_conf(np.asarray([0,0,0,0,0,0]), conf) for conf in solutions]
		#
		#
		# feasible_solutions = self.select_ik_solution(n_conf_list)
		#
		# if feasible_solutions == []:
		# 	print("can not find feasible soltion for robot eef: {}, use pybullet ik ".format(position))
		# 	jointPoses = self._p.calculateInverseKinematics(self.robot_body.bodies[0], self.parts['ee_link'].bodyPartIndex,
		# 													target_position,self.orientation)
		# else:
		#
		# 	jointPoses = feasible_solutions[0]



		#------------  pose from simulator -------------#
		jointPoses = self._p.calculateInverseKinematics(self.robot_body.bodies[0], self.parts['ee_link'].bodyPartIndex,
														target_position,self.orientation
														)

		target_jp = np.asarray(jointPoses[:6])
		# print("next joint: ", target_jp)

		for i, joint_name in enumerate(self.select_joints):
			self.jdict[joint_name].set_position(target_jp[i], maxVelocity=0.8)



	def getCurrentEEPos(self):
		ee_state = self._p.getLinkState(self.robot_body.bodies[0], self.parts['ee_link'].bodyPartIndex,
									computeLinkVelocity=True,
									computeForwardKinematics=True)
		ee_lin_pos = np.array(ee_state[0])
		ee_lin_ori = np.array(ee_state[1])
		ee_lin_vel = np.array(ee_state[6])  # worldLinkLinearVelocity
		ee_ang_vel = np.array(ee_state[7])  # worldLinkAngularVelocity
		return ee_lin_pos, ee_lin_ori, ee_lin_vel, ee_ang_vel

	def getCurrentJointPosVel(self):

		cur_joint_states = self._p.getJointStates(self.robot_body.bodies[0], self.controlled_joints_id)
		cur_joint_pos = [cur_joint_states[i][0] for i in range(self.n_dofs)]
		cur_joint_vel = [cur_joint_states[i][1] for i in range(self.n_dofs)]
		return cur_joint_pos.copy(), cur_joint_vel.copy()


	def computeGravityCompensationControlPolicy(self):
		[cur_joint_pos, cur_joint_vel] = self.getCurrentJointPosVel()
		grav_comp_torque = self._p.calculateInverseDynamics(self.robot_body.bodies[0], cur_joint_pos,
														[0] * self.n_dofs,
														[0] * self.n_dofs)
		return np.array(grav_comp_torque)



	def calc_state(self):

		# link_position = np.asarray([self.parts[j].get_position() for j in self.select_links if j in self.parts])

		joint_position = np.asarray(
			[self.jdict[i].get_position() for i in self.select_joints if
			 i in self.jdict])  # position


		joint_velocity = np.asarray(
			[self.jdict[i].get_velocity() for i in self.select_joints if i in self.jdict])  # velocity

		# eef_pose = self.parts[self.ee_link].get_pose()  # position [0:3], orientation [3:7]

		ee_lin_pos, _, ee_lin_vel,_ = self.getCurrentEEPos()

		# obs = np.concatenate([ee_lin_pos, ee_lin_vel, self.last_ee_vel, joint_position[:-1].flatten()])
		obs = np.concatenate([ee_lin_pos, joint_position.flatten(), ee_lin_vel, self.last_ee_vel])
		# obs = np.concatenate((ee_lin_pos, ee_lin_vel,joint_position.flatten()))




		# obs = np.concatenate([ee_lin_pos, ee_lin_vel, self.last_ee_vel])
		self.last_ee_vel = ee_lin_vel

		return obs



	def calc_potential(self):
		return 0
		# return -100 * np.linalg.norm(self.to_target_vec)