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

class UR5EefRobot(UR5Robot):
	TARG_LIMIT = 0.27
	def __init__(self, action_dim=3, obs_dim=19):
		# self.select_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",\
		# 					"wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
		# self.select_links = ["shoulder_link", "upper_arm_link","forearm_link","ee_link"]

		self.select_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", \
							  "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
		# self.select_links = ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_3_link", "wrist_2_link"]
		self.select_links = ["upper_arm_link", "wrist_3_link"]

		self.last_position = [0,0,0]
		self.ee_link = "ee_link"
		self.lower_limits = [-6, -6, 0, -6, -6, -6]
		self.upper_limits = [6, 0, 6, 0, 6, 6]
		# self.orientation = [0, 0.7071068, 0, 0.7071068]
		# self.orientation = [0.707, 0, 0.707, 0]
		self.orientation = None


		super(UR5EefRobot, self).__init__(action_dim=action_dim, obs_dim=obs_dim)


	def select_ik_solution(self, solutions):
		feasible_solution = []
		for jointPoses in solutions:
			self.jdict['shoulder_pan_joint'].reset_position(jointPoses[0], 0)
			self.jdict['shoulder_lift_joint'].reset_position(jointPoses[1], 0)
			self.jdict['elbow_joint'].reset_position(jointPoses[2], 0)
			self.jdict['wrist_1_joint'].reset_position(jointPoses[3], 0)
			self.jdict['wrist_2_joint'].reset_position(jointPoses[4], 0)
			self.jdict['wrist_3_joint'].reset_position(jointPoses[5], 0)

			if self._contact_detection() is False:
				# print("feasible_solution", feasible_solution)
				if self._is_in_range(jointPoses):
					feasible_solution.append(jointPoses)

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
			# print("bodyinfo1: ", bodyinfo1, "bodyinfo2: ", bodyinfo2)
			# # print("collisions", collisions)
			# print("linkid 1 ", c[3])
			# print("linkid 2", c[4])
			if c[3] ==3 and c[4] ==5:
				continue
			if c[3] ==0 or c[4] ==0:
				continue
			#
			# print("robot parts", self.agents[0].parts)
			# p = self._p.getLinkState(c[1], c[3])[0]
			collision_bodies.append(bodyinfo1[1].decode("utf-8"))
			collision_bodies.append(bodyinfo2[1].decode("utf-8"))

		# todo: check collision bodies

		# print("collision_bodies: ", collision_bodies)

		if len(collision_bodies) != 0:
			if "ur5" in collision_bodies:  # robot collision
				return True
			else:
				return False

		return False

	def robot_specific_reset(self, bullet_client, base_position, base_rotation,
							 eef_position=None, eef_orienration=[0, 0.841471, 0, 0.5403023 ]):
		for n in self.select_joints:
			self.jdict[n].reset_current_position(
				self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)

		self.robot_body.reset_pose(base_position, base_rotation)

		success = None

		ik_fn = ikfast_ur5.get_ik

		# print("eef position, ", eef_position)
		rot = np.array(self._p.getMatrixFromQuaternion(eef_orienration)).reshape(3, 3)
		solutions = ik_fn(list(eef_position), list(rot), [1])

		n_conf_list = [normalize_conf(np.asarray([0,0,0,0,0,0]), conf) for conf in solutions]


		feasible_solutions = self.select_ik_solution(n_conf_list)
		if feasible_solutions == []:
			return False
		print("feasible solutions", feasible_solutions)
		jointPoses = feasible_solutions[0]

		self.jdict['shoulder_pan_joint'].reset_position(jointPoses[0], 0)
		self.jdict['shoulder_lift_joint'].reset_position(jointPoses[1], 0)
		self.jdict['elbow_joint'].reset_position(jointPoses[2], 0)
		self.jdict['wrist_1_joint'].reset_position(jointPoses[3], 0)
		self.jdict['wrist_2_joint'].reset_position(jointPoses[4], 0)
		self.jdict['wrist_3_joint'].reset_position(jointPoses[5], 0)

		self.last_position = list(self._p.getLinkState(self.robot_body.bodies[0], self.parts['ee_link'].bodyPartIndex)[0])
		# self.last_state = self._p.getLinkState(self.robot_body.bodies[0], self.parts['ee_link'].bodyPartIndex)
		# print("link state is: ", state)

	def reset(self, bullet_client, base_position=[0, 0, -1.2], base_rotation=[0, 0, 0, 1], eef_pose=None):
		self._p = bullet_client
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

			else:
				self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
					self._p,
					self._p.loadURDF(os.path.join(assets.getDataPath(), self.model_urdf),
									 basePosition=self.basePosition,
									 baseOrientation=self.baseOrientation,
									 useFixedBase=self.fixed_base))

		if self.robot_specific_reset(self._p, base_position, base_rotation, eef_pose) is False:
			return False
		self.jdict['shoulder_pan_joint'].max_velocity = 3.15
		self.jdict['shoulder_lift_joint'].max_velocity = 3.15
		self.jdict['elbow_joint'].max_velocity = 3.15
		self.jdict['wrist_1_joint'].max_velocity = 3.2
		self.jdict['wrist_2_joint'].max_velocity = 3.2
		self.jdict['wrist_3_joint'].max_velocity = 3.2

		s = self.calc_state(
		)  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
		self.potential = self.calc_potential()

		return s


	def apply_action(self, a):
		# todo: add dynamic limitation
		assert (np.isfinite(a).all())
		#scale
		max_eef_velocity = 0.02
		scale = max_eef_velocity
		# state = self._p.getLinkState(self.robot_body.bodies[0], self.parts['ee_link'].bodyPartIndex)
		# # print("link state is: ", state)
		# position = list(state[0])

		position = self.last_position.copy()
		for i in range((self.action_space.shape)[0]):
			position[i]+=a[i] * scale

		# #---for debug---#
		# debug_v = a *scale
		# print("velocity is: ", debug_v)
		# #---end debug---#

		jointPoses = self._p.calculateInverseKinematics(self.robot_body.bodies[0], self.parts['ee_link'].bodyPartIndex,
														position
														)
		#velocity
		self.last_position=position

		self.jdict['shoulder_pan_joint'].reset_position(jointPoses[0], 0)
		self.jdict['shoulder_lift_joint'].reset_position(jointPoses[1], 0)
		self.jdict['elbow_joint'].reset_position(jointPoses[2], 0)
		self.jdict['wrist_1_joint'].reset_position(jointPoses[3], 0)
		self.jdict['wrist_2_joint'].reset_position(jointPoses[4], 0)
		self.jdict['wrist_3_joint'].reset_position(jointPoses[5], 0)

		# self.jdict['shoulder_pan_joint'].set_position(jointPoses[0])
		# self.jdict['shoulder_lift_joint'].set_position(jointPoses[1])
		# self.jdict['elbow_joint'].set_position(jointPoses[2])
		# self.jdict['wrist_1_joint'].set_position(jointPoses[3])
		# self.jdict['wrist_2_joint'].set_position(jointPoses[4])
		# self.jdict['wrist_3_joint'].set_position(jointPoses[5])

		# self.last_position=list(self._p.getLinkState(self.robot_body.bodies[0], self.parts['ee_link'].bodyPartIndex)[0])

	def calc_state(self):
		# state = self._p.getLinkState(self.robot_body.bodies[0], self.parts['ee_link'].bodyPartIndex, computeLinkVelocity=1)
		# # print("link state is: ", state)
		# position = np.asarray(list(state[0]))
		# velocity = np.asarray(list(state[-2]))
		# # obs = np.concatenate( (velocity, position))
		# print("!!!!!!!!!!!!!obs shape is: ", position.shape)
		# return position

		# link_position = np.asarray([self.parts[j].get_position() for j in self.select_links if j in self.parts])

		joint_position = np.asarray(
			[self.jdict[i].get_position() for i in self.select_joints if
			 i in self.jdict])  # position
		joint_velocity = np.asarray(
			[self.jdict[i].get_velocity() for i in self.select_joints if i in self.jdict])  # velocity

		eef_pose = self.parts[self.ee_link].get_pose()  # position [0:3], orientation [3:7]
		# obs = np.concatenate((eef_pose, link_position.flatten()))
		obs = np.concatenate((eef_pose, joint_position.flatten(), joint_velocity.flatten()))
		return obs



	def calc_potential(self):
		return 0
		# return -100 * np.linalg.norm(self.to_target_vec)
