import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet
from pybullet_envs import robot_bases, env_bases, scene_stadium, scene_abstract

import numpy as np
import assets
from scenes.stadium import StadiumScene


class FetchRobot(robot_bases.URDFBasedRobot):
	TARG_LIMIT = 0.27
	def __init__(self):
		self.select_list = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",\
							"elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]

		super(FetchRobot, self).__init__(
			'fetch_description/fetch.urdf', "fetch_robot", action_dim=7, obs_dim=7, fixed_base=1, self_collision=False)


	def reset(self, bullet_client):
		self._p = bullet_client
		self.ordered_joints = []

		print(os.path.join(os.path.dirname(__file__), "data", self.model_urdf))

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

		self.robot_specific_reset(self._p)
		s = self.calc_state(
		)  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
		self.potential = self.calc_potential()
		return s

	def robot_specific_reset(self, bullet_client):
		self.jdict["shoulder_pan_joint"].reset_current_position(
			self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
		self.jdict["shoulder_lift_joint"].reset_current_position(
			self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
		self.jdict["upperarm_roll_joint"].reset_current_position(
			self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
		self.jdict["elbow_flex_joint"].reset_current_position(
			self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
		self.jdict["forearm_roll_joint"].reset_current_position(
			self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
		self.jdict["wrist_flex_joint"].reset_current_position(
			self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
		self.jdict["wrist_roll_joint"].reset_current_position(
			self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)


	def apply_action(self, a):
		self.jdict["shoulder_pan_joint"].set_position(0.0)
		self.jdict["shoulder_lift_joint"].set_position(0.1)
		self.jdict["upperarm_roll_joint"].set_position(0.0)
		self.jdict["elbow_flex_joint"].set_position(0.0)
		self.jdict["forearm_roll_joint"].set_position(0.0)
		self.jdict["wrist_flex_joint"].set_position(0.0)
		self.jdict["wrist_roll_joint"].set_position(0.0)
		assert (np.isfinite(a).all())

		# self.central_joint.set_motor_torque(0.05 * float(np.clip(a[0], -1, +1)))
		# self.elbow_joint.set_motor_torque(0.05 * float(np.clip(a[1], -1, +1)))

	def calc_state(self):
		obs = [self.jdict[i] for i in self.select_list if i in self.jdict]
		return np.array(obs)


	def calc_potential(self):
		return 0
		# return -100 * np.linalg.norm(self.to_target_vec)

	
class FetchReachEnv(env_bases.MJCFBaseBulletEnv):
	
	def __init__(self, render=False):
		self.robot = FetchRobot()
		super(FetchReachEnv, self).__init__(self.robot, render)

	def create_single_player_scene(self, bullet_client):
		self.stadium_scene = StadiumScene(bullet_client, gravity=9.8, timestep=0.0165/4, frame_skip=4)
		return self.stadium_scene

 #    def reset(self):
 #        if self.stateId >= 0:
 #            # print("restoreState self.stateId:",self.stateId)
 #            self._p.restoreState(self.stateId)

 #        r = BaseBulletEnv._reset(self)
 #        self._p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

 #        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(self._p,
 #                                                                                             self.stadium_scene.ground_plane_mjcf)
 #        self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex], self.parts[f].bodyPartIndex) for f in
 #                               self.foot_ground_object_names])
 #        self._p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
 #        if self.stateId < 0:
 #            self.stateId=self._p.saveState()
 #        #print("saving state self.stateId:",self.stateId)


	# def create_single_player_scene(self, bullet_client):
	# 	return scene_abstract.SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=0.0165, frame_skip=1)
	# 	self.stadium_scene = StadiumScene(bullet_client, gravity=9.8, timestep=0.0165/4, frame_skip=4)
 #        return self.stadium_scene

	def step(self, a):
		assert (not self.scene.multiplayer)
		self.robot.apply_action(a)
		self.scene.global_step()

		state = self.robot.calc_state()  # sets self.to_target_vec

		potential_old = self.potential
		self.potential = self.robot.calc_potential()
		#
		# electricity_cost = (
		# -0.10 * (np.abs(a[0] * self.robot.theta_dot) + np.abs(a[1] * self.robot.gamma_dot)
		# 		)  # work torque*angular_velocity
		# - 0.01 * (np.abs(a[0]) + np.abs(a[1]))  # stall torque require some energy
		# )
		# stuck_joint_cost = -0.1 if np.abs(np.abs(self.robot.gamma) - 1) < 0.01 else 0.0
		# self.rewards = [
		# float(self.potential - potential_old),
		# float(electricity_cost),
		# float(stuck_joint_cost)
		# ]
		# self.HUD(state, a, False)
		# return state, sum(self.rewards), False, {}
		return state, 0 ,False, {}

	def camera_adjust(self):
		x, y, z = self.robot.fingertip.pose().xyz()
		x *= 0.5
		y *= 0.5
		self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)



