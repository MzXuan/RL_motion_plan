import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet
from pybullet_utils import bullet_client
from pybullet_envs import env_bases, scene_stadium, scene_abstract

import random

import numpy as np
import assets
import robot_bases
import math
from scenes.stadium import StadiumScene, PlaneScene
import gym, gym.spaces, gym.utils



class RealHumanoid(robot_bases.MJCFBasedRobot):
    self_collision = True
    foot_list = ["right_foot", "left_foot"]  # "left_hand", "right_hand"

    def __init__(self, obs_dim=12, random_yaw=False, random_lean=False):
        self.power = 0.41
        self.camera_x = 0
        self.select_joints = ["left_shoulder1", "left_shoulder2", "left_elbow"]
        self.select_links = ["left_upper_arm", "left_lower_arm", "left_hand_true"]

        high = np.inf * np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)

        # self.model_xml = model_xml
        self.robot_name = 'humanoid'


    def reset(self, bullet_client):
        self._p = bullet_client
        self.leftarm1 = bullet_client.loadURDF(os.path.join(assets.getDataPath(), "scenes_data", "cylinder/cylinder.urdf"),
                               [0, 0, 0],
                               [0.000000, 0.000000, 0.0, 1])
        self.leftarm2 = bullet_client.loadURDF(os.path.join(assets.getDataPath(), "scenes_data", "cylinder/cylinder.urdf"),
                                           [0, 0, 0],
                                           [0.000000, 0.000000, 0.0, 1])

        self.robot_id = self.leftarm1
        # self.arm1 = self._p.createCollisionShape(shapeType=pybullet.GEOM_CYLINDER, radius=0.03, height=0.25)
        # self.arm2 = self._p.createCollisionShape(shapeType=pybullet.GEOM_CYLINDER, radius=0.03, height=0.25)

        # print("Created bullet_client with id=", self._p._client)

        # if (self.doneLoading == 0):
        #     self.ordered_joints = []
        #     self.doneLoading = 1
        #     if self.self_collision:
        #         self.objects = self._p.loadMJCF(os.path.join(assets.getDataPath(), self.model_xml),
        #                                         flags=pybullet.URDF_USE_SELF_COLLISION |
        #                                               pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        #         self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
        #             self._p, self.objects)
        #     else:
        #         self.objects = self._p.loadMJCF(os.path.join(assets.getDataPath(), self.model_xml))
        #         self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
        #             self._p, self.objects)
        #
        # self.robot_specific_reset(self._p)
        #
        # self.jdict['left_shoulder1'].max_velocity = 2.0
        # self.jdict['left_shoulder2'].max_velocity = 1.8
        # self.jdict['left_elbow'].max_velocity = 1.5

        s = self.calc_state(
        )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use

        return s

    def robot_specific_reset(self, bullet_client):
        # WalkerBase.robot_specific_reset(self, bullet_client)
        self._p = bullet_client
        # for j in self.ordered_joints:
        #     j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)
        # self.robot_body.reset_pose(base_position, base_rotation)
        # self.feet = [self.parts[f] for f in self.foot_list]
        # self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
        # self.scene.actor_introduce(self)
        # self.initial_z = None
        #
        #
        # self.jdict["abdomen_z"].reset_position(0.0, 0)
        # self.jdict["abdomen_y"].reset_position(0.0, 0)
        # self.jdict["abdomen_x"].reset_position(0.0, 0)
        # self.jdict["right_hip_x"].reset_position(0.0, 0)
        # self.jdict["right_hip_z"].reset_position(0.0, 0)
        # self.jdict["right_hip_y"].reset_position(0.0, 0)
        # self.jdict["right_knee"].reset_position(0, 0)
        # self.jdict["left_hip_x"].reset_position(0.0, 0)
        # self.jdict["left_hip_z"].reset_position(0.0, 0)
        # self.jdict["left_hip_y"].reset_position(0.0, 0)
        # self.jdict["left_knee"].reset_position(0.0, 0)
        # self.jdict["right_shoulder1"].reset_position(0, 0)
        # self.jdict["right_shoulder2"].reset_position(0, 0)
        # self.jdict["right_elbow"].reset_position(0, 0)
        #
        #
        #
        #
        # self.motor_names = ["right_shoulder1", "right_shoulder2", "right_elbow"]
        # self.motor_power = [75, 75, 75]
        # self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
        # self.motor_power += [75, 75, 75]
        # self.motors = [self.jdict[n] for n in self.motor_names]


    def calc_state(self):
        obs = np.zeros(12)
        return obs
        # link_position = np.asarray([self.parts[j].get_position() for j in self.select_links if j in self.parts])
        # velocity = link_position-self.last_link_position
        # obs = np.concatenate((np.asarray(link_position.flatten()), np.asarray(velocity.flatten())))
        # self.last_link_position = link_position
        # return obs

    def apply_action(self, a):
        return 0

        # #control arm
        # assert (np.isfinite(a).all())
        # self.jdict["right_shoulder1"].set_position(0)
        # self.jdict["right_shoulder2"].set_position(0)
        # self.jdict["right_elbow"].set_position(0)
        # # scale
        #
        # for i in range((self.action_space.shape)[0]):
        #     scale = self.jdict[self.select_joints[i]].max_velocity
        #     action = a[i] * (scale) / 2
        #     # action = a[i] * (high - low) / 2 + (high + low) / 2
        #     self.jdict[self.select_joints[i]].set_velocity(action)


    def alive_bonus(self, z, pitch):
        return +2 if z > 0.78 else -1  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying


