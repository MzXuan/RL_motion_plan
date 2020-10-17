import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet
from pybullet_utils import bullet_client
from pybullet_envs import env_bases, scene_stadium, scene_abstract

import random

from pybullet_gym.pybullet_ur5.envs.human_connection import HumanModel

import numpy as np
import assets
import robot_bases
import math
from scenes.stadium import StadiumScene, PlaneScene
import gym, gym.spaces, gym.utils
import pyquaternion



class RealHumanoid(robot_bases.MJCFBasedRobot):
    self_collision = True
    foot_list = ["right_foot", "left_foot"]  # "left_hand", "right_hand"

    def __init__(self,max_obs_dist_threshold, obs_dim=18, random_yaw=False, random_lean=False, ):
        self.power = 0.41
        self.camera_x = 0
        # self.select_joints = ["left_shoulder1", "left_shoulder2", "left_elbow"]
        # self.select_links = ["left_upper_arm", "left_lower_arm", "left_hand_true"]

        self.display_joints = ['HandLeft','ElbowLeft','ShoulderLeft','HandRight','ElbowRight','ShoulderRight']
        # self.state_joints = ['ShoulderLeft','ElbowLeft','HandLeft']
        self.state_joints = ['ElbowLeft', 'HandLeft']

        high = np.inf * np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        self.human_model = HumanModel()

        trans_mat =  pyquaternion.Quaternion([0.415, 0.535, 0.577, 0.457]).transformation_matrix
        trans_mat[:3,3]=[-1.301, -0.295, 0.652]
        self.trans_matrix = trans_mat

        self.robot_name = 'humanoid'

        self.max_obs_dist_threshold = max_obs_dist_threshold
        self.last_state = {"elbow": np.zeros(3) + self.max_obs_dist_threshold,
                     "arm": np.zeros(3) + self.max_obs_dist_threshold,
                     "hand": np.zeros(3) + self.max_obs_dist_threshold}

    def reset(self, bullet_client):
        self._p = bullet_client
        # self.leftarm1 = bullet_client.loadURDF(os.path.join(assets.getDataPath(), "scenes_data", "cylinder/cylinder.urdf"),
        #                        [0, 0, 0],
        #                        [0.000000, 0.000000, 0.0, 1])
        # self.leftarm2 = bullet_client.loadURDF(os.path.join(assets.getDataPath(), "scenes_data", "cylinder/cylinder.urdf"),
        #                                    [0, 0, 0],
        #                                    [0.000000, 0.000000, 0.0, 1])

        # self.robot_id = self.leftarm1

        s = self.calc_state(
        )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use


        return s

    def robot_specific_reset(self, bullet_client):
        # WalkerBase.robot_specific_reset(self, bullet_client)
        self._p = bullet_client

    def calc_state(self):
        try:
            elbow = self.human_model.joint_queue["ElbowLeft"]
            hand = self.human_model.joint_queue["HandLeft"]

            elbow_trans = [self.trans_point(p) for p in elbow]
            hand_trans = [self.trans_point(p) for p in hand]

        except:
            elbow_trans = np.zeros((3,3))+ + self.max_obs_dist_threshold
            hand_trans = np.zeros((3,3))+ + self.max_obs_dist_threshold


        obs={"current":[elbow_trans[0], hand_trans[0]],
             "last":[elbow_trans[1], hand_trans[1]],
             "next":[elbow_trans[2], hand_trans[2]]}

        self._p.addUserDebugLine(elbow_trans[2], hand_trans[2], lineColorRGB=[0, 0, 1], lineWidth=10,
                                 lifeTime=0.5)

        return obs



    # def calc_current_state(self):
    #     try:
    #         # current_position = [self.trans_point(self.human_model.joints[joint_name]) for joint_name in self.state_joints]
    #         elbow = self.trans_point(self.human_model.joints["ElbowLeft"])
    #         hand = self.trans_point(self.human_model.joints["HandLeft"])
    #
    #         state = {"elbow": elbow,
    #                  "arm": (elbow + hand) / 2,
    #                  "hand": hand}
    #
    #
    #     except:
    #         state = {"elbow": np.zeros(3) + self.max_obs_dist_threshold,
    #                  "arm": np.zeros(3) + self.max_obs_dist_threshold,
    #                  "hand": np.zeros(3) + self.max_obs_dist_threshold}
    #     return state
    #
    # def calc_state(self):
    #     current_state = self.calc_current_state()
    #
    #     obs = {"current":[current_state["elbow"], current_state["arm"], current_state["hand"]],
    #            "last": [self.last_state["elbow"], self.last_state["arm"],self.last_state["hand"]]
    #     }
    #     # print("human obs:", obs)
    #     self._p.addUserDebugLine(current_state["elbow"], current_state["hand"], lineColorRGB=[0, 0, 1], lineWidth=10,
    #                              lifeTime=1)
    #
    #     self.last_state = current_state
    #
    #     return obs




    def trans_point(self,p):
        point=np.zeros(4)
        point[:3] = p
        point[3] = 1
        p_new = np.matmul(self.trans_matrix, point)[:3]
        return p_new

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


