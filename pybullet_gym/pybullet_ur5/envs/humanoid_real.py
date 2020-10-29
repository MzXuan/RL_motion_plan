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
import time
import pickle

class FileHuman(object):
    def __init__(self, file):
        try:
            with open(file, 'rb') as handle:
                self.joint_queue_list = pickle.load(handle)
            print("load human data successfully")
            self.data_length = len(self.joint_queue_list)
        except:
            print("!!!!!!!!!!!!!!fail to load data !!!!!!!")
            exit()

        self.index = 0
        # self.joint_queue = self.joint_queue_list[0]

        self.update_joint_queue()


    def update_joint_queue(self):
        print("self.index: ", self.index)
        if self.index > self.data_length-1:
            self.index = np.random.randint(low=0, high=self.data_length - 1)
        self.joint_queue = self.joint_queue_list[self.index]

        self.index += 1


class RealHumanoid(robot_bases.MJCFBasedRobot):
    self_collision = True
    foot_list = ["right_foot", "left_foot"]  # "left_hand", "right_hand"

    def __init__(self,max_obs_dist_threshold, obs_dim=27, load=False):
        self.power = 0.41
        self.camera_x = 0

        self.load =load
        if self.load:
            self.file_human = FileHuman(file = '/home/xuan/demos/human_data_1.pkl')




        high = np.inf * np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        self.human_model = HumanModel()

        trans_mat =  pyquaternion.Quaternion([0.415, 0.535, 0.577, 0.457]).transformation_matrix
        trans_mat[:3,3]=[-1.301, -0.295, 0.652]
        self.trans_matrix = trans_mat

        self.robot_name = 'humanoid'

        self.max_obs_dist_threshold = max_obs_dist_threshold
        self.last_state = {"elbow": np.ones(3) + self.max_obs_dist_threshold,
                     "arm": np.ones(3) + self.max_obs_dist_threshold,
                     "hand": np.ones(3) + self.max_obs_dist_threshold}
        self.arm_id = None

    def reset(self, bullet_client):
        self._p = bullet_client
        # if self.arm_id is None:
        #     self.arm_id = self._p.loadURDF(os.path.join(assets.getDataPath(),
        #                                     "scenes_data", "cylinder/cylinder.urdf"),
        #                        [1, 1, 1], [0.000000, 0.000000, 0.0, 0.1], useFixedBase=True)
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


    def calc_one_state(self, name_e, name_h, draw=True):

        if self.load:

            # try:
            elbow = self.file_human.joint_queue[name_e]
            hand = self.file_human.joint_queue[name_h]

            elbow_trans = [self.trans_point(p) for p in elbow]
            hand_trans = [self.trans_point(p) for p in hand]

            # except:
            #     elbow_trans = np.ones((3, 3)) + self.max_obs_dist_threshold
            #     hand_trans = np.ones((3, 3)) + self.max_obs_dist_threshold


        else:
            try:
                elbow = self.human_model.joint_queue[name_e]
                hand = self.human_model.joint_queue[name_h]


                elbow_trans = [self.trans_point(p) for p in elbow]
                hand_trans = [self.trans_point(p) for p in hand]

            except:
                elbow_trans = np.ones((3,3))+ self.max_obs_dist_threshold
                hand_trans = np.ones((3,3))+ self.max_obs_dist_threshold



        obs={"current":[elbow_trans[0], (elbow_trans[0]+hand_trans[0])/2, hand_trans[0]],
             "next":[elbow_trans[1], (elbow_trans[1]+hand_trans[1])/2, hand_trans[1]],
             "next2":[elbow_trans[2], (elbow_trans[2]+hand_trans[2])/2, hand_trans[2]]}

            # print("elbow trans {} and hand_trans {}".format(elbow_trans[2], hand_trans[2]))

        if draw:
            self._p.addUserDebugLine(elbow_trans[1], hand_trans[1], lineColorRGB=[0, 0, 1], lineWidth=10,
                                     lifeTime=0.5)  # ！！！！耗时大户，画一根0.017s


        #--------------- for moving object
        # hand_raw =  hand_trans[1]
        # elbow_raw = elbow_trans[1]
        # center = (hand_raw + elbow_raw) / 2
        #
        # if np.linalg.norm([hand_raw - center])!=0:
        #     hand_trans = (hand_raw - center) / np.linalg.norm([hand_raw - center])
        # else:
        #     hand_trans = [0, 0, 1]
        #     center = [1,1,1]
        #
        # alpha = -np.arcsin(hand_trans[1])
        # beta = np.arcsin(hand_trans[0] / np.cos(alpha))
        # next_ori = self._p.getQuaternionFromEuler([alpha, beta, 0])
        # next_ori = self._p.getQuaternionFromEuler([alpha, beta, 0])
        # self._p.resetBasePositionAndOrientation(bodyUniqueId=self.arm_id, posObj=center, ornObj=next_ori)
        # self._p.addUserDebugLine(hand_raw , elbow_raw,
        #                          lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=0.5)


        return obs


    def calc_state(self, draw=True):
        obs = [self.calc_one_state("ElbowLeft", "HandLeft", draw=draw), self.calc_one_state("ElbowRight", "HandRight",draw=draw)]
        return obs


    def trans_point(self,p):
        point=np.zeros(4)
        point[:3] = p
        point[3] = 1
        p_new = np.matmul(self.trans_matrix, point)[:3]
        return p_new

    def apply_action(self, a):
        if self.load:
            self.file_human.update_joint_queue()

        return 0


    def alive_bonus(self, z, pitch):
        return +2 if z > 0.78 else -1  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying


