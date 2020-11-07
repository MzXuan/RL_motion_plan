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
from scipy.optimize import minimize
from scipy.optimize import SR1

import human_optimization



class FileHuman(object):
    def __init__(self, file):
        try:
            with open(file, 'rb') as handle:
                self.joint_list = pickle.load(handle)
            print("load human data successfully")
            self.data_length = len(self.joint_list)
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
        self.joints = self.joint_list[self.index]

        self.index += 1


class URDFHumanoid(robot_bases.URDFBasedRobot):
    self_collision = True
    foot_list = ["right_foot", "left_foot"]  # "left_hand", "right_hand"

    def __init__(self,max_obs_dist_threshold, obs_dim=27, load=False):
        self.power = 0.41
        self.camera_x = 0

        self.load =load

        # self.translation_pairs = [['ShoulderLeft', 'ElbowLeft','LShoulder'],['ElbowLeft', 'WristLeft', 'LElbow'],
        #                           ['ShoulderRight', 'ElbowRight', 'RShoulder'], ['ElbowRight', 'WristRight', 'RElbow']]
        # self.translation_pairs = [['ShoulderLeft', 'LShoulder'],['ElbowLeft', 'LElbow']]
        self.translation_pairs = [['ShoulderLeft', 'LShoulder']]
        self.moveable_joints = ["ShoulderSY", "ShoulderSZ",
                                'LShoulderX', 'LShoulderY','LShoulderZ',
                                'LElbowX','LElbowZ']
        self.human_base_link = "SpineBase"
        if self.load:
            print("use recorded data")
            self.human_file = FileHuman(file = '/home/xuan/demos/human_data_normal_py3.pkl')
            # self.human_file = FileHuman(file='/home/xuan/demos/human_data_1.pkl')

        else:
            print("use data from camera")
            self.human_model = HumanModel()


        high = np.inf * np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        trans_mat =  pyquaternion.Quaternion([0.415, 0.535, 0.577, 0.457]).transformation_matrix
        trans_mat[:3,3]=[-1.301, -0.295, 0.652]
        self.trans_matrix = trans_mat

        self.robot_name = 'humanoid'

        self.max_obs_dist_threshold = max_obs_dist_threshold
        self.last_state = {"elbow": np.ones(3) + self.max_obs_dist_threshold,
                     "arm": np.ones(3) + self.max_obs_dist_threshold,
                     "hand": np.ones(3) + self.max_obs_dist_threshold}
        self.arm_id = None



        super(URDFHumanoid, self).__init__(
            'kinect_human/upper_body.urdf', "human", action_dim=0, obs_dim=obs_dim, fixed_base=1,
            self_collision=True)


    def reset(self, bullet_client, client_id, base_rotation):
        self._p = bullet_client
        self.client_id = client_id

        self.ordered_joints = []

        if self.load:
            print("use recorded data")
        else:
            print("use data from camera")
        # print(os.path.join(os.path.dirname(__file__), "data", self.model_urdf))

        if self.jdict is None:
            if self.self_collision:
                self.human_id = self._p.loadURDF(os.path.join(assets.getDataPath(), self.model_urdf),
                                     basePosition=self.basePosition,
                                     baseOrientation=self.baseOrientation,
                                     useFixedBase=self.fixed_base,
                                     flags=pybullet.URDF_USE_SELF_COLLISION)

                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p,
                    self.human_id)

            else:
                self.human_id = self._p.loadURDF(os.path.join(assets.getDataPath(), self.model_urdf),
                                     basePosition=self.basePosition,
                                     baseOrientation=self.baseOrientation,
                                     useFixedBase=self.fixed_base)
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p, self.human_id)
        if self.robot_specific_reset(self._p, base_rotation) is False:
            return False


        self.get_initial_trans()




        s = self.calc_state(
        )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
        return s


    def get_initial_trans(self):
        self.initial_trans = {}
        for j in self.jdict:
            self.jdict[j].reset_position(0,0)
        for p in self.translation_pairs:
            p_pose = self.parts[p[0]].get_pose() #x,y,zxyzw
            spin_pose = self.parts[self.human_base_link].get_pose()
            self.initial_trans [p[0]] = self.calculate_relative_trans(spin_pose, self.parts[p[0]].get_pose())


    def robot_specific_reset(self, bullet_client, base_rotation):
        # WalkerBase.robot_specific_reset(self, bullet_client)
        self._p = bullet_client
        self._p.resetBasePositionAndOrientation(self.human_id,
                                                posObj = [0, -0.8, 0.25], ornObj = base_rotation)
        # self._p.reset

    def calculate_relative_trans(self, j1, j2): #both relative to world
        # j1.inv * j2, 2相对于1的变换
        j1_inv = self._p.invertTransform(position=j1[:3], orientation=j1[3:])  # tuple [0]: pos; [1]: orientation
        j3 = self._p.multiplyTransforms(positionA=np.asarray(j1_inv[0]), orientationA=np.asarray(j1_inv[1]),
                                        positionB=j2[:3], orientationB=j2[3:])

        # j3 = self._p.multiplyTransforms(positionA=j2[:3], orientationA=j2[3:],
        #                                 positionB=np.asarray(j1_inv[0]), orientationB=np.asarray(j1_inv[1]))

        j3 = np.concatenate([np.asarray(j3[0]), np.asarray(j3[1])])
        return j3

    def calc_one_state(self, joints, pair, draw=True):
        # # ----------------test frame-------#
        # try:
        #     j1 = joints["SpineBase"]
        #     j2 = joints["ElbowLeft"]
        #     j3 = self.calculate_relative_trans(j1, j2) #elbow -> spinbase
        #     # print("euler angle for j3 {} is {}".format(pair[0], self._p.getEulerFromQuaternion(j3[3:])))
        #     print("translation j3 is: {}".format(j3[:3]))
        # except:
        #     pass
        #

        # #-----------read data-------------------
        #
        # try:
        #     j1 = joints[self.human_base_link]
        #     j2 = joints[pair[0]]
        #
        #     #step2, trans form initial trans
        #     #self.initialtrans inv *j3
        #     j3 = self.calculate_relative_trans(j1,j2) #pair[0] -> spinbase
        #     j4 = self.calculate_relative_trans(self.initial_trans[pair[0]], j3) #trans shoulder-> initial
        #
        #     print("pair[0]", pair[0])
        #     # print("initial trans", self.initial_trans[pair[0]])
        #
        #     euler = self._p.getEulerFromQuaternion(j4[3:])
        #     print("euler angle for j3 {} is {}".format(pair[0], self._p.getEulerFromQuaternion(j3[3:])))
        #     print("euler angle for j4 {} is {}".format(pair[0], self._p.getEulerFromQuaternion(j4[3:])))
        #
        #     self.jdict[pair[1] + "X"].reset_position(euler[0], 0)
        #     self.jdict[pair[1] + "Y"].reset_position(euler[1], 0)
        #     self.jdict[pair[1] + "Z"].reset_position(euler[2], 0)
        #
        #     spin_pose = self.parts[self.human_base_link].get_pose()
        #     current_relative = self.calculate_relative_trans(spin_pose, self.parts["ShoulderLeft"].get_pose())
        #     print("j3, human shoudler - > spineshoulder in real",
        #           self._p.getEulerFromQuaternion(j3[3:]))
        #     print("current relative  human shoudler - > spineshoulder in simulation:",
        #           self._p.getEulerFromQuaternion(current_relative[3:]))
        # except:
        #     pass
        #
        # #-------------------------------------------------------------------------------------------------#

        # except:
        #     elbow_trans = np.ones((3,3))+ self.max_obs_dist_threshold
        #     hand_trans = np.ones((3,3))+ self.max_obs_dist_threshold



        # obs={"current":[elbow_trans[0], (elbow_trans[0]+hand_trans[0])/2, hand_trans[0]],
        #      "next":[elbow_trans[1], (elbow_trans[1]+hand_trans[1])/2, hand_trans[1]],
        #      "next2":[elbow_trans[2], (elbow_trans[2]+hand_trans[2])/2, hand_trans[2]]}

        obs = {"current": np.zeros(6),
               "next": np.zeros(6),
               "next2": np.zeros(6)}

        # print("elbow trans {} and hand_trans {}".format(elbow_trans[2], hand_trans[2]))

        if draw:
            self._p.addUserDebugLine(elbow_trans[1], hand_trans[1], lineColorRGB=[0, 0, 1], lineWidth=10,
                                     lifeTime=0.5)  # ！！！！耗时大户，画一根0.017s

        return obs

    def optimize_joint(self, joints):
        x0 = np.asarray([self.jdict[j].get_position() for j in self.moveable_joints])

        # x0 = np.zeros(7)

        try:
            jsl = self.calculate_relative_trans(joints["SpineBase"],  joints["ShoulderLeft"]) #elbow -> spinbase
            print("translation j3 is: {}".format(jsl[:3]))
            inputP_sl = jsl[:3]

            jel = self.calculate_relative_trans(joints["SpineBase"],  joints["ElbowLeft"]) #elbow -> spinbase
            print("translation j3 is: {}".format(jel[:3]))
            inputP_el = jel[:3]

            jwl = self.calculate_relative_trans(joints["SpineBase"], joints["WristLeft"])
            print("translation j3 is: {}".format(jwl[:3]))
            inputP_wl = jwl[:3]

        except:
            return False



        res = minimize(human_optimization.left, x0, args=(inputP_sl,inputP_el, inputP_wl),
                       method='trust-constr', jac="2-point", hess=SR1(),
                       options={'gtol': 0.008, 'disp': True})

        x = res.x

        print("result of theta is: ", x)
        human_optimization.left(x,inputP_sl,inputP_el, inputP_wl, disp=True)


        #set joint angle
        for i in range(len(self.moveable_joints)):
            self.jdict[self.moveable_joints[i]].reset_position(x[i],0)


        # todo: draw real data
        spine_base_sim = self.parts['SpineBase'].get_pose()



        points= []

        points.append(spine_base_sim[:3])
        points.append(self.trans_joint_to_sim(self.calculate_relative_trans(joints["SpineBase"], joints["SpineShoulder"])))


        points.append(self.trans_joint_to_sim(self.calculate_relative_trans(joints["SpineBase"], joints["ShoulderLeft"])))

        points.append(self.trans_joint_to_sim(self.calculate_relative_trans(joints["SpineBase"], joints["ElbowLeft"])))
        points.append(self.trans_joint_to_sim(self.calculate_relative_trans(joints["SpineBase"], joints["WristLeft"])))
        for i in range(len(points)-1):
            self._p.addUserDebugLine(points[i],points[i+1],lineColorRGB=[0.9,0.1,0.1], lineWidth=2 , lifeTime=1)

        return x


    def trans_joint_to_sim(self, trans2):
        w_sb = self.parts['SpineBase'].get_pose()

        point = self._p.multiplyTransforms(positionA = w_sb[:3], orientationA = w_sb[3:],
                                   positionB = trans2[:3], orientationB = trans2[3:])[0]


        return point

    def calc_state(self, draw=True):
        if self.load:
            joints = self.human_file.joints

        else:
            joints = self.human_model.joints


        self.optimize_joint(joints)
        # for pair in self.translation_pairs:
        #     self.calc_one_state(joints, pair, draw=False)

        if self.load:
            self.human_file.update_joint_queue()


        return np.zeros(9)

        # obs = [self.calc_one_state("ElbowLeft", "HandLeft", draw=draw), self.calc_one_state("ElbowRight", "HandRight",draw=draw)]
        # return obs


    def trans_point(self,p):
        point=np.zeros(4)
        point[:3] = p
        point[3] = 1
        p_new = np.matmul(self.trans_matrix, point)[:3]
        return p_new

    def apply_action(self, a):

        # if self.load:
        #     self.file_human.update_joint_queue()

        return 0


    def alive_bonus(self, z, pitch):
        return +2 if z > 0.78 else -1  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying


