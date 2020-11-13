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
    def __init__(self, file, index_range):
        self.file_index_range = index_range
        self.file_list = []
        self.len_list = []
        for i in index_range:
            file_name = file+str(i)+'.pkl'
            try:
                with open(file_name, 'rb') as handle:
                    demo = pickle.load(handle)
                    self.file_list.append(demo)
                    self.len_list.append(len(demo))
                print("load human data successfully")
            except:
                print("!!!!!!!!!!!!!!failed to load data !!!!!!!")
                exit()

        self.file_index =0
        self.index = 0
        self.reset_flag=True
        # self.joint_queue = self.joint_queue_list[0]

        self.update_joint_queue()
        self.update_joint_queue()

    def write_optimized_result(self,name, angles):
        self.joints[name] = angles


    def update_joint_queue(self):
        # print("self.index: ", self.index)
        if self.index > self.len_list[self.file_index]-1:
            self.file_index = np.random.choice(range(len(self.len_list)))
            self.index = np.random.randint(low=0, high=int(self.len_list[self.file_index]/2))
            self.reset_flag=True
        else:
            self.reset_flag=False
        self.joints = self.file_list[self.file_index][self.index]
        self.index += 1



class URDFHumanoid(robot_bases.URDFBasedRobot):
    self_collision = True
    foot_list = ["right_foot", "left_foot"]  # "left_hand", "right_hand"

    def __init__(self,max_obs_dist_threshold, obs_dim=27, load=False):
        self.power = 0.41
        self.camera_x = 0

        self.load =load

        self.left_moveable_joints = ["LShoulderSY", "LShoulderSZ",
                                'LShoulderX', 'LShoulderY','LShoulderZ',
                                'LElbowX','LElbowZ']

        self.right_moveable_joints = ["RShoulderSY", "RShoulderSZ",
                                'RShoulderX', 'RShoulderY', 'RShoulderZ',
                                'RElbowX', 'RElbowZ']

        self.obs_links = ["ShoulderLeft","ElbowLeft","WristLeft","ShoulderRight","ElbowRight","WristRight"]

        self.human_base_link = "SpineBase"
        if self.load:
            print("use recorded data")
            # self.human_file = FileHuman(file = '/home/xuan/demos/human_data_normal_py3.pkl')
            self.human_file = FileHuman(file='/home/xuan/demos/human_data_', index_range=range(2,9))

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


    def reset(self, bullet_client, client_id, base_rotation, rob_goal=None):

        self._p = bullet_client
        self.client_id = client_id

        self.ordered_joints = []

        # if self.load:
        #     print("use recorded data")
        # else:
        #     print("use data from camera")
        # print(os.path.join(os.path.dirname(__file__), "data", self.model_urdf))

        if self.jdict is None:
            if self.self_collision:
                self.human_id = self._p.loadURDF(os.path.join(assets.getDataPath(), self.model_urdf),
                                     basePosition=self.basePosition,
                                     baseOrientation=self.baseOrientation,
                                     useFixedBase=self.fixed_base)

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


        if rob_goal is not None:
            r = np.linalg.norm(rob_goal)+np.random.uniform(0.3, 0.5)

            if r< 0.7:
                r = 0.7

            bp = r*rob_goal/np.linalg.norm(rob_goal)+\
                 [np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1),0]
            bp[2] -=np.random.uniform(0.4,0.6)

            y = [0,0,1]
            z = [-bp[0],-bp[1],0]/np.linalg.norm([-bp[0],-bp[1],0])
            x = np.cross(y,z)

            rotation = np.linalg.inv(np.asarray([x,y,z]))
            # print("rotation", rotation)

            rot_q = pyquaternion.Quaternion(matrix=rotation)

            self.robot_specific_reset(self._p, base_position=bp, base_rotation=[rot_q[1],rot_q[2],rot_q[3],rot_q[0]])
        else:
            self.robot_specific_reset(self._p, base_position = [0, -0.8, 0.2], base_rotation=base_rotation)


        s = self.calc_state(
        )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
        return s




    def robot_specific_reset(self, bullet_client, base_position, base_rotation):
        # WalkerBase.robot_specific_reset(self, bullet_client)
        self._p = bullet_client
        self._p.resetBasePositionAndOrientation(self.human_id,
                                                posObj = base_position, ornObj = base_rotation)
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


    def optimize_joint(self, joints, arm, disp=False, reset_flag=False):

        # x0 = np.zeros(7)
        if arm == "Left":
            if reset_flag:
                x0 = np.zeros(7)
            else:
                x0 = np.asarray([self.jdict[j].get_position() for j in self.left_moveable_joints])
            link_name = ["ShoulderLeft","ElbowLeft","WristLeft"]
            func_shoulder = human_optimization.left_shoulder
            func_elbow = human_optimization.left_elbow
        else:
            if reset_flag:
                x0 = np.zeros(7)
            else:
                x0 = np.asarray([self.jdict[j].get_position() for j in self.right_moveable_joints])
            link_name = ["ShoulderRight", "ElbowRight", "WristRight"]
            func_shoulder = human_optimization.right_shoulder
            func_elbow = human_optimization.right_elbow


        try:
            js = self.calculate_relative_trans(joints["SpineBase"],  joints[link_name[0]]) #elbow -> spinbase
            inputP_s = js[:3]

            je = self.calculate_relative_trans(joints["SpineBase"],  joints[link_name[1]]) #elbow -> spinbase
            # print("translation j3 is: {}".format(je[:3]))
            inputP_e = je[:3]

            jw = self.calculate_relative_trans(joints["SpineBase"], joints[link_name[2]])
            # print("translation j3 is: {}".format(jw[:3]))
            inputP_w = jw[:3]

        except:
            return False


        res1 = minimize(func_shoulder, x0[:-2], args=(inputP_s,inputP_e),
                           method='trust-constr', jac="2-point", hess=SR1(),
                           options={'gtol': 0.001, 'disp': False})
        theta_s = res1.x

        res2 = minimize(func_elbow, x0[-2:], args=(theta_s, inputP_w),
                       method='trust-constr', jac="2-point", hess=SR1(),
                       options={'gtol': 0.001, 'disp': False})
        theta_w = res2.x

        x = np.concatenate([theta_s, theta_w])


        if disp:
            print("result of theta is: ", x)
            # human_optimization.left_shoulder(x, inputP_s, inputP_e, inputP_w, disp=True)
            # human_optimization.left_wrist(x, inputP_s, inputP_e, inputP_w, disp=True)
            spine_base_sim = self.parts['SpineBase'].get_pose()
            points= []
            points.append(spine_base_sim[:3])
            points.append(self.trans_joint_to_sim(self.calculate_relative_trans(joints["SpineBase"], joints["SpineShoulder"])))
            points.append(self.trans_joint_to_sim(self.calculate_relative_trans(joints["SpineBase"], joints[link_name[0]])))
            points.append(self.trans_joint_to_sim(self.calculate_relative_trans(joints["SpineBase"], joints[link_name[1]])))
            points.append(self.trans_joint_to_sim(self.calculate_relative_trans(joints["SpineBase"], joints[link_name[2]])))

            for i in range(len(points)-1):
                self._p.addUserDebugLine(points[i],points[i+1],lineColorRGB=[0.9,0.1,0.1], lineWidth=2 , lifeTime=1)
        return x


    def trans_joint_to_sim(self, trans2):
        w_sb = self.parts['SpineBase'].get_pose()

        point = self._p.multiplyTransforms(positionA = w_sb[:3], orientationA = w_sb[3:],
                                   positionB = trans2[:3], orientationB = trans2[3:])[0]


        return point

    def calc_state(self, draw=True):
        # if self.load:
        #     joints = self.human_file.joints
        #     reset_flag = self.human_file.reset_flag
        #
        #     try:
        #         xl = joints["LeftAngle"]
        #         xr = joints["RightAngle"]
        #         # print("use calculated result")
        #     except:
        #         xl = self.optimize_joint(joints, "Left", disp=False, reset_flag=reset_flag)
        #         xr = self.optimize_joint(joints, "Right", disp=False, reset_flag=reset_flag)
        #         self.human_file.write_optimized_result(name="LeftAngle", angles=xl)
        #         self.human_file.write_optimized_result(name="RightAngle", angles=xr)
        #         # print("calculating result")
        #
        # else:
        #     joints = self.human_model.joints
        #     xl = self.optimize_joint(joints, "Left", disp=False, reset_flag=False)
        #     xr = self.optimize_joint(joints, "Right", disp=False, reset_flag=False)
        #



        # #set
        # for i in range(len(self.left_moveable_joints)):
        #     self.jdict[self.left_moveable_joints[i]].reset_position(xl[i],0)
        # for i in range(len(self.right_moveable_joints)):
        #     self.jdict[self.right_moveable_joints[i]].reset_position(xr[i],0)

        #
        # if self.load:
        #     self.human_file.update_joint_queue()

        link_positions = [self.parts[l].get_position() for l in self.obs_links]

        obs = link_positions
        return obs




    def trans_point(self,p):
        point=np.zeros(4)
        point[:3] = p
        point[3] = 1
        p_new = np.matmul(self.trans_matrix, point)[:3]
        return p_new

    def apply_action(self, a):

        disp = False

        if self.load:
            joints = self.human_file.joints
            reset_flag = self.human_file.reset_flag

            # xl = self.optimize_joint(joints, "Left", disp=disp, reset_flag=reset_flag)
            # xr = self.optimize_joint(joints, "Right", disp=disp, reset_flag=reset_flag)
            # self.human_file.write_optimized_result(name="LeftAngle", angles=xl)
            # self.human_file.write_optimized_result(name="LeftAngle", angles=xr)

            try:
                xl = joints["LeftAngle"]
                xr = joints["RightAngle"]
                # print("use calculated result")
            except:
                xl = self.optimize_joint(joints, "Left", disp=disp, reset_flag=reset_flag)
                xr = self.optimize_joint(joints, "Right", disp=disp, reset_flag=reset_flag)
                self.human_file.write_optimized_result(name="LeftAngle", angles=xl)
                self.human_file.write_optimized_result(name="RightAngle", angles=xr)
                # print("calculating result")

        else:
            joints = self.human_model.joints
            xl = self.optimize_joint(joints, "Left", disp=disp, reset_flag=False)
            xr = self.optimize_joint(joints, "Right", disp=disp, reset_flag=False)

        #set

        # print("xl is: ", xl)
        for i in range(len(self.left_moveable_joints)):
            self.jdict[self.left_moveable_joints[i]].reset_position(xl[i],0)
        for i in range(len(self.right_moveable_joints)):
            self.jdict[self.right_moveable_joints[i]].reset_position(xr[i],0)


        if self.load:
            self.human_file.update_joint_queue()

        return 0


    def alive_bonus(self, z, pitch):
        return +2 if z > 0.78 else -1  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying




