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
        self.index = 30
        self.reset_flag=True
        self.play_end = False
        # self.joint_queue = self.joint_queue_list[0]

        self.update_joint_queue()
        self.update_joint_queue()

    def write_optimized_result(self,name, angles):
        self.joints[name] = angles


    def update_joint_queue(self):
        # print("self.index: ", self.index)
        # print("current fild index", self.file_index)
        if self.index > self.len_list[self.file_index]-1:
            self.file_index = np.random.choice(range(len(self.len_list)))
            self.index = np.random.randint(low=0, high=int(self.len_list[self.file_index]/2))
            self.reset_flag=True
            self.play_end = True
        else:
            self.reset_flag=False
        self.joints = self.file_list[self.file_index][self.index]
        self.index += 1






class URDFHumanoid(robot_bases.URDFBasedRobot):
    self_collision = True
    foot_list = ["right_foot", "left_foot"]  # "left_hand", "right_hand"

    def __init__(self, max_obs_dist_threshold, obs_dim=27, load=False, test=False):
        self.power = 0.41
        self.camera_x = 0

        self.load = load
        self.test = test

        self.left_moveable_joints = ["LShoulderSY", "LShoulderSZ",
                                'LShoulderX', 'LShoulderY','LShoulderZ',
                                'LElbowX','LElbowZ']

        self.right_moveable_joints = ["RShoulderSY", "RShoulderSZ",
                                'RShoulderX', 'RShoulderY', 'RShoulderZ',
                                'RElbowX', 'RElbowZ']

        self.obs_links = ["ShoulderLeft","ElbowLeft","WristLeft","ShoulderRight","ElbowRight","WristRight"]

        self.human_base_link = "SpineBase"
        if self.load and self.test:
            print("use test recorded data")
            self.human_file = FileHuman(file='/home/xuan/demos/human_test_', index_range=range(3, 7))
            # self.human_file = FileHuman(file='/home/xuan/demos/human_test1204_', index_range=range(1, 3))
            # self.human_file = FileHuman(file='/home/xuan/demos/human_test_', index_range=[1])
        elif self.load:
            print("use recorded data")
            self.human_file = FileHuman(file='/home/xuan/demos/human_data_', index_range=range(2,9))

        else:
            print("use data from camera")
            self.human_model = HumanModel()

        trans_mat = pyquaternion.Quaternion([0.423, 0.547, 0.565, 0.450]).transformation_matrix
        # trans_mat[:3, 3] = [-1.305, -0.290, 0.656]
        trans_mat[:3, 3] = [-0.52, -0.480, 0.656]

        # trans_mat[:3, 3] = [-0.85, -0.75, 0.656]
        # trans_mat[:3, 3] = [-2, -2, 2]
        self.trans_matrix = trans_mat

        high = np.inf * np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)

        self.robot_name = 'humanoid'

        self.max_obs_dist_threshold = max_obs_dist_threshold
        self.last_state = {"elbow": np.ones(3) + self.max_obs_dist_threshold,
                     "arm": np.ones(3) + self.max_obs_dist_threshold,
                     "hand": np.ones(3) + self.max_obs_dist_threshold}
        self.arm_id = None

        self.human_iter = 0

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

        if self.test:
            self.apply_action(0)

        else:
            if rob_goal is not None:
                self.rob_goal=rob_goal
                self.human_base_reset(rob_goal)

            else:
                self.robot_specific_reset(self._p, base_position = [0, -0.8, 0.2], base_rotation=base_rotation)
                self.human_base_velocity=np.zeros(3)

        s = self.calc_state(
        )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
        return s


    def trans_point(self,p):
        point=np.zeros(4)
        point[:3] = p
        point[3] = 1
        p_new = np.matmul(self.trans_matrix, point)[:3]
        return p_new

    def trans_pose(self,pos, ori):
        # ori: xyzw

        point=np.zeros(4)
        point[:3] = pos
        point[3] = 1
        mat0 = pyquaternion.Quaternion(ori[3],ori[0],ori[1],ori[2]).transformation_matrix
        mat0[:,3]=point

        trans_new = np.matmul(self.trans_matrix, mat0)
        pos = trans_new[:3,3]
        rot = pyquaternion.Quaternion(matrix=trans_new) #w,x,y,z
        # rot_q =

        return pos,[rot[1],rot[2],rot[3],rot[0]]



    def human_base_reset(self, rob_goal):
        '''
        add moving human base
        '''
        # set human goal
        r = np.linalg.norm(rob_goal[:2]) + np.random.uniform(0.22, 0.35)
        if r < 0.7:
            r = 0.7
        human_goal = rob_goal.copy()
        human_goal[:2] = r * rob_goal[:2] / np.linalg.norm(rob_goal[:2])
        human_goal[2] =  0+np.random.uniform(-0.2, 0.25)

        y = [0, 0, 1]
        z = [-human_goal[0], -human_goal[1], 0] / np.linalg.norm([-human_goal[0], -human_goal[1], 0])
        x = np.cross(y, z)
        rotation = np.linalg.inv(np.asarray([x, y, z]))
        rot_q = pyquaternion.Quaternion(matrix=rotation)

        #set human start position
        theta = np.arctan2(human_goal[1], human_goal[0])
        self.r = r
        self.theta_range = [theta+0.8, theta-0.8]
        self.v_theta = 0.02
        self.reach_flag= True

        self.robot_specific_reset(self._p, base_position=human_goal, base_rotation=[rot_q[1], rot_q[2], rot_q[3], rot_q[0]])




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


    def calc_state(self):
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

            #1.-----move human base--------
            if self.test:
                # if self.human_file.play_end:
                #     print("!!!!!!!!!!!!!!!!end test!!!!!")
                #     return
                #testing using real human data
                base = joints['SpineBase']
                pos, ori = self.trans_pose(pos = base[:3], ori = base[3:])
                self.robot_specific_reset(self._p, base_position=pos,
                                          base_rotation=ori)


            else:
                #training
                pos, orn = self._p.getBasePositionAndOrientation(self.human_id)
                theta = np.arctan2(pos[1], pos[0])


                if theta<self.theta_range[1]:
                    self.reach_flag=True

                elif theta>self.theta_range[0]:
                    self.reach_flag=False

                if self.reach_flag:
                    v = self.v_theta+np.random.uniform(-0.01,0.01)
                else:
                    v= -self.v_theta+np.random.uniform(-0.01,0.01)


                t = theta+v
                pos_new=np.zeros(3)
                pos_new[0] = self.r*np.cos(t)
                pos_new[1] = self.r * np.sin(t)
                pos_new[2] = pos[2]

                y = [0, 0, 1]
                z = [-pos_new[0], -pos_new[1], 0] / np.linalg.norm([-pos_new[0], -pos_new[1], 0])
                x = np.cross(y, z)
                rotation = np.linalg.inv(np.asarray([x, y, z]))
                rot_q = pyquaternion.Quaternion(matrix=rotation)

                self.robot_specific_reset(self._p, base_position=pos_new,
                                          base_rotation=[rot_q[1], rot_q[2], rot_q[3], rot_q[0]])

            # if abs(pos[0]) > hrange[0] or abs(pos[1]) > hrange[1] \
            #         or abs(pos[2]) > hrange[2]:
            #     self.human_base_reset(self.rob_goal)
            # else:
            #     self.robot_specific_reset(self._p, base_position=pos+self.human_base_velocity,
            #                       base_rotation=orn)

            #2. set arm position
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
            # print(joints)

            try:
                base = joints['SpineBase']
                pos, ori = self.trans_pose(pos=base[:3], ori=base[3:])
            except:
                pos = [2,0,0]
                ori = [0.7068252, 0, 0, 0.7073883]


            self.robot_specific_reset(self._p, base_position=pos,
                                          base_rotation=ori)

            if joints is {}:
                xl = np.zeros(7)
                xr = np.zeros(7)
            else:
                xl = self.optimize_joint(joints, "Left", disp=disp, reset_flag=False)
                xr = self.optimize_joint(joints, "Right", disp=disp, reset_flag=False)
            if xl is False or xr is False:
                xl = np.zeros(7)
                xr = np.zeros(7)

        #set

        # print("xl is: ", xl)
        for i in range(len(self.left_moveable_joints)):
            self.jdict[self.left_moveable_joints[i]].reset_position(xl[i],0)
        for i in range(len(self.right_moveable_joints)):
            self.jdict[self.right_moveable_joints[i]].reset_position(xr[i],0)


        if self.load:
            self.human_file.update_joint_queue()
            # self.human_iter+=1
            # if self.human_iter % 2 == 0:
            #     self.human_file.update_joint_queue()

        return 0


    def alive_bonus(self, z, pitch):
        return +2 if z > 0.78 else -1  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying




class RealHumanoid(robot_bases.MJCFBasedRobot):
    self_collision = True

    def __init__(self,max_obs_dist_threshold, obs_dim=27):
        self.power = 0.41
        self.camera_x = 0

        high = np.inf * np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        self.human_model = HumanModel()

        trans_mat = pyquaternion.Quaternion([0.423, 0.547, 0.565, 0.450]).transformation_matrix
        trans_mat[:3, 3] = [-1.305, -0.290, 0.656]
        # trans_mat[:3, 3] = [-1.305, -0.290, 0.606]
        self.trans_matrix = trans_mat
        self.robot_name = 'humanoid'
        self.obs_links = ["ShoulderLeft", "ElbowLeft", "WristLeft", "ShoulderRight", "ElbowRight", "WristRight"]
        self.max_obs_dist_threshold = max_obs_dist_threshold



    def reset(self, bullet_client, client_id=None, base_rotation=None):
        self._p = bullet_client

        s = self.calc_state(
        )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use


        return s

    def robot_specific_reset(self, bullet_client):
        # WalkerBase.robot_specific_reset(self, bullet_client)
        self._p = bullet_client



    def calc_state(self, draw=False):
        obs = []
        joints = self.human_model.joints
        print("joints", joints)
        for link_name in self.obs_links:
            try:
                pos = self.trans_point(joints[link_name][:3])
                # pos = joints[link_name][:3]
            except:
                pos = [2,0,0]

            # print("pos", pos)

            # pos = [2, 2, 2]
            obs.append(pos)

        # print("human obs:", obs)

        self.obs_links = ["ShoulderLeft", "ElbowLeft", "WristLeft", "ShoulderRight", "ElbowRight", "WristRight"]
        if draw:
            self._p.addUserDebugLine(obs[0],obs[1], lineColorRGB=[0, 0, 1], lineWidth=5,
                                     lifeTime=0.5)  # ！！！！耗时大户，画一根0.017s
            self._p.addUserDebugLine(obs[1], obs[2], lineColorRGB=[0, 0, 1], lineWidth=5,
                                     lifeTime=0.5)  # ！！！！耗时大户，画一根0.017s
            self._p.addUserDebugLine(obs[0], obs[3], lineColorRGB=[0, 0, 1], lineWidth=5,
                                     lifeTime=0.5)  # ！！！！耗时大户，画一根0.017s
            self._p.addUserDebugLine(obs[3], obs[4], lineColorRGB=[0, 0, 1], lineWidth=5,
                                     lifeTime=0.5)  # ！！！！耗时大户，画一根0.017s
            self._p.addUserDebugLine(obs[4], obs[5], lineColorRGB=[0, 0, 1], lineWidth=5,
                                     lifeTime=0.5)  # ！！！！耗时大户，画一根0.017s

        return obs


    def trans_point(self,p):
        point=np.zeros(4)
        point[:3] = p
        point[3] = 1
        p_new = np.matmul(self.trans_matrix, point)[:3]
        return p_new

    def apply_action(self, a):
        return 0


    def alive_bonus(self, z, pitch):
        return +2 if z > 0.78 else -1  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying

