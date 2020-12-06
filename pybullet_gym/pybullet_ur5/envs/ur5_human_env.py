import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, currentdir)

from ur5eef import UR5EefRobot
from ur5_real import UR5RealRobot
from humanoid import URDFHumanoid, RealHumanoid
import assets
from scenes.stadium import StadiumScene, PlaneScene

import gym, gym.spaces, gym.utils, gym.utils.seeding
from gym.spaces import Tuple
import numpy as np
import random
import pybullet
from pybullet_utils import bullet_client
import time
import pickle
from pybullet_ur5.utils.ws_path_gen import WsPathGen
import colorsys

from .ur5_dynamic_reach_obs import UR5DynamicReachObsEnv





def load_demo():
    try:
        with open('/home/xuan/demos/demo5.pkl', 'rb') as handle:
            data = pickle.load(handle)
        print("load data successfully")
    except:
        print("fail to load data, stop")
        return

    print("length of data is: ", len(data))
    return data

def load_demo_lst():
    # file_lst = ['/home/xuan/demos/task_demo1.pkl','/home/xuan/demos/task_demo2.pkl','/home/xuan/demos/task_demo3.pkl']
    # file_lst = ['/home/xuan/demos/task_demo1.pkl', '/home/xuan/demos/task_demo3.pkl']
    # file_lst = ['/home/xuan/demos/task_demo1.pkl']

    file_lst = ['/home/xuan/demos/task_cloth_demo1.pkl', '/home/xuan/demos/task_cloth_demo2.pkl']
    # file_lst = ['/home/xuan/demos/task_cloth_demo2.pkl']

    # file_lst = ['/home/xuan/demos/task_task_demo1.pkl']

    # file_lst = ['/home/xuan/demos/plan_2.pkl']
    data_lst = []
    for f in file_lst:
        try:
            with open(f, 'rb') as handle:
                data_lst.append(pickle.load(handle))
            print("load data successfully")
        except:
            print("fail to load data, stop")
            return

    # print("length of data is: ", len(data))
    return data_lst


def move_to_start(ur5, start_joint):
    print(f"move to {start_joint}, please be careful")
    ur5.set_joint_position(start_joint, wait=True)
    print(f"success move to {start_joint}")


def move_along_path(ur5, ws_path_gen, dt=0.02):
    toolp,_,_ = ur5.get_tool_state()
    next_goal, next_vel,_ = ws_path_gen.next_goal(toolp, 0.01)

    ref_vel = (next_goal-toolp)
    print(f"current goal {toolp}, next goal {next_goal}, next_vel {next_vel}, ref_vel {ref_vel}")
    tool_vel = np.zeros(6)
    tool_vel[:3]=ref_vel
    ur5.set_tool_velocity(tool_vel)



class UR5HumanEnv(UR5DynamicReachObsEnv):
    def __init__(self, render=False, max_episode_steps=8000,
                 early_stop=True, distance_threshold = 0.25,
                 max_obs_dist = 0.8 ,dist_lowerlimit=0.02, dist_upperlimit=0.2,
                 reward_type="sparse",  use_rnn = True):
        super(UR5HumanEnv, self).__init__(render=render, max_episode_steps=max_episode_steps,
                 early_stop=early_stop, distance_threshold = distance_threshold,
                 max_obs_dist = max_obs_dist ,dist_lowerlimit=dist_lowerlimit, dist_upperlimit=dist_upperlimit,
                 reward_type=reward_type,  use_rnn = use_rnn)

        # --- following demo----
        self.demo_data_lst = load_demo_lst()
        self.demo_id = 0
        self.sphere_radius = 0.1
        self.last_collision = False
        #--------------------------

    def _set_agents(self, max_obs_dist):
        self.agents = [UR5EefRobot(dt=self.sim_dt * self.frame_skip),
                       URDFHumanoid(max_obs_dist, load=True, test=True)]

        # self.agents = [UR5EefRobot(dt=self.sim_dt * self.frame_skip),
        #                RealHumanoid(max_obs_dist)]


    def _special_rob_reset(self, start_joint):
        ar = self.agents[0].reset(self._p, client_id=self.physicsClientId, base_position=self.robot_base,
                                  base_rotation=[0, 0, 0, 1], joint_angle=start_joint)
        return ar

    def reset(self):

        self.last_obs_human = np.full(18,self.max_obs_dist_threshold+0.2)
        self.last_robot_joint = np.zeros(6)
        self.current_safe_dist = self._set_safe_distance()


        if (self.physicsClientId < 0):
            self.ownsPhysicsClient = True

            if self.isRender:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            else:
                self._p = bullet_client.BulletClient()

            self._p.setGravity(0, 0, -9.81)
            self._p.setTimeStep(self.sim_dt)

            self.physicsClientId = self._p._client
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)


        if self.scene is None:
            self.scene = self.create_single_player_scene(self._p)
        if not self.scene.multiplayer and self.ownsPhysicsClient:
            self.scene.episode_restart(self._p)

        self.camera_adjust()

        for a in self.agents:
            a.scene = self.scene


        self.frame = 0
        self.iter_num = 0

        self.robot_base = [0, 0, 0]

        #-----------------initial agent and set to start
        demo = self.demo_data_lst[self.demo_id]

        start_joint  = demo[0]['robjp']
        ar = self._special_rob_reset(start_joint)
        ah = self.agents[1].reset(self._p, client_id=self.physicsClientId,
                                  base_rotation=[0.0005629, 0.707388, 0.706825, 0.0005633])

        #------prepare path----------------
        demo = self.demo_data_lst[self.demo_id]
        path = [demo[i]['toolp'] for i in range(len(demo))]
        vel_path = [demo[i]['toolv'] for i in range(len(demo))]
        joint_path = [demo[i]['robjp'] for i in range(len(demo))]

        self.reference_path = joint_path.copy()

        self.ws_path_gen = WsPathGen(path, vel_path, joint_path, 0.12)

        #set goal from record demo-------------
        rob_eef = ar[:3]
        self.final_goal = np.array(demo[-1]['robjp'])
        next_goal = self._get_next_goal(rob_eef)
        self.eef_goal, self.goal, self.goal_indices = next_goal[0], next_goal[1], next_goal[2]


        self.demo_id+=1
        if self.demo_id>=len(self.demo_data_lst):
            self.demo_id=0
        #------------------------------------------

        self._p.stepSimulation()
        obs = self._get_obs()


        s = []
        s.append(ar)
        s.append(ah)
        self._p.resetBasePositionAndOrientation(self.goal_id, posObj=self.eef_goal[:3], ornObj=self.eef_goal[3:])

        return obs


    def step(self, action):
        self.iter_num += 1

        self.agents[0].apply_action(action)
        self.agents[1].apply_action(0)


        self.scene.global_step()
        obs = self._get_obs()
        done = False


        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.final_goal),
            'is_collision': self._contact_detection(),
            'min_dist': self.obs_min_dist,
            'safe_threshold': self.current_safe_dist,
            'joint_vel':obs["observation"][9:21]
        }

        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)


        if self.iter_num > self.max_episode_steps:
            done = True
        if self.early_stop:
            if info["is_success"] or info["is_collision"]:
                done = True

        self._p.resetBasePositionAndOrientation(self.goal_id, posObj=self.eef_goal[:3], ornObj=self.eef_goal[3:])

        return obs, reward, done, info


    def _get_next_goal(self, ur5_eef_position):
        try:
            eef_goal, goal, _, goal_indices = self.ws_path_gen.next_goal(ur5_eef_position, self.sphere_radius)
            eef_goal = np.concatenate([eef_goal,np.array([0,0,0,1])])
            return (eef_goal, np.array(goal), goal_indices)
        except:
            print("!!!!!!!!!!!!!!not exist self.ws_path_gen")
            return False

    def draw_path(self):
        color_list = [[0.8,0.8,0.0], [0.8,0.0,0.0],[0.0,0.0,0.8]]
        for d_idx, demo in enumerate(self.demo_data_lst):
            idxs = list(np.linspace(0, len(demo)-2, 20))
            for i in range(len(idxs)-1):
                self._p.addUserDebugLine(demo[int(idxs[i])]['toolp'],
                                         demo[int(idxs[i+1])]['toolp'],
                                         lineColorRGB=color_list[d_idx],lineWidth=3 )

    def set_sphere(self, r):
        self.sphere_radius = r

    # def update_r(self, obs_lst, q_lst, draw=True):
    #     # for obstacles in self.move_obstacles:
    #     #     self._p.addUserDebugLine(obstacles.pos_list[0], 5*(obstacles.pos_list[2]-obstacles.pos_list[0])+obstacles.pos_list[0],
    #     #                              lineColorRGB=[0.8, 0.8, 0.0], lineWidth=4)
    #
    #     self.q_thre = -0.1
    #     q_lst = np.asarray(q_lst)
    #     q_sum =np.sum(q_lst)
    #     print("!!!!!!q_sum!!!!!!!",q_sum)
    #     try:
    #         min_q = min(q_lst)
    #         # print("min_q", min_q)
    #     except:
    #         self.last_collision = False
    #         self.set_sphere(0.16)
    #         return 0
    #
    #     print("min_q", min_q)
    #
    #
    #     if min_q<self.q_thre:
    #         print("min_q", min_q)
    #         self.last_collision = True
    #         self.set_sphere(0.3)
    #         #normalize Q for color
    #         minimum_q = -2
    #         b = 0.6
    #         k=-b/minimum_q
    #         color_lst = q_lst*k+b
    #         # color_lst = (q_lst - min_q) / (max(q_lst) - min_q + 0.000001)
    #         # color_lst = (q_lst-min_q)/(max(q_lst)-min_q+0.000001)
    #
    #         if draw:
    #             self.draw_q(obs_lst, q_lst, color_lst)
    #         # for obs, q, c in zip(obs_lst, q_lst, color_lst):
    #         #     if q<-0.025:
    #         #         self._p.addUserDebugText(text = str(q)[1:7], textPosition=obs['observation'][:3],
    #         #                                  textSize=1.2, textColorRGB=colorsys.hsv_to_rgb(0.5-c/2, c+0.5, c), lifeTime=2)
    #     elif self.last_collision is False:
    #         self.set_sphere(0.16)
    #     else:
    #         self.last_collision=False
    #         return 0

    def update_r(self, obs_lst, q_lst, draw=True):
        # for obstacles in self.move_obstacles:
        #     self._p.addUserDebugLine(obstacles.pos_list[0], 5*(obstacles.pos_list[2]-obstacles.pos_list[0])+obstacles.pos_list[0],
        #                              lineColorRGB=[0.8, 0.8, 0.0], lineWidth=4)

        self.q_thre = -0.1
        q_lst = np.asarray(q_lst)
        q_sum = np.sum(q_lst)
        print("!!!!!!q_sum!!!!!!!", q_sum)
        try:
            min_q = min(q_lst)
            # print("min_q", min_q)
        except:
            self.last_collision = False
            self.set_sphere(0.16)
            return 0

        if q_sum < -1.0:
            print("min_q", min_q)
            self.last_collision = True
            self.set_sphere(0.3)
            # normalize Q for color
            minimum_q = -2
            b = 0.6
            k = -b / minimum_q
            color_lst = q_lst * k + b
            # color_lst = (q_lst - min_q) / (max(q_lst) - min_q + 0.000001)
            # color_lst = (q_lst-min_q)/(max(q_lst)-min_q+0.000001)

            if draw:
                self.draw_q(obs_lst, q_lst, color_lst)
            # for obs, q, c in zip(obs_lst, q_lst, color_lst):
            #     if q<-0.025:
            #         self._p.addUserDebugText(text = str(q)[1:7], textPosition=obs['observation'][:3],
            #                                  textSize=1.2, textColorRGB=colorsys.hsv_to_rgb(0.5-c/2, c+0.5, c), lifeTime=2)
        elif self.last_collision is False:
            self.set_sphere(0.16)
        else:
            self.last_collision = False
            return 0

    def draw_q(self, obs_lst, q_lst, color_lst):


        for obs, q, c in zip(obs_lst, q_lst, color_lst):
            if q < self.q_thre:
                # self._p.addUserDebugText(text=str(q)[1:7], textPosition=obs['observation'][:3],
                #                          textSize=1.2, textColorRGB=colorsys.hsv_to_rgb(0.5 - c / 2, c + 0.5, c),
                #                          lifeTime=1)
                self._p.addUserDebugText(text=str(q)[1:7], textPosition=obs['observation'][:3],
                                         textSize=1.2, textColorRGB=colorsys.hsv_to_rgb(c, 1-c, 0.8),
                                         lifeTime=1)

    def update_robot_obs(self, current_obs, next_rob_state):
        # change robot state to certain state and update obs
        ur5_states = current_obs[0:21].copy()
        ur5_states[:9] = next_rob_state
        ur5_eef_position = ur5_states[:3]

        self.human_states = self.agents[1].calc_state()


        delta_p = np.asarray([(p - ur5_eef_position) for p in self.human_states])
        d = np.linalg.norm(delta_p, axis=1)
        min_dist = np.min(d)

        # clip obs
        obs_human = delta_p.copy()
        indices = np.where(d > self.max_obs_dist_threshold)
        obs_human[indices] = np.full((1, 3), self.max_obs_dist_threshold + 0.2)

        if self.USE_RNN:
            human_obs_input = np.asarray(self.last_human_obs_list).flatten()

            # print("human_obs_input", self.last_human_obs_list)
            obs = np.concatenate([np.asarray(ur5_states), human_obs_input,
                                  np.asarray(self.goal).flatten(), np.asarray([min_dist])])
        else:
            human_obs_input = np.asarray(self.last_human_obs_list[-2:]).flatten()
            obs = np.concatenate([np.asarray(ur5_states), human_obs_input,
                                  np.asarray(self.goal).flatten(), np.asarray([min_dist])])


        achieved_goal = ur5_states[3:9]  # ur5 joint
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

        return obs

    def update_goal_obs(self, next_goal):
        # 2. change goal state
        obs = self.get_obs()
        obs["desired_goal"] = next_goal

        return obs

    def get_robot_ik(self, pos, ori=None):
        target_jp = self.agents[0].bullet_ik(pos, ori)
        return target_jp

    # def get_hands_position(self):
    #     human_current = self.last_human_obs_list[-1]
    #     wrist_right = human_current[-3:]
    #     wrist_left = human_current[-12:-9]
    #     return (wrist_right)



class UR5HumanRealEnv(UR5HumanEnv):
    def __init__(self, render=False, max_episode_steps=8000,
                 early_stop=True, distance_threshold = 0.4,
                 max_obs_dist = 0.8 ,dist_lowerlimit=0.02, dist_upperlimit=0.2,
                 reward_type="sparse",  use_rnn = True):
        super(UR5HumanRealEnv, self).__init__(render=render, max_episode_steps=max_episode_steps,
                 early_stop=early_stop, distance_threshold = distance_threshold,
                 max_obs_dist = max_obs_dist ,dist_lowerlimit=dist_lowerlimit, dist_upperlimit=dist_upperlimit,
                 reward_type=reward_type,  use_rnn = use_rnn)


        #--------------------------
    def _set_agents(self, max_obs_dist):
        # self.agents = [UR5EefRobot(dt=self.sim_dt * self.frame_skip),
        #                URDFHumanoid(max_obs_dist, load=True, test=True)]
        # self.agents = [UR5RealRobot(action_dim=6, obs_dim=21, joint_control = True),
        #                URDFHumanoid(max_obs_dist, load=True, test=True)]

        self.agents = [UR5RealRobot(action_dim=6, obs_dim=21, joint_control = True),
                       RealHumanoid(max_obs_dist)]

    def _special_rob_reset(self, start_joint):
        self.agents[0].reset(self._p, client_id=self.physicsClientId, base_position=self.robot_base)
        move_to_start(self.agents[0].ur5_rob_control, start_joint)
        ar =  self.agents[0].calc_state(
        )
        return ar





class UR5HumanPlanEnv(UR5HumanEnv):
    def __init__(self, render=False, max_episode_steps=1000,
                 early_stop=True, distance_threshold = 0.04,
                 max_obs_dist = 0.8 ,dist_lowerlimit=0.02, dist_upperlimit=0.2,
                 reward_type="sparse",  use_rnn = True):
        super(UR5HumanPlanEnv, self).__init__(render=render, max_episode_steps=max_episode_steps,
                 early_stop=early_stop, distance_threshold = distance_threshold,
                 max_obs_dist = max_obs_dist ,dist_lowerlimit=dist_lowerlimit, dist_upperlimit=dist_upperlimit,
                 reward_type=reward_type,  use_rnn = use_rnn)

        # --- following demo----
        # self.demo_data = load_demo()
        # self.sphere_radius = 0.03
        # self.last_collision = False
        #--------------------------


    def _set_agents(self, max_obs_dist):
        # self.agents = [UR5EefRobot(dt=self.sim_dt * self.frame_skip),
        #                URDFHumanoid(max_obs_dist, load=True, test=True)]
        self.agents = [UR5EefRobot(dt=self.sim_dt * self.frame_skip),
                       URDFHumanoid(max_obs_dist, load=False, test=True)]

    def step(self, action):
        self.iter_num += 1

        # self.agents[0].apply_action(action)
        # self.agents[1].apply_action(0)

        self.scene.global_step()
        obs = self._get_obs()
        done = False

        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'is_collision': self._contact_detection(),
            'min_dist': self.obs_min_dist,
            'safe_threshold': self.current_safe_dist,
            'joint_vel': obs["observation"][9:21]
        }

        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)


        if self.iter_num > self.max_episode_steps:
            done = True
        if self.early_stop:
            if info["is_success"]:
                done = True


        self._p.resetBasePositionAndOrientation(self.goal_id, posObj=self.eef_goal[:3], ornObj=self.eef_goal[3:])
        return obs, reward, done, info

    def is_contact(self):
        return self._contact_detection()








#
# class UR5HumanPlanEnv(UR5DynamicReachObsEnv):
#     def __init__(self, render=False, max_episode_steps=1000,
#                  early_stop=True, distance_threshold = 0.04,
#                  max_obs_dist = 0.8 ,dist_lowerlimit=0.02, dist_upperlimit=0.2,
#                  reward_type="sparse",  use_rnn = True):
#         super(UR5HumanPlanEnv, self).__init__(render=render, max_episode_steps=max_episode_steps,
#                  early_stop=early_stop, distance_threshold = distance_threshold,
#                  max_obs_dist = max_obs_dist ,dist_lowerlimit=dist_lowerlimit, dist_upperlimit=dist_upperlimit,
#                  reward_type=reward_type,  use_rnn = use_rnn)
#
#         # --- following demo----
#         self.demo_data = load_demo()
#         self.sphere_radius = 0.03
#         self.last_collision = False
#         #--------------------------
#
#
#     def _set_agents(self, max_obs_dist):
#         self.agents = [UR5EefRobot(dt=self.sim_dt * self.frame_skip),
#                        URDFHumanoid(max_obs_dist, load=True, test=True)]
#
#
#     def reset(self):
#
#         self.last_obs_human = np.full(18,self.max_obs_dist_threshold+0.2)
#         self.last_robot_joint = np.zeros(6)
#         self.current_safe_dist = self._set_safe_distance()
#
#
#         if (self.physicsClientId < 0):
#             self.ownsPhysicsClient = True
#
#             if self.isRender:
#                 self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
#             else:
#                 self._p = bullet_client.BulletClient()
#
#             self._p.setGravity(0, 0, -9.81)
#             self._p.setTimeStep(self.sim_dt)
#
#             self.physicsClientId = self._p._client
#             self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
#
#
#         if self.scene is None:
#             self.scene = self.create_single_player_scene(self._p)
#         if not self.scene.multiplayer and self.ownsPhysicsClient:
#             self.scene.episode_restart(self._p)
#
#         self.camera_adjust()
#
#         for a in self.agents:
#             a.scene = self.scene
#
#
#         self.frame = 0
#         self.iter_num = 0
#
#         self.robot_base = [0, 0, 0]
#
#         #-----------------set robot to start joint ------#
#         start_joint  = self.demo_data[0]['robjp']
#         ar = self.agents[0].reset(self._p, client_id=self.physicsClientId, base_position=self.robot_base,
#                                   base_rotation=[0, 0, 0, 1], joint_angle=start_joint)
#
#         #---------------real human data----------------------------#
#         ah = self.agents[1].reset(self._p, client_id=self.physicsClientId,
#                                   base_rotation=[0.0005629, 0.707388, 0.706825, 0.0005633])
#
#         #------prepare reference path-----------------
#         for i in range(len(self.demo_data) - 1):
#             if np.linalg.norm(np.array(self.demo_data[i]['robjp']) -np.array(self.demo_data[i+1]['robjp']))<0.01:
#                 start_idx = i
#             else:
#                 break
#
#         self.reference_path = [self.demo_data[i]['robjp'] for i in range(start_idx, len(self.demo_data))]
#
#         self.goal = self.demo_data[-1]['toolp']
#
#
#         print("goal,", self.goal)
#         #------------------------------------------
#
#         self._p.stepSimulation()
#         obs = self._get_obs()
#
#
#         s = []
#         s.append(ar)
#         s.append(ah)
#         self._p.resetBasePositionAndOrientation(self.goal_id, posObj=self.goal, ornObj=[0.0, 0.0, 0.0, 1.0])
#
#         return obs
#
#     def step(self, action):
#         self.iter_num += 1
#
#         # self.agents[0].apply_action(action)
#         # self.agents[1].apply_action(0)
#
#         self.scene.global_step()
#         obs = self._get_obs()
#         done = False
#
#         info = {
#             'is_success': self._is_success(obs['achieved_goal'], self.goal),
#             'is_collision': self._contact_detection(),
#             'min_dist': self.obs_min_dist,
#             'safe_threshold': self.current_safe_dist,
#             'ee_vel':obs["observation"][3:9]
#         }
#
#         reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
#
#
#         if self.iter_num > self.max_episode_steps:
#             done = True
#         if self.early_stop:
#             if info["is_success"]:
#                 done = True
#
#         self._p.resetBasePositionAndOrientation(self.goal_id, posObj=self.goal, ornObj=[0.0, 0.0, 0.0, 1.0])
#
#         return obs, reward, done, info
#
#     def is_contact(self):
#         return self._contact_detection()

