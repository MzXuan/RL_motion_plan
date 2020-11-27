import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, currentdir)

from ur5eef import UR5EefRobot
from humanoid import URDFHumanoid

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
from gym_rlmp.envs.ws_path_gen import WsPathGen

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

def move_to_start(ur5, data):
    start_joint = data[0]['robjp']
    print(f"move to {start_joint}, please be careful")
    ur5.set_joint_position(start_joint, wait=True)
    print(f"success move to {start_joint}")


def move_along_path(ur5, ws_path_gen, dt=0.02):
    toolp,_,_ = ur5.get_tool_state()
    next_goal, next_vel,_ = ws_path_gen.next_goal(toolp, 0.08)

    ref_vel = (next_goal-toolp)
    print(f"current goal {toolp}, next goal {next_goal}, next_vel {next_vel}, ref_vel {ref_vel}")
    tool_vel = np.zeros(6)
    tool_vel[:3]=ref_vel
    ur5.set_tool_velocity(tool_vel)



class UR5HumanEnv(UR5DynamicReachObsEnv):
    def __init__(self, render=False, max_episode_steps=1000,
                 early_stop=False, distance_threshold = 0.04,
                 max_obs_dist = 0.8 ,dist_lowerlimit=0.02, dist_upperlimit=0.2,
                 reward_type="sparse",  use_rnn = True):
        super(UR5HumanEnv, self).__init__(render=render, max_episode_steps=max_episode_steps,
                 early_stop=early_stop, distance_threshold = distance_threshold,
                 max_obs_dist = max_obs_dist ,dist_lowerlimit=dist_lowerlimit, dist_upperlimit=dist_upperlimit,
                 reward_type=reward_type,  use_rnn = use_rnn)

        # --- following demo----
        self.demo_data = load_demo()
        self.sphere_radius = 0.03
        self.last_collision = False
        #--------------------------

    def _set_agents(self, max_obs_dist):
        self.agents = [UR5EefRobot(dt=self.sim_dt * self.frame_skip),
                       URDFHumanoid(max_obs_dist, load=True, test=True)]


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

        #-----------------set to start
        start_joint  = self.demo_data[0]['robjp']
        ar = self.agents[0].reset(self._p, client_id=self.physicsClientId, base_position=self.robot_base,
                                  base_rotation=[0, 0, 0, 1], joint_angle=start_joint)

        # # set robot to start
        # x = np.random.uniform(0.2, 0.4)
        # y = np.random.uniform(-0.6, -0.1)
        # z = np.random.uniform(0.2, 0.5)
        #
        # robot_eef_pose = [np.random.choice([-1, 1]) * x, y, z]
        #
        # self.robot_start_eef = robot_eef_pose.copy()
        # ar = self.agents[0].reset(self._p, client_id=self.physicsClientId, base_position=self.robot_base,
        #                           base_rotation=[0, 0, 0, 1], eef_pose=self.robot_start_eef, joint_angle = start_joint)


        #---------------real human data----------------------------#
        ah = self.agents[1].reset(self._p, client_id=self.physicsClientId,
                                  base_rotation=[0.0005629, 0.707388, 0.706825, 0.0005633])

        #------prepare path-----------------
        path = [self.demo_data[i]['toolp'] for i in range(len(self.demo_data))]
        vel_path = [self.demo_data[i]['toolv'] for i in range(len(self.demo_data))]
        self.ws_path_gen = WsPathGen(path, vel_path, self.distance_threshold)

        self.draw_path(path)

        #-------set goal from record demo-------------
        rob_eef = ar[:3]
        self.final_goal = self.demo_data[-1]['toolp']
        self.goal,_,_ = self.ws_path_gen.next_goal(rob_eef, self.sphere_radius)
        print("goal,", self.goal)
        #------------------------------------------

        self._p.stepSimulation()
        obs = self._get_obs()


        s = []
        s.append(ar)
        s.append(ah)
        self._p.resetBasePositionAndOrientation(self.goal_id, posObj=self.goal, ornObj=[0.0, 0.0, 0.0, 1.0])

        return obs

    def _get_obs(self):
        '''
        :return:
        obs_robot: [robot states, human states]
        obs_humanoid: [human states, robot states]
        info of every agent: [enf_effector_position, done]
        done of every agent: [True or False]
        '''
        infos = {}
        dones = [False for _ in range(self._n_agents)]
        ur5_states = self.agents[0].calc_state()
        ur5_eef_position = ur5_states[:3]
        achieved_goal = ur5_eef_position
        self.human_states = self.agents[1].calc_state()
        infos['succeed'] = dones


        delta_p = np.asarray([(p-ur5_eef_position) for p in self.human_states])
        d = np.linalg.norm(delta_p,axis=1)
        min_dist = np.min(d)

        #clip obs
        obs_human = delta_p.copy()
        indices = np.where(d > self.max_obs_dist_threshold)
        obs_human[indices] = np.full((1,3), self.max_obs_dist_threshold+0.2)


        try:
            self.goal, _,self.goal_indices = self.ws_path_gen.next_goal(ur5_eef_position, self.sphere_radius)
        except:
            print("!!!!!!!!!!!!!!not exist self.ws_path_gen")


        self.obs_min_dist = min_dist
        self.last_human_obs_list.append(np.asarray(obs_human.copy()).flatten())
        # print("shape of human states: ", np.asarray(self.last_human_obs_list).shape)

        if self.USE_RNN:
            human_obs_input = np.asarray(self.last_human_obs_list).flatten()

            # print("human_obs_input", self.last_human_obs_list)
            obs = np.concatenate([np.asarray(ur5_states), human_obs_input,
                                  np.asarray(self.goal).flatten(), np.asarray([self.obs_min_dist])])
        else:
            human_obs_input = np.asarray(self.last_human_obs_list[-2:]).flatten()
            obs = np.concatenate([np.asarray(ur5_states), human_obs_input,
                                  np.asarray(self.goal).flatten(), np.asarray([self.obs_min_dist])])

        # print("shape of obs is: ", obs.shape)

        self.last_obs_human = obs_human.copy()

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def draw_path(self, path):
        for i in range(len(path)-1):
            self._p.addUserDebugLine(path[i], path[i+1],
                                     lineColorRGB=[0.8,0.8,0.0],lineWidth=4 )

    def set_sphere(self, r):
        self.sphere_radius = r

    def update_r(self, obs_lst, q_lst, draw=True):
        # for obstacles in self.move_obstacles:
        #     self._p.addUserDebugLine(obstacles.pos_list[0], 5*(obstacles.pos_list[2]-obstacles.pos_list[0])+obstacles.pos_list[0],
        #                              lineColorRGB=[0.8, 0.8, 0.0], lineWidth=4)

        q_lst = np.asarray(q_lst)
        try:
            min_q = min(q_lst)
            print("min_q", min_q)
        except:
            self.last_collision = False
            self.set_sphere(0.1)
            return 0

        if min_q<-0.025:
            self.last_collision = True
            self.set_sphere(0.5)
            #normalize Q for color
            color_lst = (q_lst-min_q)/(max(q_lst)-min_q+0.000001)

            if draw:
                self.draw_q(obs_lst, q_lst, color_lst)
            # for obs, q, c in zip(obs_lst, q_lst, color_lst):
            #     if q<-0.025:
            #         self._p.addUserDebugText(text = str(q)[1:7], textPosition=obs['observation'][:3],
            #                                  textSize=1.2, textColorRGB=colorsys.hsv_to_rgb(0.5-c/2, c+0.5, c), lifeTime=2)
        elif self.last_collision is False:
            self.set_sphere(0.1)
        else:
            self.last_collision=False
            return 0

    def draw_q(self, obs_lst, q_lst, color_lst):
        for obs, q, c in zip(obs_lst, q_lst, color_lst):
            if q < -0.025:
                self._p.addUserDebugText(text=str(q)[1:7], textPosition=obs['observation'][:3],
                                         textSize=1.2, textColorRGB=colorsys.hsv_to_rgb(0.5 - c / 2, c + 0.5, c),
                                         lifeTime=2)



# class UR5HumanPlanEnv(UR5RealPlanTestEnv):
#     def __init__(self, render=False, max_episode_steps=1000,
#                  early_stop=True,  distance_threshold=0.05,
#                  max_obs_dist=0.35, dist_lowerlimit=0.05, dist_upperlimit=0.3,
#                  reward_type="sparse"):
#         self.collision_weight = 0
#         self.iter_num = 0
#         self.max_episode_steps = max_episode_steps
#
#         self.scene = None
#         self.physicsClientId = -1
#         self.ownsPhysicsClient = 0
#         self.isRender = render
#
#         self.hz = 240
#         self.sim_dt = 1.0 / self.hz
#         self.frame_skip = 4
#
#         # --- following demo----
#         self.demo_data = load_demo()
#         self.sphere_radius = 0.03
#
#         self.agents = [UR5Robot(dt=self.sim_dt * self.frame_skip), RealHumanoid(max_obs_dist)]
#         self._n_agents = 2
#         self.seed()
#         self._cam_dist = 3
#         self._cam_yaw = 0
#         self._cam_pitch = 30
#
#         self.target_off_set=0.2
#         self.safe_dist_threshold = 0.6
#
#         self.distance_threshold = distance_threshold # for success check
#         self.early_stop=early_stop
#         self.reward_type = reward_type
#
#         self.max_obs_dist_threshold = max_obs_dist
#         self.safe_dist_lowerlimit = dist_lowerlimit
#         self.safe_dist_upperlimit = dist_upperlimit
#         self._set_safe_distance()
#
#
#
#         self.n_actions=3
#         self.action_space = gym.spaces.Box(-1., 1., shape=( self.n_actions,), dtype='float32')
#         self.observation_space = gym.spaces.Dict(dict(
#             desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
#             achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
#             observation=gym.spaces.Box(-np.inf, np.inf, shape=(40,), dtype='float32'),
#         ))
#
#
#         # Set observation and action spaces
#         self.agents_observation_space = Tuple([
#             agent.observation_space for agent in self.agents
#         ])
#
#         self.first_reset = True
#
#
#
#     def reset(self):
#         self.last_human_eef = [0, 0, 0]
#         self.last_robot_eef = [0, 0, 0]
#         self.last_robot_joint = np.zeros(6)
#         self.current_safe_dist = self._set_safe_distance()
#
#         if (self.physicsClientId < 0):
#             self.ownsPhysicsClient = True
#
#             if self.isRender:
#                 self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
#             else:
#                 self._p = bullet_client.BulletClient()
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
#         self.frame = 0
#         self.iter_num = 0
#
#         self.robot_base = [0, 0, 0]
#
#         start_joint  = self.demo_data[0]['robjp']
#
#         # set robot to start
#
#         x = np.random.uniform(0.2, 0.4)
#         y = np.random.uniform(-0.6, -0.1)
#         z = np.random.uniform(0.2, 0.5)
#
#         robot_eef_pose = [np.random.choice([-1, 1]) * x, y, z]
#
#         self.robot_start_eef = robot_eef_pose.copy()
#         ar = self.agents[0].reset(self._p, client_id=self.physicsClientId, base_position=self.robot_base,
#                                   base_rotation=[0, 0, 0, 1], eef_pose=self.robot_start_eef, joint_angle = start_joint)
#
#
#         #---------------real human----------------------------#
#         ah = self.agents[1].reset(self._p)
#
#
#         #-------set goal from record demo-------------
#         rob_eef = ar[:3]
#         self.final_goal = self.demo_data[-1]['toolp']
#         self.goal = self.final_goal
#         #------------------------------------------
#
#         self._p.stepSimulation()
#
#         obs = self._get_obs()
#
#         # ---- for motion planner---#
#         # initial joint states from file
#
#         initial_conf = self.demo_data[0]['robjp']
#         final_conf = self.demo_data[-1]['robjp']
#
#
#         start_time = time.time()
#         result = self.motion_planner(initial_conf=initial_conf, final_conf=final_conf)
#         print("use time: ", time.time() - start_time)
#         if result is not None:
#             (joint_path, cartesian_path) = result
#             self.reference_traj = np.asarray(joint_path)
#
#             #
#             #     print("cartesian_path", cartesian_path)
#             # todo: draw cartesian path; and move through joint path
#             # self.draw_path(cartesian_path)
#
#             self.goal = np.asarray(cartesian_path[2])
#         print("initial conf", initial_conf)
#
#
#
#         s = []
#         s.append(ar)
#         s.append(ah)
#         self._p.resetBasePositionAndOrientation(self.goal_id, posObj=self.goal, ornObj=[0.0, 0.0, 0.0, 1.0])
#
#         return obs
#
#     # def draw_path(self, path):
#     #     for i in range(len(path) - 1):
#     #         self._p.addUserDebugLine(path[i], path[i + 1],
#     #                                  lineColorRGB=[0.8, 0.8, 0.0], lineWidth=4, lifeTime=4)
#
#
#
