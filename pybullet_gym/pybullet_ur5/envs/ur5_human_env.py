import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, currentdir)

from ur5eef import UR5EefRobot
from ur5 import UR5Robot
from humanoid_real import RealHumanoid

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
from .ur5_human_real_env import UR5RealTestEnv, UR5RealPlanTestEnv





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
    next_goal, next_vel = ws_path_gen.next_goal(toolp, 0.08)

    ref_vel = (next_goal-toolp)
    print(f"current goal {toolp}, next goal {next_goal}, next_vel {next_vel}, ref_vel {ref_vel}")
    tool_vel = np.zeros(6)
    tool_vel[:3]=ref_vel
    ur5.set_tool_velocity(tool_vel)



class UR5HumanEnv(UR5RealTestEnv):
    def __init__(self, render=False, max_episode_steps=1000,
                 early_stop=True,  distance_threshold=0.05,
                 max_obs_dist=0.35, dist_lowerlimit=0.05, dist_upperlimit=0.3,
                 reward_type="sparse"):

        self.distance_close = 0.3

        self.collision_weight = 0
        self.iter_num = 0
        self.max_episode_steps = max_episode_steps

        self.scene = None
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self.isRender = render

        self.hz = 240
        self.sim_dt = 1.0 / self.hz
        self.frame_skip = 4

        # --- following demo----
        self.demo_data = load_demo()
        self.sphere_radius = 0.03


        self.agents = [UR5EefRobot(dt= self.sim_dt*self.frame_skip, action_dim =3 ), RealHumanoid(max_obs_dist)]


        self._n_agents = 2
        self.seed()
        self._cam_dist = 3
        self._cam_yaw = 0
        self._cam_pitch = 30

        self.target_off_set=0.2
        self.safe_dist_threshold = 0.6

        self.distance_threshold = distance_threshold # for success check
        self.early_stop=early_stop
        self.reward_type = reward_type

        self.max_obs_dist_threshold = max_obs_dist
        self.safe_dist_lowerlimit = dist_lowerlimit
        self.safe_dist_upperlimit = dist_upperlimit
        self._set_safe_distance()


        self.n_actions=3
        self.action_space = gym.spaces.Box(-1., 1., shape=( self.n_actions,), dtype='float32')
        self.observation_space = gym.spaces.Dict(dict(
            desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            observation=gym.spaces.Box(-np.inf, np.inf, shape=(54,), dtype='float32'),
        ))



        # Set observation and action spaces
        self.agents_observation_space = Tuple([
            agent.observation_space for agent in self.agents
        ])


        self.first_reset = True


    def reset(self):

        self.last_human_eef = [0, 0, 0]
        self.last_robot_eef = [0, 0, 0]
        self.last_robot_joint = np.zeros(6)
        self.current_safe_dist = self._set_safe_distance()

        if (self.physicsClientId < 0):
            self.ownsPhysicsClient = True

            if self.isRender:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            else:
                self._p = bullet_client.BulletClient()

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

        start_joint  = self.demo_data[0]['robjp']

        # set robot to start

        x = np.random.uniform(0.2, 0.4)
        y = np.random.uniform(-0.6, -0.1)
        z = np.random.uniform(0.2, 0.5)

        robot_eef_pose = [np.random.choice([-1, 1]) * x, y, z]

        self.robot_start_eef = robot_eef_pose.copy()
        ar = self.agents[0].reset(self._p, client_id=self.physicsClientId, base_position=self.robot_base,
                                  base_rotation=[0, 0, 0, 1], eef_pose=self.robot_start_eef, joint_angle = start_joint)


        #---------------real human----------------------------#
        ah = self.agents[1].reset(self._p)

        #------prepare path-----------------
        path = [self.demo_data[i]['toolp'] for i in range(len(self.demo_data))]
        vel_path = [self.demo_data[i]['toolv'] for i in range(len(self.demo_data))]
        self.ws_path_gen = WsPathGen(path, vel_path, self.distance_threshold)

        self.draw_path(path)

        #-------set goal from record demo-------------
        rob_eef = ar[:3]
        self.final_goal = self.demo_data[-1]['toolp']
        self.goal = self.ws_path_gen.next_goal(rob_eef, self.sphere_radius)
        #------------------------------------------

        self._p.stepSimulation()

        obs = self._get_obs()


        s = []
        s.append(ar)
        s.append(ah)
        self._p.resetBasePositionAndOrientation(self.goal_id, posObj=self.goal, ornObj=[0.0, 0.0, 0.0, 1.0])

        return obs



class UR5HumanPlanEnv(UR5RealPlanTestEnv):
    def __init__(self, render=False, max_episode_steps=1000,
                 early_stop=True,  distance_threshold=0.05,
                 max_obs_dist=0.35, dist_lowerlimit=0.05, dist_upperlimit=0.3,
                 reward_type="sparse"):
        self.collision_weight = 0
        self.iter_num = 0
        self.max_episode_steps = max_episode_steps

        self.scene = None
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self.isRender = render

        self.hz = 240
        self.sim_dt = 1.0 / self.hz
        self.frame_skip = 4

        # --- following demo----
        self.demo_data = load_demo()
        self.sphere_radius = 0.03

        self.agents = [UR5Robot(dt=self.sim_dt * self.frame_skip), RealHumanoid(max_obs_dist)]
        self._n_agents = 2
        self.seed()
        self._cam_dist = 3
        self._cam_yaw = 0
        self._cam_pitch = 30

        self.target_off_set=0.2
        self.safe_dist_threshold = 0.6

        self.distance_threshold = distance_threshold # for success check
        self.early_stop=early_stop
        self.reward_type = reward_type

        self.max_obs_dist_threshold = max_obs_dist
        self.safe_dist_lowerlimit = dist_lowerlimit
        self.safe_dist_upperlimit = dist_upperlimit
        self._set_safe_distance()



        self.n_actions=3
        self.action_space = gym.spaces.Box(-1., 1., shape=( self.n_actions,), dtype='float32')
        self.observation_space = gym.spaces.Dict(dict(
            desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            observation=gym.spaces.Box(-np.inf, np.inf, shape=(40,), dtype='float32'),
        ))


        # Set observation and action spaces
        self.agents_observation_space = Tuple([
            agent.observation_space for agent in self.agents
        ])

        self.first_reset = True



    def reset(self):
        self.last_human_eef = [0, 0, 0]
        self.last_robot_eef = [0, 0, 0]
        self.last_robot_joint = np.zeros(6)
        self.current_safe_dist = self._set_safe_distance()

        if (self.physicsClientId < 0):
            self.ownsPhysicsClient = True

            if self.isRender:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            else:
                self._p = bullet_client.BulletClient()

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

        start_joint  = self.demo_data[0]['robjp']

        # set robot to start

        x = np.random.uniform(0.2, 0.4)
        y = np.random.uniform(-0.6, -0.1)
        z = np.random.uniform(0.2, 0.5)

        robot_eef_pose = [np.random.choice([-1, 1]) * x, y, z]

        self.robot_start_eef = robot_eef_pose.copy()
        ar = self.agents[0].reset(self._p, client_id=self.physicsClientId, base_position=self.robot_base,
                                  base_rotation=[0, 0, 0, 1], eef_pose=self.robot_start_eef, joint_angle = start_joint)


        #---------------real human----------------------------#
        ah = self.agents[1].reset(self._p)


        #-------set goal from record demo-------------
        rob_eef = ar[:3]
        self.final_goal = self.demo_data[-1]['toolp']
        self.goal = self.final_goal
        #------------------------------------------

        self._p.stepSimulation()

        obs = self._get_obs()

        # ---- for motion planner---#
        # initial joint states from file

        initial_conf = self.demo_data[0]['robjp']
        final_conf = self.demo_data[-1]['robjp']


        start_time = time.time()
        result = self.motion_planner(initial_conf=initial_conf, final_conf=final_conf)
        print("use time: ", time.time() - start_time)
        if result is not None:
            (joint_path, cartesian_path) = result
            self.reference_traj = np.asarray(joint_path)

            #
            #     print("cartesian_path", cartesian_path)
            # todo: draw cartesian path; and move through joint path
            # self.draw_path(cartesian_path)

            self.goal = np.asarray(cartesian_path[2])
        print("initial conf", initial_conf)



        s = []
        s.append(ar)
        s.append(ah)
        self._p.resetBasePositionAndOrientation(self.goal_id, posObj=self.goal, ornObj=[0.0, 0.0, 0.0, 1.0])

        return obs

    # def draw_path(self, path):
    #     for i in range(len(path) - 1):
    #         self._p.addUserDebugLine(path[i], path[i + 1],
    #                                  lineColorRGB=[0.8, 0.8, 0.0], lineWidth=4, lifeTime=4)



