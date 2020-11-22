import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, currentdir)

import assets
from scenes.stadium import StadiumScene, PlaneScene

from ur5eef import UR5EefRobot
from ur5 import UR5Robot
from humanoid import URDFHumanoid

import gym, gym.spaces, gym.utils, gym.utils.seeding
from gym.spaces import Tuple

import numpy as np
import pybullet
from pybullet_utils import bullet_client


from pybullet_planning import link_from_name, get_moving_links, get_link_name
from pybullet_planning import get_joint_names, get_movable_joints
from pybullet_planning import multiply, get_collision_fn
from pybullet_planning import sample_tool_ik
from pybullet_planning import set_joint_positions
from pybullet_planning import get_joint_positions, plan_joint_motion, compute_forward_kinematics


import ikfast_ur5
import pyquaternion

import pickle

import utils
import random
import time


from scipy.interpolate import griddata

from pkg_resources import parse_version

from mpi4py import MPI
import colorsys




def normalize_conf(start, end):
    circle = 6.2831854
    normal_e = []
    for s, e in zip(start, end):
        if e > s:
            test = e - circle
        else:
            test = e + circle
        normal_e.append(test if np.linalg.norm(test-s)<np.linalg.norm(e-s) else e)
    return normal_e

def min_dist_conf(initial_conf, conf_list):

    dist_list = [np.linalg.norm([np.asarray(initial_conf)- np.asarray(conf)]) for conf in conf_list]
    id_min = np.argmin(np.asarray(dist_list))
    return conf_list[id_min]

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class L(list):
     def append(self, item):
         list.append(self, item)
         if len(self) > 6: del self[0]



class UR5DynamicReachObsEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 60}

    def __init__(self, render=False, max_episode_steps=1000,
                 early_stop=False, distance_threshold = 0.04,
                 max_obs_dist = 0.8 ,dist_lowerlimit=0.02, dist_upperlimit=0.2,
                 reward_type="sparse"):
        self.iter_num = 0
        self.max_episode_steps = max_episode_steps

        self.scene = None
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self.isRender = render

        self.hz = 240
        self.sim_dt = 1.0 / self.hz
        self.frame_skip = 8


        self.agents = [UR5EefRobot(dt= self.sim_dt*self.frame_skip),
                       URDFHumanoid(max_obs_dist, load = True)]

        self.arm_states = None

        self._n_agents = 2
        self.seed()
        self._cam_dist = 1
        self._cam_yaw = 0
        self._cam_pitch = -30

        self.target_off_set=0.2
        self.distance_threshold = distance_threshold #for success

        self.max_obs_dist_threshold = max_obs_dist
        self.safe_dist_lowerlimit= dist_lowerlimit
        self.safe_dist_upperlimit = dist_upperlimit

        # self.distance_threshold = distance_threshold
        self.early_stop=early_stop
        self.reward_type = reward_type

        self.n_actions = 3



        self.action_space = gym.spaces.Box(-1., 1., shape=( self.n_actions,), dtype='float32')

        self.USE_RNN = True
        if self.USE_RNN:
            self.observation_space = gym.spaces.Dict(dict(
                desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
                achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
                observation=gym.spaces.Box(-np.inf, np.inf, shape=(126,), dtype='float32'),
            ))
        else:
            self.observation_space = gym.spaces.Dict(dict(
                desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
                achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
                observation=gym.spaces.Box(-np.inf, np.inf, shape=(54,), dtype='float32'),
            ))

        # Set observation and action spaces

        # Set observation and action spaces
        self.agents_observation_space = Tuple([
            agent.observation_space for agent in self.agents
        ])

        self.last_human_obs_list = L(np.zeros((6,18)))





    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = PlaneScene(bullet_client, gravity=0, timestep=self.sim_dt, frame_skip=self.frame_skip)

        # self.long_table_body = bullet_client.loadURDF(os.path.join(assets.getDataPath(), "scenes_data", "longtable/longtable.urdf"),
        #                        [-1, -0.9, -1.0],
        #                        [0.000000, 0.000000, 0.0, 1])

        self.long_table_body = bullet_client.loadURDF(os.path.join(assets.getDataPath(), "scenes_data", "circletable/circletable.urdf"),
                               [0, 0, -0.005],
                               [0.7068252, 0, 0, -0.7073883])

        self.goal_id = bullet_client.loadURDF(
            os.path.join(assets.getDataPath(), "scenes_data", "targetball/targetball.urdf"),
            [0, 0, 0],
            [0.000000, 0.000000, 0.0, 1])

        # self.agents[1].set_goal_position( self.human_pos)
        return self.stadium_scene


    def set_training(self, is_training):
        self.is_training = is_training


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        for a in self.agents:
            a.np_random = self.np_random
        return [seed]

    def set_render(self):
        self.physicsClientId = -1
        self.isRender = True


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

        #---- random goal -----#
        self.goal=np.zeros(3)

        collision_flag = True
        while collision_flag:

            x = np.random.uniform(-0.7, 0.7)
            y = np.random.uniform(-0.7, 0.7)
            z = np.random.uniform(0.1, 0.7)

            self.robot_start_eef = [x, y, z]

            self.goal = self.random_set_goal()
            if np.linalg.norm(self.goal) < 0.35 or np.linalg.norm(self.goal) >0.85:
                continue


            ar = self.agents[0].reset(self._p, client_id=self.physicsClientId, base_position=self.robot_base,
                                      base_rotation=[0, 0, 0, 1], eef_pose=self.robot_start_eef)
            # ---------------real human----------------------------#
            ah = self.agents[1].reset(self._p, client_id=self.physicsClientId,
                                      base_rotation = [0.0005629, 0.707388, 0.706825, 0.0005633], rob_goal=self.goal.copy())


            if ar is False:
                # print("failed to find valid robot solution of pose", robot_eef_pose)
                continue

            self._p.stepSimulation()
            obs = self._get_obs()


            collision_flag = self._contact_detection()
            # print("collision_flag is :", collision_flag)

        s = []
        s.append(ar)
        s.append(ah)
        self._p.resetBasePositionAndOrientation(self.goal_id, posObj=self.goal, ornObj=[0, 0, 0, 1])

        return obs

    def random_set_goal(self):
        max_xyz = [0.7, 0.7, 0.6]
        goal_reset = False
        while not goal_reset:
            goal = np.asarray(self.robot_start_eef.copy())
            goal[0] += np.random.uniform(-0.45, 0.45)
            goal[1] += np.random.uniform(-0.45, 0.45)
            goal[2] += np.random.uniform(-0.2, 0.2)
            if abs(goal[0]) < max_xyz[0] and abs(goal[1]) < max_xyz[1]\
                    and  abs(goal[1]) > 0.3 and goal[2] > 0\
                    and abs(goal[2]) < max_xyz[2]:
                goal_reset = True
        return goal


    def _set_safe_distance(self):
        return 0.1
        # return np.random.uniform(self.safe_dist_lowerlimit, self.safe_dist_upperlimit)

    def get_obs(self):
        return self._get_obs()


    def render(self, mode='human', close=False):
        if mode == "human":
            self.isRender = True

        if mode != "rgb_array":
            return np.array([])

        base_pos = [0, 0, 10]
        if (hasattr(self, 'robot')):
            if (hasattr(self.robot, 'body_xyz')):
                base_pos = self.robot.body_xyz

        # self._p.setRealTimeSimulation(0)
        view_matrix = self.viewmat
        proj_matrix = self.projmat


        w = 320
        h = 240
        (_, _, px, _, _) = self._p.getCameraImage(width=w,
                                                  height=h,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.reshape(np.array(px), (h, w, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array


    def close(self):
        if (self.ownsPhysicsClient):
            if (self.physicsClientId >= 0):
                self._p.disconnect()
        self.physicsClientId = -1


    def draw_Q(self, obs_lst, q_lst):
        # for obstacles in self.move_obstacles:
        #     self._p.addUserDebugLine(obstacles.pos_list[0], 5*(obstacles.pos_list[2]-obstacles.pos_list[0])+obstacles.pos_list[0],
        #                              lineColorRGB=[0.8, 0.8, 0.0], lineWidth=4)

        q_lst = np.asarray(q_lst)
        #normalize Q for color
        color_lst = (q_lst-min(q_lst))/(max(q_lst)-min(q_lst))

        for obs, q, c in zip(obs_lst, q_lst, color_lst):
            self._p.addUserDebugText(text = str(q)[2:7], textPosition=obs, textSize=1.2, textColorRGB=colorsys.hsv_to_rgb(0.5-c/2, c, c))


        return 0


    def update_robot_obs(self, next_eef):
        #change robot state to certain end_effector and update obs
        self.agents[0].bullet_ik(next_eef)
        obs = self.get_obs()


        return obs

    def update_goal_obs(self, next_goal):
        # 2. change goal state
        obs = self.get_obs()
        obs["desired_goal"] = next_goal

        return obs


    def step(self, action):
        self.iter_num += 1

        self.agents[0].apply_action(action)
        self.agents[1].apply_action(0)


        self.scene.global_step()
        obs = self._get_obs()
        done = False


        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'is_collision': self._contact_detection(),
            'min_dist': self.obs_min_dist,
            'safe_threshold': self.current_safe_dist,
            'ee_vel':obs["observation"][3:9]
        }

        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)


        if self.iter_num > self.max_episode_steps:
            done = True
        if self.early_stop:
            if info["is_success"] or info["is_collision"]:
                done = True

        return obs, reward, done, info



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



    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)

        #collision
        _is_collision = info['is_collision']
        if isinstance(_is_collision, np.ndarray):
            _is_collision = _is_collision.flatten()

        #safe distance
        min_dist = info['min_dist']
        safe_dist = info ['safe_threshold']

        if isinstance(min_dist, np.ndarray):
            min_dist = min_dist.flatten()
            safe_dist = safe_dist.flatten()

        distance = safe_dist - min_dist
        if isinstance(distance, np.ndarray):
            distance[(distance < 0)] = 0
        else:
            distance = 0 if distance<0 else distance

        smoothness = np.linalg.norm([info['ee_vel'][0:3] - info['ee_vel'][3:6]])

        # sum of reward
        a1 = -1
        a2 = -14
        a3 = -0.1
        asmooth = -0.1

        reward = a1 * (d > self.distance_threshold).astype(np.float32) \
                 + a2 * (_is_collision > 0) + a3 * distance + asmooth*smoothness
        reward_collision = a2 * (_is_collision > 0) + a3 * distance

        return [reward, reward_collision]

    def _contact_detection(self):
        # collision detection

        collisions = self._p.getContactPoints()
        collision_bodies = []
        for c in collisions:
            bodyinfo1 = self._p.getBodyInfo(c[1])
            bodyinfo2 = self._p.getBodyInfo(c[2])

            # print("bodyinfo1: ", bodyinfo1, "bodyinfo2: ", bodyinfo2)
            # print("collisions", collisions)
            # print("linkid 1 ", c[3])
            # print("linkid 2", c[4])

            # print("robot parts",self.agents[0].parts)
            if c[3] == 3 and c[4] == 5:
                continue
            if c[3] == 0 or c[4] == 0:
                continue
            # p = self._p.getLinkState(c[1], c[3])[0]
            collision_bodies.append(bodyinfo1[1].decode("utf-8"))
            collision_bodies.append(bodyinfo2[1].decode("utf-8"))

        if len(collision_bodies) != 0:
            if "ur5" in collision_bodies:  # robot collision
                # print("collision_bodies: ", collision_bodies)
                return True
            else:
                return False
        return False


    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)


    def camera_adjust(self):
        lookat = [0, -0.6, 0.8]
        distance = 1.0
        yaw = 0
        pitch = -30
        self._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)
        # self._p.getDebugVisualizerCamera(self._p.GUI)
        self.viewmat = \
            self._p.computeViewMatrixFromYawPitchRoll(lookat, distance, yaw, pitch, 0, upAxisIndex=2)

        self.projmat = self._p.computeProjectionMatrixFOV(fov=90, aspect=1, nearVal=0.1, farVal=3)

    def HUD(self, state, a, done):
        pass

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed



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



class UR5DynamicReachPlannerEnv(UR5DynamicReachObsEnv):
    def __init__(self, render=False, max_episode_steps=1000,
                 early_stop=False, distance_threshold = 0.025,
                 max_obs_dist = 0.8 ,dist_lowerlimit=0.02, dist_upperlimit=0.2,
                 reward_type="sparse"):
        super(UR5DynamicReachPlannerEnv, self).__init__(render=render, max_episode_steps=max_episode_steps,
                 early_stop=early_stop, distance_threshold = distance_threshold,
                 max_obs_dist = max_obs_dist ,dist_lowerlimit=dist_lowerlimit, dist_upperlimit=dist_upperlimit,
                 reward_type=reward_type)

        # self.iter_num = 0
        # self.max_episode_steps = max_episode_steps
        #
        # self.scene = None
        # self.physicsClientId = -1
        # self.ownsPhysicsClient = 0
        # self.isRender = render
        #
        # self.hz = 240
        # self.sim_dt = 1.0 / self.hz
        # self.frame_skip = 8
        #
        # self.agent = UR5Robot(dt= self.sim_dt*self.frame_skip)
        #
        # self._n_agents = 1
        # self.seed()
        # self._cam_dist = 1
        # self._cam_yaw = 0
        # self._cam_pitch = -30
        #
        # self.target_off_set=0.2
        # self.distance_threshold = distance_threshold #for success
        #
        # self.max_obs_dist_threshold = max_obs_dist
        # self.safe_dist_lowerlimit= dist_lowerlimit
        # self.safe_dist_upperlimit = dist_upperlimit
        #
        # # self.distance_threshold = distance_threshold
        # self.early_stop=early_stop
        # self.reward_type = reward_type
        #
        # self.n_actions = 3
        # self.action_space = gym.spaces.Box(-1., 1., shape=( self.n_actions,), dtype='float32')
        # self.observation_space = gym.spaces.Dict(dict(
        #     desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
        #     achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
        #     observation=gym.spaces.Box(-np.inf, np.inf, shape=(54,), dtype='float32'),
        # ))
        #
        # # Set observation and action spaces
        # self.agents_observation_space = self.agent.observation_space
        # self.agents_action_space = self.agent.action_space


    #
    # def reset(self):
    #     self.last_robot_joint = np.zeros(6)
    #     self.current_safe_dist = self._set_safe_distance()
    #
    #
    #     if (self.physicsClientId < 0):
    #         self.ownsPhysicsClient = True
    #
    #         if self.isRender:
    #             self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    #         else:
    #             self._p = bullet_client.BulletClient()
    #
    #         self._p.setGravity(0, 0, -9.81)
    #         self._p.setTimeStep(self.sim_dt)
    #
    #         self.physicsClientId = self._p._client
    #         self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    #
    #
    #     if self.scene is None:
    #         self.scene = self.create_single_player_scene(self._p)
    #     if not self.scene.multiplayer and self.ownsPhysicsClient:
    #         self.scene.episode_restart(self._p)
    #
    #     self.camera_adjust()
    #
    #     self.agent.scene = self.scene
    #
    #
    #     self.frame = 0
    #     self.iter_num = 0
    #
    #     self.robot_base = [0, 0, 0]
    #
    #     #---- random goal -----#
    #     self.goal=np.zeros(3)
    #     self.goal_orient = [0.0, 0.707, 0.0, -0.707]
    #
    #
    #     reset_success_flag = False
    #     while not reset_success_flag:
    #
    #         x = np.random.uniform(-0.6, 0.6)
    #         y = np.random.uniform(-0.6, 0.6)
    #         z = np.random.uniform(0.1, 0.5)
    #
    #         self.robot_start_eef = [x, y, z]
    #
    #         self.goal = self.random_set_goal()
    #
    #         #initial joint states from file
    #         initial_conf =[3.2999753952026367, -1.6784923712359827, 1.9284234046936035,
    #                        -1.791076962147848, -1.490676228200094, -0.0026128927813928726]
    #
    #         final_conf = [0.4617765545845032, -1.4414008299456995, 1.71844482421875,
    #                       -1.7910407225238245, -1.490664307271139, -0.002636734639303029]
    #
    #
    #         ar = self.agent.reset(self._p, client_id=self.physicsClientId,base_position=self.robot_base,
    #                                   base_rotation=[0, 0, 0, 1], eef_pose=self.robot_start_eef, joint_angle = initial_conf )
    #
    #         if ar is False:
    #             # print("failed to find valid robot solution of pose", robot_eef_pose)
    #             continue
    #
    #         self._p.stepSimulation()
    #         obs = self._get_obs()
    #         # collision_flag = self._contact_detection()
    #         # print("collision_flag is :", collision_flag)
    #
    #         # ---- for motion planner---#
    #         collisions = self._p.getContactPoints()
    #         if len(collisions) != 0:
    #             reset_success_flag = False
    #             continue
    #
    #
    #         start_time = time.time()
    #         result = self.motion_planner(initial_conf = initial_conf, final_conf = final_conf)
    #         print("use time: ", time.time()-start_time)
    #         if result is not None:
    #             (joint_path, cartesian_path) = result
    #             self.reference_traj = np.asarray(joint_path)
    #             reset_success_flag = True
    #         #
    #         #     print("cartesian_path", cartesian_path)
    #             #todo: draw cartesian path; and move through joint path
    #             # self.draw_path(cartesian_path)
    #
    #             self.goal = np.asarray(cartesian_path[2])
    #         print("initial conf",initial_conf)
    #         for obstacles in self.move_obstacles:
    #             obstacles.reset(self._p, self.goal)
    #         print("ar ", ar)
    #
    #     s = []
    #     s.append(ar)
    #     self._p.resetBasePositionAndOrientation(self.goal_id, posObj=self.goal, ornObj=[0, 0, 0, 1])
    #
    #
    #     return obs

    def draw_path(self, path):
        for i in range(len(path)-1):
            self._p.addUserDebugLine(path[i], path[i+1],
                                     lineColorRGB=[0.8,0.8,0.0],lineWidth=4,lifeTime=4)


    def get_planned_path(self):
        return self.reference_traj


    def motion_planner(self, initial_conf = None, final_conf =None):
        robot, ik_joints = self._test_moving_links_joints()
        workspace = [obstacles.id for obstacles in self.move_obstacles]
        robot_base_link_name = 'base_link'
        robot_base_link = link_from_name(robot, robot_base_link_name)

        ik_fn = ikfast_ur5.get_ik
        fk_fn = ikfast_ur5.get_fk

        # we have to specify ik fn wrapper and feed it into pychoreo
        def get_sample_ik_fn(robot, ik_fn, robot_base_link, ik_joints,tool_from_root=None):
            def sample_ik_fn(world_from_tcp):
                if tool_from_root:
                    world_from_tcp = multiply(world_from_tcp, tool_from_root)
                return sample_tool_ik(ik_fn, robot, ik_joints, world_from_tcp, robot_base_link, get_all=True)

            return sample_ik_fn

        collision_fn = get_collision_fn(robot, ik_joints,
                                        obstacles=workspace, attachments=[],
                                        self_collisions=True,
                                        #    disabled_collisions=disabled_collisions,
                                        #    extra_disabled_collisions=extra_disabled_collisions,
                                        custom_limits={})
        # Let's check if our ik sampler is working properly
        sample_ik_fn = get_sample_ik_fn(robot, ik_fn, robot_base_link, ik_joints)
        p_end = (self.goal, self.goal_orient)

        # # calculate initial ik and end ik
        # initial_conf = get_joint_positions(robot, ik_joints)
        #
        # #calculate end ik
        # qs = sample_ik_fn(p_end)
        # if collision_fn is not None:
        #
        #     conf_list = [conf for conf in qs if conf and not collision_fn(conf, diagnosis=True)]
        # try:
        #     conf_list[0]
        # except:
        #     return None
        #
        # n_conf_list = [normalize_conf(np.asarray(initial_conf), conf) for conf in conf_list]
        # final_conf = min_dist_conf(initial_conf, n_conf_list)

        # print("initial conf: ", initial_conf)
        # print("goal is: ", self.goal)
        # print("fina_conf: ", final_conf)

        # set robot to initial configuration
        if initial_conf is not None:
            initial_conf = initial_conf
        else:
            initial_conf =[3.2999753952026367, -1.6784923712359827, 1.9284234046936035,
                           -1.791076962147848, -1.490676228200094, -0.0026128927813928726]

        if final_conf is not None:
            final_conf = final_conf
        else:
            final_conf =[0.4617765545845032, -1.4414008299456995, 1.71844482421875,
                         -1.7910407225238245, -1.490664307271139, -0.002636734639303029]


        set_joint_positions(robot, ik_joints, initial_conf)

        # start planning
        path = plan_joint_motion(robot, ik_joints, final_conf, obstacles=workspace,
                                 self_collisions=True, diagnosis=False)

        #set robot to initial configuration
        set_joint_positions(robot, ik_joints, initial_conf)

        if path is None:
            return None
        else:
            cartesion_path = [compute_forward_kinematics(fk_fn, conf)[0] for conf in path]

        # rotate z for 180 degree, x=-x, y = -y
        for p in cartesion_path:
            p[0]=-p[0]
            p[1]=-p[1]

        return (path, cartesion_path)


    def _test_moving_links_joints(self):
        robot = self.agent.robot_body.bodies[0]
        # workspace = [self.move_obstacles[0].id, self.move_obstacles[1].id]
        assert isinstance(robot, int)

        movable_joints = get_movable_joints(robot)
        assert isinstance(movable_joints, list) and all([isinstance(mj, int) for mj in movable_joints])
        assert 6 == len(movable_joints)
        assert [b'shoulder_pan_joint', b'shoulder_lift_joint', b'elbow_joint', b'wrist_1_joint', b'wrist_2_joint',
                b'wrist_3_joint'] == \
               get_joint_names(robot, movable_joints)

        return robot, movable_joints

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)

        #collision
        _is_collision = info['is_collision']
        if isinstance(_is_collision, np.ndarray):
            _is_collision = _is_collision.flatten()

        #safe distance
        min_dist = info['min_dist']
        safe_dist = info ['safe_threshold']

        if isinstance(min_dist, np.ndarray):
            min_dist = min_dist.flatten()
            safe_dist = safe_dist.flatten()

        distance = safe_dist - min_dist
        if isinstance(distance, np.ndarray):
            distance[(distance < 0)] = 0
        else:
            distance = 0 if distance<0 else distance


        return 0


