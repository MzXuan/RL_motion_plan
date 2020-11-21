import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, currentdir)
from humanoid import SelfMoveHumanoid
from ur5_real import UR5RealRobot
from ur5eef import UR5EefRobot
from humanoid_real import RealHumanoid
from .ur5_dynamic_reach import UR5DynamicReachEnv


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

from pkg_resources import parse_version

from pybullet_planning import link_from_name, get_moving_links, get_link_name
from pybullet_planning import get_joint_names, get_movable_joints
from pybullet_planning import multiply, get_collision_fn
from pybullet_planning import sample_tool_ik
from pybullet_planning import set_joint_positions
from pybullet_planning import get_joint_positions, plan_joint_motion, compute_forward_kinematics


import ikfast_ur5



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

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)



class UR5RealTestEnv(gym.Env):
    def __init__(self, render=False, max_episode_steps=3000,
                 early_stop=False,  distance_threshold=0.05,
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

        self.agents = [UR5RealRobot(3, ), RealHumanoid(max_obs_dist)]

        self._n_agents = 2
        self.seed()
        self._cam_dist = 3
        self._cam_yaw = 0
        self._cam_pitch = 30

        self.target_off_set=0.2


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

        #--- following demo----
        self.demo_data = load_demo()
        # path = [self.demo_data[i]['toolp'] for i in range(len(self.demo_data))]
        # vel_path = [self.demo_data[i]['tool_v'] for i in range(len(self.demo_data))]
        # self.ws_path_gen = WsPathGen(path, vel_path)
        self.sphere_radius=0.1
        self.hand_velocity=np.zeros(3)


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

        move_to_start(self.agents[0].ur5_rob_control, self.demo_data)


        #---------------real human----------------------------#
        ah = self.agents[1].reset(self._p)
        # #----real robot-----------------------------------------------#
        if self.first_reset is True:
            ar = self.agents[0].reset(self._p, client_id=self.physicsClientId, base_position=self.robot_base)
        else:
            ar = self.agents[0].calc_state()


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

    def draw_path(self, path):
        for i in range(len(path)-1):
            self._p.addUserDebugLine(path[i], path[i+1],
                                     lineColorRGB=[0.8,0.8,0.0],lineWidth=4 )


    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = PlaneScene(bullet_client, gravity=0, timestep=self.sim_dt, frame_skip=self.frame_skip)

        self.long_table_body = bullet_client.loadURDF(os.path.join(assets.getDataPath(), "scenes_data", "longtable/longtable.urdf"),
                               [-1, -0.9, -1.0],
                               [0.000000, 0.000000, 0.0, 1])

        # # add target box goals
        # self.box_ids = []
        # self.box_pos = []
        #
        # self.human_pos = []
        # for i in range(6):
        #     for j in range(2):
        #         x = (i - 2.5) / 5
        #         y = (j - 3.5) / 5
        #         z = 0
        #
        #         id_temp = bullet_client.loadURDF(os.path.join(assets.getDataPath(),
        #                                                       "scenes_data", "targetbox/targetbox.urdf"),
        #                                          [x, y, z], [0.000000, 0.000000, 0.0, 0.1])
        #         if j >0 :
        #             self.box_ids.append(id_temp)
        #             self.box_pos.append([x, y, z])
        #
        #
        #         bullet_client.changeVisualShape(id_temp, -1, rgbaColor=[1,1,0,1])
        #         self.human_pos.append([x,y,z])



        self.goal_id = bullet_client.loadURDF(
            os.path.join(assets.getDataPath(), "scenes_data", "targetball/targetball.urdf"),
            [0, 0, 0],
            [0.000000, 0.000000, 0.0, 1])

        #----for debug-------------------------------------------
        # self.agents[1].set_goal_position(self.human_pos)


        return self.stadium_scene

    def _set_safe_distance(self):
            return 0.1

    def get_obs(self):
        return self._get_obs()

    def close(self):
        self.agents[0].close()

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
        arm_states = self.agents[1].calc_state()

        infos['succeed'] = dones

        self._p.addUserDebugLine(self.last_robot_eef, ur5_eef_position, \
                                 lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=10)
        self.last_robot_eef = ur5_eef_position

        #------------------------------------------------------------------

        # ------------------------------------------------------------------
        obs_human_states = []
        min_dists = []
        for arm_s in arm_states:

            d = [np.linalg.norm([p - ur5_eef_position]) for p in arm_s["current"]]

            min_dist = np.min(np.asarray(d))
            if min_dist > self.max_obs_dist_threshold:
                min_dist = self.max_obs_dist_threshold
            min_dists.append(min_dist)

            for p in arm_s["current"]:
                if np.linalg.norm([p - ur5_eef_position]) > self.max_obs_dist_threshold:
                    obs_human_states.append(np.zeros(3) + self.max_obs_dist_threshold)
                else:
                    obs_human_states.append(p - ur5_eef_position)

            for p in arm_s["next"]:
                if np.linalg.norm([p - ur5_eef_position]) > self.max_obs_dist_threshold:
                    obs_human_states.append(np.zeros(3) + self.max_obs_dist_threshold)
                else:
                    obs_human_states.append(p - ur5_eef_position)

            # for p in arm_state["next2"]:
            #     if np.linalg.norm([p-ur5_eef_position]) >  self.max_obs_dist_threshold:
            #         obs_human_states.append(np.zeros(3)+self.max_obs_dist_threshold)
            #     else:
            #         obs_human_states.append(p-ur5_eef_position)

            # print("obs human states: ", obs_human_states)

            # todo: need to update velocity for reactive method
            if (arm_s['current'][2] == self.max_obs_dist_threshold).all() \
                    and (arm_s['next'][2] == self.max_obs_dist_threshold).all():
                self.hand_velocity = np.zeros(3)
            else:
                self.hand_velocity = (arm_s['next'][2] - arm_s['current'][2]) / 0.033

        try:
            self.goal, _ = self.ws_path_gen.next_goal(ur5_eef_position, self.sphere_radius)
        except:
            print("!!!!!!!!!!!!!!not exist self.ws_path_gen")

        self.obs_min_dist = np.min(np.asarray(min_dists))

        obs = np.concatenate([np.asarray(ur5_states), np.asarray(obs_human_states).flatten(),
                              np.asarray(self.goal).flatten(), np.asarray([self.obs_min_dist])])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }



    def set_sphere(self, r):
        self.sphere_radius = r


    def get_hand_velocity(self):
        return self.hand_velocity


    def step(self, actions):
        self.iter_num += 1


        self.agents[0].apply_action(actions)

        self.scene.global_step()

        obs = self._get_obs()

        done = False

        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.final_goal),
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
        self._p.resetBasePositionAndOrientation(self.goal_id, posObj=self.goal, ornObj=[0.0, 0.0, 0.0, 1.0])

        return obs, reward, done, info


    def set_render(self):
        self.physicsClientId = -1
        self.isRender = True

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

    def set_training(self, is_training):
        self.is_training = is_training

    def configure(self, args):
        self.agents[0].args = args[0]
        self.agents[1].args = args[1]

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)

        for a in self.agents:
            a.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def compute_reward(self, achieved_goal, goal, info):
        return 0

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

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed



class UR5RealPlanTestEnv(UR5RealTestEnv):
    def __init__(self, render=False, max_episode_steps=3000,
                 early_stop=False,  distance_threshold=0.05,
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

        self.agents = [UR5RealRobot(6, ), RealHumanoid(max_obs_dist)]

        self._n_agents = 2
        self.seed()
        self._cam_dist = 3
        self._cam_yaw = 0
        self._cam_pitch = 30

        self.target_off_set=0.2


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

        #--- following demo----
        self.demo_data = load_demo()
        # path = [self.demo_data[i]['toolp'] for i in range(len(self.demo_data))]
        # vel_path = [self.demo_data[i]['tool_v'] for i in range(len(self.demo_data))]
        # self.ws_path_gen = WsPathGen(path, vel_path)
        self.sphere_radius=0.1
        self.hand_velocity=np.zeros(3)


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

        move_to_start(self.agents[0].ur5_rob_control, self.demo_data)


        #---------------real human----------------------------#
        ah = self.agents[1].reset(self._p)
        # #----real robot-----------------------------------------------#
        if self.first_reset is True:
            ar = self.agents[0].reset(self._p, client_id=self.physicsClientId, base_position=self.robot_base)
        else:
            ar = self.agents[0].calc_state()


        # #------prepare path-----------------
        # path = [self.demo_data[i]['toolp'] for i in range(len(self.demo_data))]
        # vel_path = [self.demo_data[i]['toolv'] for i in range(len(self.demo_data))]
        # self.ws_path_gen = WsPathGen(path, vel_path, self.distance_threshold)
        # self.draw_path(path)


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


            self.goal = np.asarray(cartesian_path[2])
        print("initial conf", initial_conf)


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
        arm_states = self.agents[1].calc_state()

        infos['succeed'] = dones

        self._p.addUserDebugLine(self.last_robot_eef, ur5_eef_position, \
                                 lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=10)
        self.last_robot_eef = ur5_eef_position

        #------------------------------------------------------------------
        obs_human_states = []
        min_dists = []
        for arm_s in arm_states:

            d = [np.linalg.norm([p-ur5_eef_position]) for p in arm_s["current"]]

            min_dist = np.min(np.asarray(d))
            if min_dist >  self.max_obs_dist_threshold:
                min_dist =  self.max_obs_dist_threshold
            min_dists.append(min_dist)


            for p in arm_s["current"]:
                if np.linalg.norm([p - ur5_eef_position]) > self.max_obs_dist_threshold:
                    obs_human_states.append(np.zeros(3) + self.max_obs_dist_threshold)
                else:
                    obs_human_states.append(p - ur5_eef_position)

            for p in arm_s["next"]:
                if np.linalg.norm([p-ur5_eef_position]) >  self.max_obs_dist_threshold:
                    obs_human_states.append(np.zeros(3)+self.max_obs_dist_threshold)
                else:
                    obs_human_states.append(p-ur5_eef_position)

            # for p in arm_state["next2"]:
            #     if np.linalg.norm([p-ur5_eef_position]) >  self.max_obs_dist_threshold:
            #         obs_human_states.append(np.zeros(3)+self.max_obs_dist_threshold)
            #     else:
            #         obs_human_states.append(p-ur5_eef_position)

            # print("obs human states: ", obs_human_states)

            #todo: need to update velocity for reactive method
            if (arm_s['current'][2] == self.max_obs_dist_threshold).all() \
                    and (arm_s['next'][2] ==  self.max_obs_dist_threshold).all():
                self.hand_velocity = np.zeros(3)
            else:
                self.hand_velocity = (arm_s['next'][2] - arm_s['current'][2])/0.033

        self.obs_min_dist = np.min(np.asarray(min_dists))

        obs = np.concatenate([np.asarray(ur5_states), np.asarray(obs_human_states).flatten(),
                              np.asarray(self.goal).flatten(), np.asarray([self.obs_min_dist])])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def get_planned_path(self):
        return self.reference_traj

    def motion_planner(self, initial_conf=None, final_conf=None):
        robot, ik_joints = self._test_moving_links_joints()
        workspace = self.agents[1].arm_id
        robot_base_link_name = 'base_link'
        robot_base_link = link_from_name(robot, robot_base_link_name)

        ik_fn = ikfast_ur5.get_ik
        fk_fn = ikfast_ur5.get_fk

        # we have to specify ik fn wrapper and feed it into pychoreo
        def get_sample_ik_fn(robot, ik_fn, robot_base_link, ik_joints, tool_from_root=None):
            def sample_ik_fn(world_from_tcp):
                if tool_from_root:
                    world_from_tcp = multiply(world_from_tcp, tool_from_root)
                return sample_tool_ik(ik_fn, robot, ik_joints, world_from_tcp, robot_base_link, get_all=True)

            return sample_ik_fn

        # set robot to initial configuration
        if initial_conf is not None:
            initial_conf = initial_conf
        else:
            initial_conf = self.demo_data[0]['robjp']

        if final_conf is not None:
            final_conf = final_conf
        else:
            final_conf = self.demo_data[-1]['robjp']

        set_joint_positions(robot, ik_joints, initial_conf)

        # start planning
        path = plan_joint_motion(robot, ik_joints, final_conf, obstacles=[workspace],
                                 self_collisions=True, diagnosis=False)

        # set robot to initial configuration
        set_joint_positions(robot, ik_joints, initial_conf)

        if path is None:
            return None
        else:
            cartesion_path = [compute_forward_kinematics(fk_fn, conf)[0] for conf in path]

        # rotate z for 180 degree, x=-x, y = -y
        for p in cartesion_path:
            p[0] = -p[0]
            p[1] = -p[1]

        return (path, cartesion_path)

    def _test_moving_links_joints(self):
        robot = self.agents[0].robot_body.bodies[0]
        workspace = self.agents[1].arm_id
        assert isinstance(robot, int) and isinstance(workspace, int)

        movable_joints = get_movable_joints(robot)
        assert isinstance(movable_joints, list) and all([isinstance(mj, int) for mj in movable_joints])
        assert 6 == len(movable_joints)
        assert [b'shoulder_pan_joint', b'shoulder_lift_joint', b'elbow_joint', b'wrist_1_joint', b'wrist_2_joint',
                b'wrist_3_joint'] == \
               get_joint_names(robot, movable_joints)

        return robot, movable_joints




