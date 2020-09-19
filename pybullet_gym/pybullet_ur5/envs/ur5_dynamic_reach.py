import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, currentdir)

import assets
from scenes.stadium import StadiumScene, PlaneScene
from humanoid import EefHumanoid, SelfMoveHumanoid, SelfMoveAwayHumanoid
from ur5_rg2 import UR5RG2Robot
from ur5eef import UR5EefRobot

import gym, gym.spaces, gym.utils, gym.utils.seeding
from gym.spaces import Tuple
import numpy as np
import pybullet
from pybullet_utils import bullet_client

import utils
import random

# from pybullet_planning import link_from_name, get_moving_links, get_link_name
# from pybullet_planning import get_joint_names, get_movable_joints
# from pybullet_planning import multiply, get_collision_fn
# from pybullet_planning import sample_tool_ik
# from pybullet_planning import set_joint_positions
# from pybullet_planning import get_joint_positions, plan_joint_motion, compute_forward_kinematics
#
#
# import ikfast_ur5

from scipy.interpolate import griddata

from pkg_resources import parse_version

from mpi4py import MPI


def calculate_bc_loss(p_rob, reference_traj):
    dist = []
    for p_ref in reference_traj:
        dist.append(np.linalg.norm([p_rob-p_ref]))
    dist = np.asarray(dist)
    min_dist = np.min(dist)
    progress = np.argmin(dist)/len(dist)
    return min_dist, progress

def calculate_bc_ratio(p_rob, reference_traj, ph, pl):
    dist = []
    for p_ref in reference_traj:
        dist.append(np.linalg.norm([p_rob-p_ref]))
    dist = np.asarray(dist)
    min_dist = np.min(dist)
    progress = np.argmin(dist) / len(dist)

    if min_dist < pl:
        ratio = 0
    elif min_dist>ph:
        ratio = 1
    else:
        ratio = ((min_dist-pl)/(ph-pl))**2
    return (1-ratio), progress

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


class UR5DynamicReachEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 60}

    def __init__(self, render=False, max_episode_steps=1000,
                 early_stop=True,  distance_threshold=0.03, reward_type="sparse"):
        self.distance_close = 0.3

        self.iter_num = 0
        self.max_episode_steps = max_episode_steps

        self.scene = None
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self.isRender = render
        # self.agents = [UR5RG2Robot(), SelfMoveHumanoid(0, 12)]
        # self.agents = [UR5Robot(), SelfMoveAwayHumanoid(0, 12)]
        self.agents = [UR5EefRobot(3,), SelfMoveHumanoid(0, 12)]

        self._n_agents = 2
        self.seed()
        self._cam_dist = 3
        self._cam_yaw = 0
        self._cam_pitch = 30
        self._render_width = 320
        self._render_height = 240
        self.is_training = True


        self.target_off_set=0.2
        self.safe_dist_threshold = 0.6

        self.distance_threshold = distance_threshold
        self.early_stop=early_stop
        self.reward_type = reward_type



        # self.reset()
        # self.goal = np.zeros(3)
        # self.goal_orient = [0.0, 0.707, 0.0, 0.707]
        # obs = self._get_obs()
        self.action_space = gym.spaces.Box(-1., 1., shape=(3,), dtype='float32')
        # self.observation_space = gym.spaces.Dict(dict(
        #     desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        #     achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        #     observation=gym.spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        # ))

        self.observation_space = gym.spaces.Dict(dict(
            desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            observation=gym.spaces.Box(-np.inf, np.inf, shape=(35,), dtype='float32'),
        ))

        print("self.observation space: ", self.observation_space)


        # Set observation and action spaces
        self.agents_observation_space = Tuple([
            agent.observation_space for agent in self.agents
        ])
        self.agents_action_space = Tuple([
            agent.action_space for agent in self.agents
        ])

        # obs_dim, act_dim = 0, 0
        # for agent in self.agents:
        #     obs_dim += agent.observation_space.shape[0]
        #     act_dim += agent.action_space.shape[0]
        #
        # obs_dim+=22
        #
        # high = np.ones([act_dim])
        # self.action_space = gym.spaces.Box(-high, high)
        # high = np.inf * np.ones([obs_dim])
        # self.observation_space = gym.spaces.Box(-high, high)



    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = PlaneScene(bullet_client, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)

        # self.long_table_body = bullet_client.loadURDF(os.path.join(assets.getDataPath(), "scenes_data", "longtable/longtable.urdf"),
        #                        [-1, -0.9, -1.0],
        #                        [0.000000, 0.000000, 0.0, 1])

        # add target box goals
        self.box_ids = []
        self.box_pos = []

        self.human_pos = []
        for i in range(6):
            for j in range(2):
                x = (i - 3.0) / 5
                y = (j - 4) / 6
                z = 0

                id_temp = bullet_client.loadURDF(os.path.join(assets.getDataPath(),
                                                              "scenes_data", "targetbox/targetbox.urdf"),
                                                 [x, y, z], [0.000000, 0.000000, 0.0, 1])
                if j == 1:
                    self.box_ids.append(id_temp)
                    self.box_pos.append([x, y, z])


                bullet_client.changeVisualShape(id_temp, -1, rgbaColor=[1,1,0,1])
                self.human_pos.append([x,y,z])



        self.goal_id = bullet_client.loadURDF(
            os.path.join(assets.getDataPath(), "scenes_data", "targetball/targetball.urdf"),
            [0, 0, 0],
            [0.000000, 0.000000, 0.0, 1])

        self.agents[1].set_goal_position( self.human_pos)

        return self.stadium_scene

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

    def reset(self):
        self.last_human_eef = [0, 0, 0]
        self.last_robot_eef = [0, 0, 0]
        self.last_robot_joint = np.zeros(6)


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

        # ---select goal from boxes---#
        self.goal = np.asarray(random.choice(self.box_pos)) #robot goal
        self.goal[2]+=0.1
        self.goal_orient = [0.0, 0.707, 0.0, 0.707]


        collision_flag = True
        while collision_flag:
            print("contact reset")
            # # ---random goal in space---#
            # self.goal = np.zeros(3)
            # while np.linalg.norm(self.goal) < 0.3:
            #     self.goal = np.asarray([np.random.uniform(-0.6, 0.6),
            #                             np.random.uniform(-0.5, -0.1),
            #                             np.random.uniform(-0.2, 0) + self.target_off_set])
            #
            # self.goal_orient = [0.0, 0.707, 0.0, 0.707]

            x = np.random.uniform(0.3, 0.45)
            y = np.random.uniform(-0.5, -0.4)
            z = np.random.uniform(0.2, 0.3)

            robot_eef_pose = [np.random.choice([-1, 1])*x, y, z]

            self.robot_start_eef = robot_eef_pose

            s = []
            ar = self.agents[0].reset(self._p, base_position=self.robot_base, base_rotation=[0, 0, 0, 1], eef_pose=self.robot_start_eef)
            if ar is False:
                continue
            ah = self.agents[1].reset(self._p, base_position=[0.0, -1.2, -1.1], \
                                      base_rotation=[0, 0, 0.7068252, 0.7073883], is_training=self.is_training)
            s.append(ar)
            s.append(ah)

            obs = self._get_obs()
            # self.step()
            collision_flag = self._contact_detection()


        return obs




    def _is_close(self, p, threshold = 0.3):
        dist = np.linalg.norm((np.asarray(self.robot_base)-np.asarray(p)))
        if dist < 0.4 or dist > 1.0:
            return True

        for pp in self.box_pos:
            dist = np.linalg.norm((pp - np.asarray(p)))
            if dist < threshold:
                return True
        else:
            return False

    def render(self, mode='human', close=False):
        if mode == "human":
            self.isRender = True
        if mode != "rgb_array":
            return np.array([])

        base_pos = [0, 0, 10]
        if (hasattr(self, 'robot')):
            if (hasattr(self.robot, 'body_xyz')):
                base_pos = self.robot.body_xyz

        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                distance=self._cam_dist,
                                                                yaw=self._cam_yaw,
                                                                pitch=self._cam_pitch,
                                                                roll=0,
                                                                upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                         aspect=float(self._render_width) /
                                                                self._render_height,
                                                         nearVal=0.1,
                                                         farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(width=self._render_width,
                                                  height=self._render_height,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        if (self.ownsPhysicsClient):
            if (self.physicsClientId >= 0):
                self._p.disconnect()
        self.physicsClientId = -1

    def step(self, actions):
        self.iter_num += 1

        for i, agent in enumerate(self.agents):
            act_dim = self.agents_action_space[i].shape[0]
            if i == 0:
                act = actions[:act_dim]
            elif i == 1:
                act = actions[-act_dim:]
            else:
                print("ERROR: Cannot support more than two agents!")
                return False

            agent.apply_action(act)

        self.scene.global_step()

        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'is_collision': self._contact_detection(),
            'alternative_goals': obs['observation'][-6:]
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
        ur5_eef_oriention = ur5_states[3:7]
        humanoid_states = self.agents[1].calc_state()

        humanoid_eef_position = humanoid_states[3:6]
        infos['succeed'] = dones

        rob_joint_state = ur5_states[7:13]



        # ------ drawing ------#
        # check human hand and elbow position
        self._p.addUserDebugLine(humanoid_states[-3:], humanoid_states[-6:-3], \
                                 lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=10)

        self._p.addUserDebugLine(self.last_human_eef, humanoid_eef_position, \
                                 lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=10)
        self._p.addUserDebugLine(self.last_robot_eef, ur5_eef_position, \
                                 lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=10)

        # todo: warning. update may need to replace
        self.last_human_eef = humanoid_eef_position
        self.last_robot_eef = ur5_eef_position
        self.last_robot_joint = rob_joint_state


        self.robot_current_p = ((ur5_eef_position), (ur5_eef_oriention))

        # human to robot base minimum distance
        pts = self._p.getClosestPoints(self.agents[0].robot_body.bodies[0],
                                       self.agents[1].robot_body.bodies[0], self.safe_dist_threshold)
        min_dist = self.safe_dist_threshold
        for i in range(len(pts)):
            if pts[i][8] < min_dist:
                min_dist = pts[i][8]
        if len(pts) == 0:
            # normalize
            obs_human_states = np.ones(len(humanoid_states))
            # obs_human_states[0:6] = humanoid_states[0:6] / np.linalg.norm(humanoid_states[0:6]) * self.safe_dist_threshold
        else:
            # add reward according to min_distance
            obs_human_states = humanoid_states


        achieved_goal = ur5_eef_position

        obs = np.concatenate([np.asarray(ur5_states), np.asarray(obs_human_states),
                              np.asarray(self.goal).flatten(), np.asarray([min_dist])])


        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }



    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        # force_reward = self._contact_force()
        _is_collision = info['is_collision']
        # print("_is_collision: ", _is_collision)
        if isinstance(_is_collision, np.ndarray):
            _is_collision = _is_collision.flatten()

        a1 = -1
        a2 = -5
        if self.reward_type == 'sparse':
            reward = a1 * (d > self.distance_threshold).astype(np.float32) \
                     + a2 * (_is_collision > 0)
        else:
            reward = a1 * d + a2 * (_is_collision > 0)
        return reward


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
            #
            # print("robot parts",self.agents[0].parts)


            if c[3] == 3 and c[4] == 5:
                continue
            if c[3] == 0 or c[4] == 0:
                continue
            # p = self._p.getLinkState(c[1], c[3])[0]
            collision_bodies.append(bodyinfo1[1].decode("utf-8"))
            collision_bodies.append(bodyinfo2[1].decode("utf-8"))

        # todo: check collision bodies

        # print("collision_bodies: ", collision_bodies)

        if len(collision_bodies) != 0:
            if "ur5" in collision_bodies:  # robot collision
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


