import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, currentdir)
from humanoid import SelfMoveHumanoid
from ur5_real import UR5RealRobot
from .ur5_dynamic_reach import UR5DynamicReachEnv

import gym, gym.spaces, gym.utils, gym.utils.seeding
from gym.spaces import Tuple
import numpy as np
import random
import pybullet
from pybullet_utils import bullet_client
import time

class UR5RealTestEnv(UR5DynamicReachEnv):
    def __init__(self, render=False, max_episode_steps=1000,
                 early_stop=True,  distance_threshold=0.03,
                 max_obs_dist=0.5, dist_lowerlimit=0.05, dist_upperlimit=0.3,
                 reward_type="sparse"):

        self.distance_close = 0.3

        self.collision_weight = 0
        self.iter_num = 0
        self.max_episode_steps = max_episode_steps

        self.scene = None
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self.isRender = render

        self.agents = [UR5RealRobot(3, ), SelfMoveHumanoid(0, 12, noise=False, move_base=False)]
        # self.agents = [UR5EefRobot(3, ),
        #                SelfMoveHumanoid(0, 12, is_training=True, move_base=True, noise=True)]

        self._n_agents = 2
        self.seed()
        self._cam_dist = 3
        self._cam_yaw = 0
        self._cam_pitch = 30
        self._render_width = 320
        self._render_height = 240

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
            observation=gym.spaces.Box(-np.inf, np.inf, shape=(35,), dtype='float32'),
        ))



        # Set observation and action spaces
        self.agents_observation_space = Tuple([
            agent.observation_space for agent in self.agents
        ])
        self.agents_action_space = Tuple([
            agent.action_space for agent in self.agents
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

        # ---select goal from boxes---#
        self.goal = np.asarray(random.choice(self.box_pos)) #robot goal
        self.goal[0] += np.random.uniform(-0.15, 0.15)
        self.goal[1] += np.random.uniform(-0.2, 0)
        self.goal[2]+=np.random.uniform(0.1,0.3)
        self.goal_orient = [0.0, 0.707, 0.0, 0.707]
        ah = self.agents[1].reset(self._p, base_position=[0.0, -1.1, -1.1],
                                  base_rotation=[0, 0, 0.7068252, 0.7073883])

        if self.first_reset is True:
            ar = self.agents[0].reset(self._p, client_id=self.physicsClientId, base_position=self.robot_base)
        else:
            ar = self.agents[0].calc_state()

        self._p.stepSimulation()
        obs = self._get_obs()

        s = []
        s.append(ar)
        s.append(ah)
        self._p.resetBasePositionAndOrientation(self.goal_id, posObj=self.goal, ornObj=[0.0, 0.0, 0.0, 1.0])

        return obs

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
        humanoid_states = self.agents[1].calc_state()

        infos['succeed'] = dones



        # todo: warning. update may need to replace
        # self.last_human_eef = humanoid_eef_position
        # self.last_robot_eef = ur5_eef_position
        # self.last_robot_joint = rob_joint_state
        #
        # self.robot_current_p = ((ur5_eef_position), (ur5_eef_oriention))

        # human to robot base minimum distance

        pts = self._p.getClosestPoints(self.agents[0].robot_body.bodies[0],
                                       self.agents[1].robot_body.bodies[0],
                                       distance=self.max_obs_dist_threshold)  # max perception threshold
        min_dist = self.max_obs_dist_threshold

        # print("pts: ", pts)
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
        self.obs_min_safe_dist = min_dist

        obs = np.concatenate([np.asarray(ur5_states), np.asarray(obs_human_states),
                              np.asarray(self.goal).flatten(), np.asarray([min_dist])])


        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }



