import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, currentdir)
from humanoid import SelfMoveHumanoid
from ur5eef import UR5EefRobot
from .ur5_dynamic_reach import UR5DynamicReachEnv

import gym, gym.spaces, gym.utils, gym.utils.seeding
from gym.spaces import Tuple
import numpy as np


class UR5DynamicTestEnv(UR5DynamicReachEnv):
    def __init__(self, render=False, max_episode_steps=1000,
                 early_stop=True,  distance_threshold=0.03, reward_type="sparse"):

        self.distance_close = 0.3

        self.collision_weight = 0
        self.iter_num = 0
        self.max_episode_steps = max_episode_steps

        self.scene = None
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self.isRender = render
        # self.agents = [UR5RG2Robot(), SelfMoveHumanoid(0, 12)]
        # self.agents = [UR5Robot(), SelfMoveAwayHumanoid(0, 12)]
        self.agents = [UR5EefRobot(3, ),
                       SelfMoveHumanoid(0, 12, is_training=True, move_base=True, noise=True)]
        self._n_agents = 2
        self.seed()
        self._cam_dist = 3
        self._cam_yaw = 0
        self._cam_pitch = 30
        self._render_width = 320
        self._render_height = 240

        self.target_off_set=0.2
        self.safe_dist_threshold = 0.6

        self.distance_threshold = distance_threshold
        self.early_stop=early_stop
        self.reward_type = reward_type


        self.n_actions=3
        self.action_space = gym.spaces.Box(-1., 1., shape=( self.n_actions,), dtype='float32')


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

        #
        # super(UR5DynamicTestEnv, self).__init__(render=render, max_episode_steps=max_episode_steps,
        #          early_stop=early_stop,  distance_threshold=distance_threshold, reward_type=reward_type)
