from .fetch_reach_new import FetchReachV2Env
import os
from gym import utils as gym_utils
import numpy as np
from gym.envs.robotics import rotations, robot_env, utils


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach_xz.xml')


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchDynamicReachTestEnv(FetchReachV2Env, gym_utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        FetchReachV2Env.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.4, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, early_stop=True)
        utils.EzPickle.__init__(self)


    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()

        self.time_step = 0
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        self.last_dist = goal_distance(
            obs['achieved_goal'], self.goal)
        return obs

    # def _sample_trajectory(self, goal):
        


    def _get_obs(self):
        #todo: follow path goal
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obstable_center_lst = []
        obstacle_rel_pos_lst = []
        obstacle_v_lst = []

        for obstacle in self.obstacles_cls:
            body_num = self.sim.model.body_name2id(obstacle.name)
            obstable_center = self.sim.model.body_pos[body_num]
            obstable_center_lst.append(obstable_center)
            obstacle_rel_pos_lst.append(obstable_center - grip_pos)
            obstacle_v_lst.append(obstacle.moving_speed)

        achieved_goal = grip_pos.copy()

        obs = np.concatenate([
            grip_pos, np.asarray(obstable_center_lst).ravel(), np.asarray(obstacle_rel_pos_lst).ravel(), gripper_state,
            np.asarray(obstacle_v_lst).ravel(), grip_velp, gripper_vel,
        ])


        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }