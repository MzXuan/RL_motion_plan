from .fetch_reach_new import FetchReachV2Env
import os
from gym import utils as gym_utils
import numpy as np
from gym.envs.robotics import rotations, robot_env, utils
from .ws_path_gen import WsPathGen


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach_xz_test.xml')


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
            obj_range=0.15, target_range=0.4, distance_threshold=0.04,
            initial_qpos=initial_qpos, reward_type=reward_type, early_stop=True, obstacle_speed=0.01)
        gym_utils.EzPickle.__init__(self)
        self.sphere_radius = 0.05

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
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        self.final_goal = self._sample_goal().copy()

        self.ws_path_gen = WsPathGen(grip_pos, self.final_goal)
        self.goal = self.ws_path_gen.next_goal(grip_pos, self.sphere_radius)

        obs = self._get_obs()
        self.last_dist = goal_distance(
            obs['achieved_goal'], self.goal)

        # send goal to obstacles
        for obstacle in self.obstacles_cls:
            obstacle.reset_obstacle(self.sim, robot_goal = None, robot_current=grip_pos)
        return obs

    def _sample_goal(self):
        #keep distance to end-effector
        body_num = self.sim.model.body_name2id('table0')
        box_center = self.sim.model.body_pos[body_num]
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        keep_distance = False
        while not keep_distance:
            goal = np.zeros(3)
            goal[0] = box_center[0] + np.random.uniform(-0.18, 0.18)
            goal[1] = box_center[1] + np.random.uniform(-0.25, 0.25)
            goal[2] = box_center[2] + np.random.uniform(0.25, 0.38)
            if np.linalg.norm([goal-grip_pos]) >0.2:
                keep_distance = True

        return goal.copy()


    # def _sample_trajectory(self, goal):
    def public_get_obs(self):
        return self._get_obs()

    def set_sphere_radius(self, r):
        self.sphere_radius = r

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        # d = goal_distance(achieved_goal, goal)
        _is_collision = info['is_collision']
        if isinstance(_is_collision, np.ndarray):
            _is_collision = _is_collision.flatten()

        reward = -12*(_is_collision>0)
        return reward

    def _get_obs(self):
        #todo: follow path goal
        #todo: reduce deminision of grippers
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
        obstacle_rel_v_lst = []

        for obstacle in self.obstacles_cls:
            body_num = self.sim.model.body_name2id(obstacle.name)
            obstable_center = self.sim.model.body_pos[body_num]
            obstable_center_lst.append(obstable_center)
            obstacle_rel_pos_lst.append(obstable_center - grip_pos)
            obstacle_v_lst.append(obstacle.moving_speed)
            obstacle_rel_v_lst.append(obstacle.moving_speed - grip_velp)

        achieved_goal = grip_pos.copy()

        try:
            self.goal = self.ws_path_gen.next_goal(grip_pos, self.sphere_radius)
        except:
            print("!!!!!!!!!!!!!!not exist self.ws_path_gen")

        # print("obstacle_rel_pos_lst: ", obstacle_rel_pos_lst)

        obs = np.concatenate([
            grip_pos, grip_velp,
            np.asarray(obstacle_rel_pos_lst).ravel(), np.asarray(obstacle_rel_v_lst).ravel()])

        # obs = np.concatenate([
        #     grip_pos, np.asarray(obstable_center_lst).ravel(), np.asarray(obstacle_rel_pos_lst).ravel(),
        #     np.asarray(obstacle_v_lst).ravel(), grip_velp
        # ])

        # obs = np.concatenate([
        #     grip_pos, np.asarray(obstable_center_lst).ravel(), np.asarray(obstacle_rel_pos_lst).ravel(), gripper_state,
        #     np.asarray(obstacle_v_lst).ravel(), grip_velp, gripper_vel,
        # ])


        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }



    def step(self, action):
        self.time_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        for obstacle in self.obstacles_cls:
            obstacle.move_obstacle(self.sim, self.goal)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()


        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.final_goal),
            'is_collision': self._contact_dection(),
            'alternative_goals': obs['observation'][-6:]
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

        if self.early_stop:
            if info["is_success"] or info["is_collision"]:
                done = True
        return obs, reward, done, info



    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        site_id = self.sim.model.site_name2id('finaltarget')
        self.sim.model.site_pos[site_id] = self.final_goal - sites_offset[0]
        self.sim.forward()



class FetchDynamicReachCollectEnv(FetchDynamicReachTestEnv, gym_utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        FetchReachV2Env.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.4, distance_threshold=0.04,
            initial_qpos=initial_qpos, reward_type=reward_type, early_stop=True, obstacle_speed=0.01)
        gym_utils.EzPickle.__init__(self)

