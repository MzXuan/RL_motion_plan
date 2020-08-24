import os
from gym import utils
from gym.envs.robotics import fetch_env
import gym.envs.robotics as robotics
import numpy as np

import copy

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place_xz.xml')


class FetchMotionPlanEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def _sample_goal(self):
        # #get box position
        # body_num = self.sim.model.body_name2id('target_box')
        # box_center = self.sim.model.body_pos[body_num]
        # goal = box_center.copy()
        # goal[0] += np.random.uniform(-0.15, 0.15)        
        # goal[1] += np.random.uniform(-0.15, 0.15)   
        # goal[2] = 0.4
        # return goal.copy()

        body_num = self.sim.model.body_name2id('table0')
        box_center = self.sim.model.body_pos[body_num]
        goal = box_center[:3].copy()

        flag = True
        try:
            self.object_xpos
        except:
            flag = False

        if flag:
            goal[:2] = self.object_xpos
        else: 
            # set goal a little bit higher than object
            goal[0] += np.random.uniform(-0.28, 0.28)
            goal[1] += np.random.uniform(-0.4, 0.4)
        
        goal[2] += 0.3
        return goal.copy()


    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)


    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

        if info["is_success"]:
            done = True
        return obs, reward, done, info


    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        self.last_dist = goal_distance(
            obs['achieved_goal'], self.goal)
        return obs

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # # set object below goal
        # if self.has_object:
        #     self._sample_goal()
        #     print("reset")
        #     object_xpos =  self.goal.copy()
        #     object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        #     assert object_qpos.shape == (7,)
        #     object_qpos[:2] = object_xpos[:2]
        #     self.sim.data.set_joint_qpos('object0:joint', object_qpos)


        #----- set object position on the table ----#
        if self.has_object:
            body_num = self.sim.model.body_name2id('table0')
            box_center = self.sim.model.body_pos[body_num]
            object_xpos = box_center[:2].copy()

            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)

            object_xpos[0] += np.random.uniform(-0.28, 0.28)
            object_xpos[1] += np.random.uniform(-0.4, 0.4)
            object_qpos[:2] = object_xpos
            self.object_xpos = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        self.last_dist = 0
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        robotics.utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        x = np.random.uniform(-0.5,0.5)
        y = np.random.uniform(-0.5,0.5)
        z = np.random.uniform(0.1,0.4)
        gripper_target = np.array([x, y, z + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

        # GoalEnv methods


    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robotics.utils.robot_get_obs(self.sim)
        # if self.has_object:
        #     object_pos = self.sim.data.get_site_xpos('object0')
        #     # rotations
        #     object_rot = robotics.rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
        #     # velocities
        #     object_velp = self.sim.data.get_site_xvelp('object0') * dt
        #     object_velr = self.sim.data.get_site_xvelr('object0') * dt
        #     # gripper state
        #     object_rel_pos = object_pos - grip_pos
        #     object_velp -= grip_velp
        # else:
            # object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)

        object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric


        achieved_goal = grip_pos.copy()
        
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        #todo: reduce observation state

        #todo: for spining up        

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

        # ----------------------------

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl = action[:3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        
        gripper_ctrl = 0.1
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        
        # Apply action to simulation.
        robotics.utils.ctrl_set_action(self.sim, action)
        robotics.utils.mocap_set_action(self.sim, action)


    def compute_reward(self, achieved_goal, goal, info):
        # r = self.sparse_reward(achieved_goal, goal)
        # return r

        if info["is_success"]:
            return 300.0
        else:
            # Compute distance between goal and the
            # r = self.sparse_reward(achieved_goal, goal)
            r = self.fast_app_reward(achieved_goal, goal)
            return r


    def sparse_reward(self, achieved_goal, goal):
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def fast_app_reward(self, achieved_goal, goal):
        current_dist = goal_distance(achieved_goal, goal)

        r = 20*(self.last_dist-current_dist)
        # print("achieved_goal {}, goal {}, current_dist{}, rewatd {}".format(achieved_goal, goal, current_dist, r))
        self.last_dist = copy.deepcopy(current_dist)
        return r



def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)
