import os
from gym import utils
from gym.envs.robotics import fetch_env
import gym.envs.robotics as robotics
import numpy as np
import math

import mujoco_py

import copy

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place_xz.xml')


class FetchMotionPlanEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='point'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.41, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.41, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.41, 1., 0., 0., 0.],
        }

        self.object_name_list = ['object0:joint', 'object1:joint', 'object2:joint']
        self.early_stop = True

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def _sample_goal(self):
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
        self.time_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'is_collision': self._contact_dection()
        }
        reward = self.compute_reward(obs['observation'], obs['achieved_goal'], self.goal, info)

        if self.early_stop:
            if info["is_success"] or info["is_collision"]:
                done = True
        return obs, reward, done, info


    def _contact_dection(self):
        #----------------------------------
        # if there is collision: return true
        # if there is no collision: return false
        #----------------------------------

        # print('number of contacts', self.sim.data.ncon)
        # for i in range(self.sim.data.ncon):
        #     # Note that the contact array has more than `ncon` entries,
        #     # so be careful to only read the valid entries.
        #     contact = self.sim.data.contact[i]
        #     print('contact', i)
        #     print('dist', contact.dist)
        #     print('geom1', contact.geom1, self.sim.model.geom_id2name(contact.geom1))
        #     print('geom2', contact.geom2, self.sim.model.geom_id2name(contact.geom2))
        #     print("exclude", contact.exclude)
        #     # There's more stuff in the data structure
        #     # See the mujoco documentation for more info!
        #     geom2_body = self.sim.model.geom_bodyid[self.sim.data.contact[i].geom2]
        #     print(' Contact force on geom2 body', self.sim.data.cfrc_ext[geom2_body])
        #     print('norm', np.sqrt(np.sum(np.square(self.sim.data.cfrc_ext[geom2_body]))))
        #     # Use internal functions to read out mj_contactForce
        #     c_array = np.zeros(6, dtype=np.float64)
        #     print('c_array', c_array)
        #     mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, c_array)
        #     print('c_array', c_array)


        contact_count = 0
        for i in range(self.sim.data.ncon):
            # Note that the contact array has more than `ncon` entries,
            # so be careful to only read the valid entries.
            contact = self.sim.data.contact[i]
            if self.sim.model.geom_id2name(contact.geom1) is None or \
                self.sim.model.geom_id2name(contact.geom2) is None:
                pass
            else:
                contact_count +=1

        if contact_count > 0:
            return True
        else:
            return False


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


    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        #reset mocap to some place
        body_num = self.sim.model.body_name2id('target_box')
        box_center = self.sim.model.body_pos[body_num]
        grip_pos = box_center.copy()
        grip_pos[0] += np.random.uniform(-0.15, 0.15)
        grip_pos[1] += np.random.uniform(-0.15, 0.15)
        grip_pos[2] += np.random.uniform(0.3, 0.5)

        gripper_rotation = np.array([1., 1., 0., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', grip_pos)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        #--- random put goals ---#
        #todo: control goal distance
        xpos_list = []
        qpos_list = []

        for name in self.object_name_list:
            xpos, qpos = self.put_object(name)
            self.sim.data.set_joint_qpos(name, qpos)
            xpos_list.append(xpos)
            qpos_list.append(qpos)

        self.object_xpos = xpos_list[0]
        # print("xposlist: {} and qpos list {}".format(xpos_list, qpos_list))

        self.sim.forward()
        self.last_dist = 0
        return True

    def put_object(self, name):
        body_num = self.sim.model.body_name2id('table0')
        table_center = self.sim.model.body_pos[body_num]
        object_xpos = table_center[:2].copy()

        object_qpos = self.sim.data.get_joint_qpos(name)
        assert object_qpos.shape == (7,)

        object_xpos[0] += np.random.uniform(-0.28, 0.28)
        object_xpos[1] += np.random.uniform(-0.4, 0.4)
        object_qpos[:2] = object_xpos
        object_qpos[2] = 0.4
        return object_xpos, object_qpos


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
        # if self.has_object:
        #     self.height_offset = self.sim.data.get_site_xpos('object0')[2]

        # GoalEnv methods


    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robotics.utils.robot_get_obs(self.sim)
        
        # add all objects into observations
        xpos_list  = []
        for name in self.object_name_list:
            xpos = self.sim.data.get_joint_qpos(name)[:3]
            xpos_list.append(xpos)
        xpos_list = np.asarray(xpos_list)
        self.alternative_goals = xpos_list

        achieved_goal = grip_pos.copy()
        

        # obs = np.concatenate([
        #     grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
        #     object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        # ])

        obs = np.concatenate([grip_pos, grip_velp, xpos_list.ravel()])

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


    def compute_reward(self, obs, achieved_goal, goal, info):
        if self.reward_type == 'sparse':
            r = self.sparse_reward(achieved_goal, goal)
        elif self.reward_type == 'fast_app':
            r = self.fast_app_reward(achieved_goal, goal, info)

        elif self.reward_type == 'point':
            r = self.point_reward(obs, achieved_goal, goal, info)

        else:
            print("reward type {} not recognize".format(self.reward_type))
            raise NotImplementedError
        return r


    def sparse_reward(self, achieved_goal, goal):
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def fast_app_reward(self, achieved_goal, goal, info):
        if info["is_success"]:
            return 300.0
        elif info["is_collision"]:
            return -30
        else:
            current_dist = goal_distance(achieved_goal, goal)
            r = 20*(self.last_dist-current_dist)
            self.last_dist = copy.deepcopy(current_dist)
            return r


    def point_reward(self, obs, achieved_goal, goal, info):
        if info["is_success"]:
            return 300.0
        elif info["is_collision"]:
            return -30
        else:
            t = self.time_step
            dis_list = []
            alternative_goals = obs[-6:]

            d0 = np.linalg.norm(achieved_goal - goal)
            for g in alternative_goals:
                if np.linalg.norm(g - goal) < 1e-7:
                    pass
                else:
                    dis_list.append(np.linalg.norm(achieved_goal - g))
            rew = self.reward_dist(d0, dis_list, t)
            return rew

    def reward_dist(self, d0, dis, t):
        rew = []
        for d in dis:
            theta = 1 if d0 < d else -1
            rew.append(theta*math.log(abs(d0-d)/abs(d0+1)+1)) #  why log???
            # rew.append(theta * abs(d0 - d) / abs(d0 + 1))
            # rew.append(theta*math.log(abs(d0-d)/abs(d0+d)+1))
        # print("distance is {} and reward list is: {} ".format(dist, rew))
        min_rew = np.asarray(rew).min()
        time_scale = math.exp(-t / 30)
        return (time_scale*min_rew)

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)
