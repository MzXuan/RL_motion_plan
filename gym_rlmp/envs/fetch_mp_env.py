import os
from gym import utils
from gym.envs.robotics import fetch_env
import gym.envs.robotics as robotics
import numpy as np

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
        #get box position
        body_num = self.sim.model.body_name2id('target_box')
        box_center = self.sim.model.body_pos[body_num]
        goal = box_center.copy()
        goal[0] += np.random.uniform(-0.15, 0.15)        
        goal[1] += np.random.uniform(-0.15, 0.15)   
        goal[2] = 0.4     

        return goal.copy()

    def _reset_sim(self):
        print("self.initial state is:", self.initial_state)
        self.sim.set_state(self.initial_state)
        print("reset")

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

            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        # # Randomize start position of object.
        # if self.has_object:
        #     object_xpos = self.initial_gripper_xpos
        #     # object_xpos[2] += self.np_random.uniform(0, 0.45)
        #     while np.linalg.norm(object_xpos - self.initial_gripper_xpos) < 0.1:
        #         object_xpos = self.initial_gripper_xpos + self.np_random.uniform(-self.obj_range, self.obj_range, size=3)
        #     object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        #     assert object_qpos.shape == (7,)
            
        #     object_xpos[2] += self.np_random.uniform(0, 0.45)
        #     self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        # print("object xpos:", object_xpos)
        # print("initial gripper xpos:", self.initial_gripper_xpos)

        self.sim.forward()
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




def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)
