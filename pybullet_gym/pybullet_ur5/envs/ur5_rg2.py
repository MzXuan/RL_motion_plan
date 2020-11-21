import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet
from pybullet_envs import env_bases, scene_stadium, scene_abstract

import random
import numpy as np
import robot_bases
import assets
from scenes.stadium import StadiumScene

import utils


class UR5RG2Robot(robot_bases.URDFBasedRobot):
    TARG_LIMIT = 0.27

    def __init__(self, action_dim=6, obs_dim=19):
        super(UR5RG2Robot, self).__init__(
            'ur5_rg2/ur5.urdf', "ur5_robot", action_dim=action_dim, obs_dim=obs_dim, fixed_base=1, self_collision=True)

        self.select_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", \
                              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        # self.select_links = ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_3_link", "wrist_2_link"]
        self.select_links = ["upper_arm_link", "wrist_3_link"]
        # self.ee_link = "rg2_eef_link"
        self.ee_link = "ee_link"


    #


    def reset(self, bullet_client, base_position=[0, 0, 0], base_rotation=[0, 0, 0, 1], eef_pose=None):
        self._p = bullet_client
        self.ordered_joints = []

        # print(os.path.join(os.path.dirname(__file__), "data", self.model_urdf))

        if self.jdict is None:
            if self.self_collision:
                self.robot_id = self._p.loadURDF(os.path.join(assets.getDataPath(), self.model_urdf),
                                     basePosition=self.basePosition,
                                     baseOrientation=self.baseOrientation,
                                     useFixedBase=self.fixed_base,
                                     flags=pybullet.URDF_USE_SELF_COLLISION)
            else:
                self.robot_id = self._p.loadURDF(os.path.join(assets.getDataPath(), self.model_urdf),
                                     basePosition=self.basePosition,
                                     baseOrientation=self.baseOrientation,
                                     useFixedBase=self.fixed_base)

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
            self._p, self.robot_id)

        self.robot_specific_reset(self._p, base_position, base_rotation, eef_pose)
        self.jdict['shoulder_pan_joint'].max_velocity = 3.15
        self.jdict['shoulder_lift_joint'].max_velocity = 3.15
        self.jdict['elbow_joint'].max_velocity = 3.15
        self.jdict['wrist_1_joint'].max_velocity = 3.2
        self.jdict['wrist_2_joint'].max_velocity = 3.2
        self.jdict['wrist_3_joint'].max_velocity = 3.2

        self.last_joint = [self.jdict[joint_id].get_position() for joint_id in self.select_joints]

        s = self.calc_state(
        )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
        self.potential = self.calc_potential()
        return s

    def robot_specific_reset(self, bullet_client, base_position, base_rotation, eef_position=None):
        for n in self.select_joints:
            self.jdict[n].reset_current_position(
                self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)

        self.robot_body.reset_pose(base_position, base_rotation)

        success = None

        # ----update initialization position from end-effector ----#
        if eef_position is not None:
            try:
                position = eef_position
                jointPoses = self._p.calculateInverseKinematics(self.robot_body.bodies[0],
                                                                self.parts[self.ee_link].bodyPartIndex,
                                                                targetPosition=position,
                                                                targetOrientation=[0, 0.707, 0, 0.707])
                success = 1
            except:
                print("can not find a valid solution for given pose")
                pass

        while success is None:
            try:
                position = [random.uniform(-0.5, 0.5), random.uniform(0.8, 1.5), random.uniform(1.4,
                                                                                                1.8)]
                jointPoses = self._p.calculateInverseKinematics(self.robot_body.bodies[0],
                                                                self.parts[self.ee_link].bodyPartIndex, \
                                                                targetPosition=position)
                success = 1
            except:
                pass


        #
        self.jdict['shoulder_pan_joint'].reset_position(jointPoses[0], 0)
        self.jdict['shoulder_lift_joint'].reset_position(jointPoses[1], 0)
        self.jdict['elbow_joint'].reset_position(jointPoses[2], 0)
        self.jdict['wrist_1_joint'].reset_position(jointPoses[3], 0)
        self.jdict['wrist_2_joint'].reset_position(jointPoses[4], 0)
        self.jdict['wrist_3_joint'].reset_position(jointPoses[5], 0)


        # # ---- change initialization position from joint space ---#
        # self.jdict['shoulder_pan_joint'].reset_position(np.random.uniform(-0.78, 3.14), 0)
        # # self.jdict['shoulder_lift_joint'].reset_position(np.random.uniform(-1.8, 0), 0)
        # # self.jdict['shoulder_pan_joint'].reset_position(np.random.uniform(1.8, 3.14), 0)
        # self.jdict['shoulder_lift_joint'].reset_position(np.random.uniform(-1.8, 0), 0)
        # self.jdict['elbow_joint'].reset_position(np.random.uniform(0, 1.8), 0)
        # self.jdict['wrist_1_joint'].reset_position(np.random.uniform(-1.8, 1.8), 0)
        # self.jdict['wrist_2_joint'].reset_position(np.random.uniform(-1.8, 1.8), 0)
        # self.jdict['wrist_3_joint'].reset_position(0, 0)

        # self.last_position = list(
        #     self._p.getLinkState(self.robot_body.bodies[0], self.parts[self.ee_link].bodyPartIndex)[0])
        self.last_joint = \
            list(self.jdict[self.select_joints[i]].get_position() for i in range((self.action_space.shape)[0]))


    # def apply_action(self, a):
    #     #position
    #     assert (np.isfinite(a).all())
    #     max_step = 0.06
    #     # scale
    #     for i in range((self.action_space.shape)[0]):
    #         scale = self.jdict[self.select_joints[i]].upperLimit
    #         action = a[i] * (scale)
    #         if action - self.last_joint[i] > max_step:
    #             action = self.last_joint[i]+max_step
    #         elif action - self.last_joint[i] < -max_step:
    #             action = self.last_joint[i] - max_step
    #         self.jdict[self.select_joints[i]].set_position(action)
    #         self.last_joint[i] = action

    #
    # def apply_action(self, a):
    #     #velocity
    #     assert (np.isfinite(a).all())
    #     # scale
    #     for i in range((self.action_space.shape)[0]):
    #         scale = self.jdict[self.select_joints[i]].max_velocity
    #         action = a[i] * (scale)/3
    #         # action = a[i] * (high - low) / 2 + (high + low) / 2
    #         self.jdict[self.select_joints[i]].set_velocity(action)

    def apply_action(self, a):
        # manual velocity
        assert (np.isfinite(a).all())
        # scale
        for i in range((self.action_space.shape)[0]):
            if abs(a[i]) > self.jdict[self.select_joints[i]].max_velocity:
                if a[i] > 0:
                    a[i] = self.jdict[self.select_joints[i]].max_velocity
                else:
                    a[i] = -self.jdict[self.select_joints[i]].max_velocity

            delta_t = 0.05
            action = a[i] * delta_t
            self.jdict[self.select_joints[i]].set_position(self.last_joint[i]+action)
        self.last_joint = \
            list(self.jdict[self.select_joints[i]].get_position() for i in range((self.action_space.shape)[0]))


    def calc_state(self):
        # link_position = np.asarray([self.parts[j].get_position() for j in self.select_links if j in self.parts])
        # joint_position = np.asarray(
        #     [self.jdict[i].get_position() for i in self.select_joints if i in self.jdict])  # position

        joint_position = np.asarray(
            [self.jdict[i].get_position()/self.jdict[i].upperLimit*2 for i in self.select_joints if i in self.jdict])  # position
        joint_velocity = np.asarray(
            [self.jdict[i].get_velocity() for i in self.select_joints if i in self.jdict])  # velocity

        eef_pose = self.parts[self.ee_link].get_pose()  # position [0:3], orientation [3:7]
        # obs = np.concatenate((eef_pose, link_position.flatten()))
        obs = np.concatenate((eef_pose, joint_position.flatten(), joint_velocity.flatten()))
        return obs

    def calc_potential(self):
        return 0
        # return -100 * np.linalg.norm(self.to_target_vec)

    def normalize_joint(self, joint):
        assert (len(joint) == 6)
        normal_j = []
        for joint_name, joint_value in zip(self.select_joints, joint):
            n_j = joint_value / self.jdict[joint_name].upperLimit*2
            normal_j.append(n_j)
        return normal_j

