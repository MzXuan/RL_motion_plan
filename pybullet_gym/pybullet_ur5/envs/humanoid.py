import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet
from pybullet_utils import bullet_client
from pybullet_envs import env_bases, scene_stadium, scene_abstract

import random

import numpy as np
import assets
import robot_bases
import math
from scenes.stadium import StadiumScene, PlaneScene
import pyquaternion


class Humanoid(robot_bases.MJCFBasedRobot):
    self_collision = True
    foot_list = ["right_foot", "left_foot"]  # "left_hand", "right_hand"

    def __init__(self, action_dim, obs_dim, random_yaw=False, random_lean=False):
        self.power = 0.41
        self.camera_x = 0
        super(Humanoid, self).__init__(
            'humanoid/humanoid_fixed.xml', 'torso', action_dim=action_dim, obs_dim=obs_dim)
        # 17 joints, 4 of them important for walking (hip, knee), others may as well be turned off, 17/4 = 4.25
        self.random_yaw = random_yaw
        self.random_lean = random_lean

        self.select_joints = ["left_shoulder1", "left_shoulder2", "left_elbow"]
        self.select_links = ["left_upper_arm", "left_lower_arm", "left_hand_true",
                             "right_upper_arm", "right_lower_arm", "right_hand_true"]

    def reset(self, bullet_client, base_position=[0, 0, -1.2], base_rotation=[0, 0, 0, 1], is_training =True):
        self._p = bullet_client
        # print("Created bullet_client with id=", self._p._client)
        if (self.doneLoading == 0):
            self.ordered_joints = []
            self.doneLoading = 1
            if self.self_collision:
                self.objects = self._p.loadMJCF(os.path.join(assets.getDataPath(), self.model_xml),
                                                flags=pybullet.URDF_USE_SELF_COLLISION |
                                                      pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p, self.objects)
            else:
                self.objects = self._p.loadMJCF(os.path.join(assets.getDataPath(), self.model_xml))
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p, self.objects)

        self.is_training = is_training
        self.robot_specific_reset(self._p, base_position, base_rotation, is_training =is_training)

        self.jdict['left_shoulder1'].max_velocity = 2.0
        self.jdict['left_shoulder2'].max_velocity = 1.8
        self.jdict['left_elbow'].max_velocity = 1.5

        s = self.calc_state(
        )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use

        return s

    def robot_specific_reset(self, bullet_client, base_position, base_rotation, is_training=True):
        # WalkerBase.robot_specific_reset(self, bullet_client)
        self._p = bullet_client
        for j in self.ordered_joints:
            j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)
        self.robot_body.reset_pose(base_position, base_rotation)
        self.feet = [self.parts[f] for f in self.foot_list]
        self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
        self.scene.actor_introduce(self)
        self.initial_z = None


        self.jdict["abdomen_z"].reset_position(0.0, 0)
        self.jdict["abdomen_y"].reset_position(0.0, 0)
        self.jdict["abdomen_x"].reset_position(0.0, 0)
        self.jdict["right_hip_x"].reset_position(0.0, 0)
        self.jdict["right_hip_z"].reset_position(0.0, 0)
        self.jdict["right_hip_y"].reset_position(0.0, 0)
        self.jdict["right_knee"].reset_position(0, 0)
        self.jdict["left_hip_x"].reset_position(0.0, 0)
        self.jdict["left_hip_z"].reset_position(0.0, 0)
        self.jdict["left_hip_y"].reset_position(0.0, 0)
        self.jdict["left_knee"].reset_position(0.0, 0)
        self.jdict["right_shoulder1"].reset_position(0, 0)
        self.jdict["right_shoulder2"].reset_position(0, 0)
        self.jdict["right_elbow"].reset_position(0, 0)

        if is_training is True:
            success = None
            while success is None:
                try:
                    position = [random.uniform(-1.5, 0.4), random.uniform(-0.5, 1.5), random.uniform(0.7, 2)]
                    jointPoses = self._p.calculateInverseKinematics(self.robot_body.bodies[0],\
                                                                    self.parts['left_hand_true'].bodyPartIndex, position)
                    success = 1
                except:
                    pass

            self.jdict["left_shoulder1"].reset_position(jointPoses[14],0)
            self.jdict["left_shoulder2"].reset_position(jointPoses[15],0)
            self.jdict["left_elbow"].reset_position(jointPoses[16],0)

        else:
            success = None
            while success is None:
                try:
                    position = [random.uniform(0.4, 1.5), random.uniform(-0.5, 1.5), random.uniform(0.7, 2)]
                    jointPoses = self._p.calculateInverseKinematics(self.robot_body.bodies[0],\
                                                                    self.parts['left_hand_true'].bodyPartIndex, position)
                    success = 1
                except:
                    pass
            # print("position {}, joint poses {}".format(position, jointPoses))

            self.jdict["left_shoulder1"].reset_position(jointPoses[14], 0)
            self.jdict["left_shoulder2"].reset_position(jointPoses[15], 0)
            self.jdict["left_elbow"].reset_position(jointPoses[16], 0)



        self.motor_names = ["right_shoulder1", "right_shoulder2", "right_elbow"]
        self.motor_power = [75, 75, 75]
        self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
        self.motor_power += [75, 75, 75]
        self.motors = [self.jdict[n] for n in self.motor_names]


    def calc_state(self):
        # joint state shape:[3,2]
        joint_states = np.asarray([self.jdict[i].get_state() for i in self.select_joints if i in self.jdict])
        # joint 3d position shape:[3,3]
        link_position = np.asarray([self.parts[j].get_position() for j in self.select_links if j in self.parts])

        obs = np.concatenate((joint_states.flatten(), link_position.flatten()))
        return obs


    def apply_action(self, a):
        #control arm
        assert (np.isfinite(a).all())
        self.jdict["right_shoulder1"].set_position(0)
        self.jdict["right_shoulder2"].set_position(0)
        self.jdict["right_elbow"].set_position(0)
        # scale

        for i in range((self.action_space.shape)[0]):
            scale = self.jdict[self.select_joints[i]].max_velocity
            action = a[i] * (scale) / 2
            # action = a[i] * (high - low) / 2 + (high + low) / 2
            self.jdict[self.select_joints[i]].set_velocity(action)


    def alive_bonus(self, z, pitch):
        return +2 if z > 0.78 else -1  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0
        return 0





class SelfMoveHumanoid(Humanoid):
    def __init__(self,action_dim=3, obs_dim=12, is_training=True, move_base=False, noise=False):

        super(SelfMoveHumanoid, self).__init__(action_dim=action_dim, obs_dim=obs_dim)
        self.select_joints = ["left_shoulder1", "left_shoulder2", "left_elbow"]
        # self.select_links = ["left_upper_arm", "left_lower_arm", "left_hand_true"]
        # self.select_links = ["left_lower_arm", "left_hand_true", "right_lower_arm", "right_hand_true"]
        self.select_links = ["left_lower_arm", "left_hand_true"]
        self.is_training = is_training
        self.move_base = move_base
        self.noise = noise
        self.initial_hand_pose = np.zeros(3)


    def reset(self, bullet_client, base_position=[0, 0, -1.4], base_rotation=[0, 0, 0, 1]):
        self._p = bullet_client
        # print("Created bullet_client with id=", self._p._client)
        if (self.doneLoading == 0):
            self.ordered_joints = []
            self.doneLoading = 1
            if self.self_collision:
                self.objects = self._p.loadMJCF(os.path.join(assets.getDataPath(), self.model_xml),
                                                flags=pybullet.URDF_USE_SELF_COLLISION |
                                                      pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p, self.objects)
            else:
                self.objects = self._p.loadMJCF(os.path.join(assets.getDataPath(), self.model_xml))
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p, self.objects)


        self.robot_specific_reset(self._p, base_position, base_rotation, is_training =self.is_training)

        self.jdict['left_shoulder1'].max_velocity = 2.0
        self.jdict['left_shoulder2'].max_velocity = 1.8
        self.jdict['left_elbow'].max_velocity = 1.5

        self.last_link_position = np.asarray([self.parts[j].get_position() for j in self.select_links if j in self.parts])

        s = self.calc_state(
        )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use

        return s

    def robot_specific_reset(self, bullet_client, base_position, base_rotation, is_training=True):
        # WalkerBase.robot_specific_reset(self, bullet_client)
        self._p = bullet_client
        for j in self.ordered_joints:
            j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)
        self.robot_body.reset_pose(base_position, base_rotation)
        self.feet = [self.parts[f] for f in self.foot_list]
        self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
        self.scene.actor_introduce(self)
        self.initial_z = None

        # self.start_center = np.asarray([-0.2, -1.0, 0])



        self.jdict["abdomen_z"].reset_position(0.0, 0)
        self.jdict["abdomen_y"].reset_position(0.0, 0)
        self.jdict["abdomen_x"].reset_position(0.0, 0)
        self.jdict["right_hip_x"].reset_position(0.0, 0)
        self.jdict["right_hip_z"].reset_position(0.0, 0)
        self.jdict["right_hip_y"].reset_position(0.0, 0)
        self.jdict["right_knee"].reset_position(0, 0)
        self.jdict["left_hip_x"].reset_position(0.0, 0)
        self.jdict["left_hip_z"].reset_position(0.0, 0)
        self.jdict["left_hip_y"].reset_position(0.0, 0)
        self.jdict["left_knee"].reset_position(0.0, 0)
        self.jdict["right_shoulder1"].reset_position(0, 0)
        self.jdict["right_shoulder2"].reset_position(0, 0)
        self.jdict["right_elbow"].reset_position(0, 0)


        self.random_initial_pose()

        self.time = np.random.uniform(0,50)
        # self.time = 0
        self.is_training = is_training

        self.motor_names = ["right_shoulder1", "right_shoulder2", "right_elbow"]
        self.motor_power = [75, 75, 75]
        self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
        self.motor_power += [75, 75, 75]
        self.motors = [self.jdict[n] for n in self.motor_names]

        # self.initial_hand_pose = {"left_hand_true": self.parts["left_hand_true"].get_position(),
        #                           "right_hand_true": self.parts["right_hand_true"].get_position()}

        self.initial_hand_pose = {"left_hand_true": self.parts["left_hand_true"].get_position()}
        self.last_hand_pose = self.initial_hand_pose.copy()


    def random_initial_pose(self):
        if np.random.choice([0,1]) == 0:
            self.jdict["left_shoulder1"].reset_position(np.random.uniform(-0.5,0), 0)
            self.jdict["left_shoulder2"].reset_position(np.random.uniform(-0.5,0), 0)
            self.jdict["left_elbow"].reset_position(np.random.uniform(-1.5,-1.0), 0)
        else:
            self.jdict["left_shoulder1"].reset_position(1.0, 0)
            self.jdict["left_shoulder2"].reset_position(0, 0)
            self.jdict["left_elbow"].reset_position(np.random.uniform(-1.5,-1.0), 0)


    def reset_base(self):

        # self.jdict["left_shoulder1"].reset_position(0, 0)
        # self.jdict["left_shoulder2"].reset_position(0, 0)
        # self.jdict["left_elbow"].reset_position(-1.5, 0)
        # self.initial_hand_pose = self.parts["left_hand_true"].get_position()

        self.human_goal =  self.robot_goal+self.random_n(max=[0.15,0.15, 0.3], min=[0,0,0.1])

        theta = np.arctan2(self.human_goal[1], self.human_goal[0])
        r = np.linalg.norm(self.human_goal) + np.random.uniform(0.2, 0.4)
        pi = 3.1415926

        xh = r * np.cos(theta)
        yh = r * np.sin(theta)
        qh = pyquaternion.Quaternion(axis=[0, 0, 1], angle=theta + pi)  # w, x, y, z
        self.reset(self._p, base_position=[xh, yh, -1.1],
                                      base_rotation=[qh.q[1], qh.q[2], qh.q[3], qh.q[0]])



    def apply_action(self, a):
        assert (np.isfinite(a).all())
        # position = [a[0], a[1], a[2]]
        # position = self.human_static_motion()
        left_position = self.human_motion_generator("left_hand_true")
        # right_position = self.human_motion_generator("right_hand_true")

        if self.noise is True:
            self.time += np.random.uniform(1,5)*0.5
            # self.time += 0
            # self.time+=np.random.uniform(4,10)
        else:
            self.time += 0.5

        self.jdict["right_shoulder1"].reset_position(-0.5, 0)
        self.jdict["right_shoulder2"].reset_position(-0.9, 0)
        self.jdict["right_elbow"].reset_position(0, 0)

        jointPoses_l = self._p.calculateInverseKinematics(self.robot_body.bodies[0], \
                                                        self.parts['left_hand_true'].bodyPartIndex,
                                                        left_position)

        # jointPoses_r = self._p.calculateInverseKinematics(self.robot_body.bodies[0], \
        #                                                 self.parts['right_hand_true'].bodyPartIndex,
        #                                                 right_position)
        self.jdict["left_shoulder1"].reset_position(jointPoses_l[14], 0)
        self.jdict["left_shoulder2"].reset_position(jointPoses_l[15], 0)
        self.jdict["left_elbow"].reset_position(jointPoses_l[16], 0)

        # self.jdict["right_shoulder1"].reset_position(jointPoses_r[11], 0)
        # self.jdict["right_shoulder2"].reset_position(jointPoses_r[12], 0)
        # self.jdict["right_elbow"].reset_position(jointPoses_r[13], 0)

        self.jdict["abdomen_z"].reset_position(0,0)
        self.jdict["abdomen_y"].reset_position(0, 0)
        self.jdict["abdomen_x"].reset_position(0, 0)

        self.jdict['right_hip_x'].reset_position(0,0)
        self.jdict['right_hip_y'].reset_position(0, 0)
        self.jdict['right_hip_z'].reset_position(0, 0)

        self.jdict['left_hip_x'].reset_position(0, 0)
        self.jdict['left_hip_y'].reset_position(0, 0)
        self.jdict['left_hip_z'].reset_position(0, 0)


        self.jdict["left_knee"].reset_position(0, 0)
        self.jdict["right_knee"].reset_position(0, 0)



        if self.move_base is True:
            id = self.objects[0]
            base_pose = self._p.getBasePositionAndOrientation(id)
            position = np.asarray(base_pose[0])
            position[0] +=np.random.uniform(-0.05,0.05)
            self._p.resetBasePositionAndOrientation(id, position, base_pose[1])
            # self.robot_body.reset_pose(base_position, base_rotation)

    def calc_state(self):
        link_position = np.asarray([self.parts[j].get_position() for j in self.select_links if j in self.parts])
        velocity = link_position-self.last_link_position
        obs = np.concatenate((np.asarray(link_position.flatten()), np.asarray(velocity.flatten())))
        self.last_link_position = link_position
        return obs


    def human_static_motion(self):
        h_action = self.left_hand_end
        return h_action


    def set_goal_position(self, goal_positon, first_set = True):
        self.robot_goal = self.human_goal = goal_positon.copy()
        if first_set:
            self.approaching = True
            self.stop = 0
            self.true_count = 0


    def human_motion_generator(self, hand_name):
        dt = 0.05

        current_hand_pos = self.parts[hand_name].get_position()


        if (np.linalg.norm(current_hand_pos-self.human_goal) > \
                np.linalg.norm(self.last_hand_pose[hand_name] - self.human_goal)\
                or self.true_count>20) and self.approaching is True:
            self.approaching = False
            self.stop = 5
            self.true_count = 0

        elif np.linalg.norm(current_hand_pos-self.initial_hand_pose[hand_name]) > \
                np.linalg.norm(self.last_hand_pose[hand_name] - self.initial_hand_pose[hand_name])  \
                and self.approaching is False and self.stop == 0:
            self.approaching = True
            # self.reset_base()


            self.human_goal = self.robot_goal + self.random_n(max=[0.15, 0.15, 0.3], min=[0, 0, 0.1])


        # print("humanoid approaching", self.approaching)

        noise_vel = self.random_n(max = [0.01, 0.01, 0.01])
        # if np.random.choice([0,1,2,3,4,5,6,7,8,9,10]) == 0:
        #     self.velocity = np.zeros(3)+noise_vel

        if self.stop >  0:
            self.velocity = np.zeros(3) + noise_vel
            self.stop = self.stop-1


        elif self.approaching is True:
            self.true_count +=1
            self.velocity = 0.4*(self.human_goal - current_hand_pos)/np.linalg.norm((self.human_goal - current_hand_pos))\
                            *np.random.uniform(0.9,1.1)+noise_vel
        else:
            self.velocity = -0.4*(current_hand_pos - self.initial_hand_pose[hand_name])/np.linalg.norm((current_hand_pos - self.initial_hand_pose[hand_name]))\
                            *np.random.uniform(0.9,1.1)+noise_vel

        self.last_hand_pose[hand_name] = current_hand_pos


        hand_position = self.parts[hand_name].get_position()+self.velocity*dt
        return hand_position



    # def human_motion_generator(self, hand_name):
    #     dt = 0.05
    #
    #     current_hand_pos = self.parts["left_hand_true"].get_position()
    #
    #     if np.linalg.norm(current_hand_pos-self.human_goal) > \
    #             np.linalg.norm(self.last_hand_pose - self.human_goal) and self.approaching is True:
    #         self.approaching = False
    #     elif np.linalg.norm(current_hand_pos-self.initial_hand_pose) > \
    #             np.linalg.norm(self.last_hand_pose - self.initial_hand_pose)  and self.approaching is False:
    #         self.approaching = True
    #         self.reset_base()
    #
    #
    #     # print("humanoid approaching", self.approaching)
    #     if self.approaching is True:
    #         self.velocity = (self.human_goal+self.random_n([0.1,0.1,0.1]) - self.initial_hand_pose)
    #     else:
    #         self.velocity = -(self.human_goal - self.initial_hand_pose)
    #
    #     self.last_hand_pose = current_hand_pos
    #
    #
    #     hand_position = self.parts["left_hand_true"].get_position()+self.velocity*dt
    #     return hand_position

    def random_n(self, max, min=None):
        result = []
        if min is None:
            for m in max:
                result.append(np.random.uniform(-m, m))
        else:
            for s,e in zip(min,max):
                result.append(np.random.uniform(s,e))
        return np.asarray(result)



