import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet
from pybullet_envs import env_bases, scene_stadium, scene_abstract

import numpy as np
import robot_bases
import assets
from scenes.stadium import StadiumScene
from ur5 import UR5Robot
import random

from ur5_control import UR5Control

import ikfast_ur5


def normalize_conf(start, end):
    circle = 6.2831854
    normal_e = []
    for s, e in zip(start, end):
        if e > s:
            test = e - circle
        else:
            test = e + circle
        normal_e.append(test if np.linalg.norm(test - s) < np.linalg.norm(e - s) else e)
    return normal_e


class UR5RealRobot(robot_bases.URDFBasedRobot):
    TARG_LIMIT = 0.27

    def __init__(self, action_dim=3, obs_dim=19, fixed_base = 1, self_collision=True):
        # self.select_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",\
        # 					"wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        # self.select_links = ["shoulder_link", "upper_arm_link","forearm_link","ee_link"]

        self.select_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", \
                              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.select_links = ["upper_arm_link", "wrist_3_link"]

        self.last_position = [0, 0, 0]
        self.ee_link = "ee_link"
        self.lower_limits = [-6, -6, 0, -6, -6, -6]
        self.upper_limits = [6, 0, 6, 0, 6, 6]

        self.orientation = None
        self.ur5_rob_control = UR5Control(ip='192.168.0.3')


        super(UR5RealRobot, self).__init__(
            'ur_description/ur5.urdf', "ur5_robot", action_dim=action_dim, obs_dim=obs_dim, fixed_base=1,
            self_collision=True)


    def select_ik_solution(self, solutions):
        feasible_solution = []

        contact_free_solution = []
        for jointPoses in solutions:
            self.jdict['shoulder_pan_joint'].reset_position(jointPoses[0], 0)
            self.jdict['shoulder_lift_joint'].reset_position(jointPoses[1], 0)
            self.jdict['elbow_joint'].reset_position(jointPoses[2], 0)
            self.jdict['wrist_1_joint'].reset_position(jointPoses[3], 0)
            self.jdict['wrist_2_joint'].reset_position(jointPoses[4], 0)
            self.jdict['wrist_3_joint'].reset_position(jointPoses[5], 0)
            self._p.stepSimulation()

            if self._contact_detection() is False:
                contact_free_solution.append(jointPoses)

        if contact_free_solution == []:
            print("can not find contact free solution")
        else:
            for jp in contact_free_solution:
                if self._is_in_range(jp):
                    feasible_solution.append(jp)
        if feasible_solution == []:
            print("can not find in range solution")

        return feasible_solution

    def _is_in_range(self, jointPoses):
        for i in range(0, 6):
            if jointPoses[i] < self.upper_limits[i] and jointPoses[i] > self.lower_limits[i]:
                continue
            else:
                return False
        return True

    def _contact_detection(self):
        # collision detection
        collisions = self._p.getContactPoints()
        collision_bodies = []
        for c in collisions:
            bodyinfo1 = self._p.getBodyInfo(c[1])
            bodyinfo2 = self._p.getBodyInfo(c[2])

            if c[3] == 3 and c[4] == 5:
                continue
            if c[3] == 0 or c[4] == 0:
                continue

            # print("bodyinfo1: ", bodyinfo1, "bodyinfo2: ", bodyinfo2)
            # print("collisions", collisions)
            # print("linkid 1 ", c[3])
            # print("linkid 2", c[4])
            #
            # print("robot parts", self.agents[0].parts)
            # p = self._p.getLinkState(c[1], c[3])[0]
            collision_bodies.append(bodyinfo1[1].decode("utf-8"))
            collision_bodies.append(bodyinfo2[1].decode("utf-8"))

        if len(collision_bodies) != 0:
            if "ur5" in collision_bodies:  # robot collision
                return True
            else:
                return False

        return False

    def robot_specific_reset(self, bullet_client):

        real_jointPoses = self.ur5_rob_control.get_joint_state()[0]

        self.jdict['shoulder_pan_joint'].reset_position(real_jointPoses[0], 0)
        self.jdict['shoulder_lift_joint'].reset_position(real_jointPoses[1], 0)
        self.jdict['elbow_joint'].reset_position(real_jointPoses[2], 0)
        self.jdict['wrist_1_joint'].reset_position(real_jointPoses[3], 0)
        self.jdict['wrist_2_joint'].reset_position(real_jointPoses[4], 0)
        self.jdict['wrist_3_joint'].reset_position(real_jointPoses[5], 0)


    def reset(self, bullet_client, client_id, base_position=[0, 0, 0], base_rotation=[0, 0, 0, 1]):
        self._p = bullet_client
        self.client_id = client_id
        self.ordered_joints = []

        # print(os.path.join(os.path.dirname(__file__), "data", self.model_urdf))

        if self.jdict is None:
            if self.self_collision:
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p,
                    self._p.loadURDF(os.path.join(assets.getDataPath(), self.model_urdf),
                                     basePosition=self.basePosition,
                                     baseOrientation=self.baseOrientation,
                                     useFixedBase=self.fixed_base,
                                     flags=pybullet.URDF_USE_SELF_COLLISION))

            else:
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p,
                    self._p.loadURDF(os.path.join(assets.getDataPath(), self.model_urdf),
                                     basePosition=self.basePosition,
                                     baseOrientation=self.baseOrientation,
                                     useFixedBase=self.fixed_base))

        self.robot_body.reset_pose(base_position, base_rotation)
        self.robot_specific_reset(self._p)

        self.jdict['shoulder_pan_joint'].max_velocity = 3.15
        self.jdict['shoulder_lift_joint'].max_velocity = 3.15
        self.jdict['elbow_joint'].max_velocity = 3.15
        self.jdict['wrist_1_joint'].max_velocity = 3.2
        self.jdict['wrist_2_joint'].max_velocity = 3.2
        self.jdict['wrist_3_joint'].max_velocity = 3.2

        s = self.calc_state(
        )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
        return s

    def apply_action(self, a):
        # todo: send to real robot control script
        assert (np.isfinite(a).all())
        # scale

        max_eef_velocity = 0.1
        scale = max_eef_velocity

        # current_position,_ = self.ur5_rob_control.get_tool_state()

        current_position = self.ur5_rob_control.get_tool_state_2()
        current_joint = self.ur5_rob_control.get_joint_state()[0]
        next_position = current_position.copy()
        for i in range((self.action_space.shape)[0]):
            next_position[i] += a[i] * scale
        # print("current_tool_position",current_position)
        # print("next_tool_position",next_position)
        # print("action ", a)


        # next_joint = np.asarray(self._p.calculateInverseKinematics(self.robot_body.bodies[0],
        #                                                 self.parts['ee_link'].bodyPartIndex,
        #                                                 next_position))

        next_joint = np.asarray(self._p.calculateInverseKinematics(self.robot_body.bodies[0], self.parts['ee_link'].bodyPartIndex,
                                                        next_position, self.orientation
                                                        ))

        joint_position, joint_velocity = self.ur5_rob_control.get_joint_state()

        joint_v = (next_joint - current_joint)


        #todo: set joint v
        # print("robot joint velocity is: ", joint_v)


        # self.ur5_rob_control.set_joint_position(next_joint, wait=False)
        # self.ur5_rob_control.set_joint_velocity(joint_v)
        delta_tool_position = next_position - current_position
        delta_tool_orientation = [0,0,0]

        print("ur5 real proposed velocity: ", np.asarray(delta_tool_position))

        self.ur5_rob_control.set_tool_velocity(np.concatenate([np.asarray(delta_tool_position), np.asarray(delta_tool_orientation)]))



    def calc_state(self):
        # todo: get from real robot
        joint_position, joint_velocity = self.ur5_rob_control.get_joint_state()
        # eef_pos, eef_vel = self.ur5_rob_control.get_tool_state()

        eef_pos = self.ur5_rob_control.get_tool_state_2()
        # joint_velocity = np.asarray(joint_velocity)*50
        # obs = np.concatenate((eef_pose, np.asarray(joint_position), np.asarray(joint_velocity)))
        # print("eef vel: ", eef_vel)

        # obs = np.concatenate((eef_pos, np.asarray(joint_position)))
        # obs = np.concatenate((eef_pos,eef_vel*1, np.asarray(joint_position)))

        # 同步obs 到simulator
        self.jdict['shoulder_pan_joint'].reset_position(joint_position[0], 0)
        self.jdict['shoulder_lift_joint'].reset_position(joint_position[1], 0)
        self.jdict['elbow_joint'].reset_position(joint_position[2], 0)
        self.jdict['wrist_1_joint'].reset_position(joint_position[3], 0)
        self.jdict['wrist_2_joint'].reset_position(joint_position[4], 0)
        self.jdict['wrist_3_joint'].reset_position(joint_position[5], 0)

        # print("joint_velocity,",joint_velocity)
        obs = eef_pos
        return obs

    def stop(self):
        self.ur5_rob_control.stop()

    def close(self):
        self.ur5_rob_control.close()



