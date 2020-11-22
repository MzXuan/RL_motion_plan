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

import ikfast_ur5


def normalize_conf(start, end):
    circle = 6.2831854
    normal_e = []
    for s, e in zip(start, end):
        if e > s:
            test = e - circle
        else:
            test = e + circle
        normal_e.append(test if np.linalg.norm(test-s)<np.linalg.norm(e-s) else e)
    return normal_e

class UR5Robot(robot_bases.URDFBasedRobot):
    TARG_LIMIT = 0.27
    def __init__(self, dt, action_dim=6, obs_dim=13):
        # self.select_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",\
        #                     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        # self.select_links = ["shoulder_link", "upper_arm_link","forearm_link","ee_link"]
        self.select_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

        self.select_links = ["upper_arm_link", "wrist_3_link"]
        self.select_links = ["upper_arm_link", "wrist_3_link"]
        self.ee_link = "ee_link"
        self.last_position = [0, 0, 0]

        self.lower_limits = [-6, -6, 0, -6, -6, -6]
        self.upper_limits = [6, 0, 6, 0, 6, 6]
        # self.orientation = [0, 0.7071068, 0, 0.7071068]
        # self.orientation = [0.707, 0, 0.707, 0]
        self.orientation = [0, 0.841471, 0, 0.5403023]
        self.dt = dt
        self.n_dofs = 6

        super(UR5Robot, self).__init__(
            'ur_description/ur5.urdf', "ur5_robot", action_dim=action_dim, obs_dim=obs_dim, fixed_base=1, self_collision=True)

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


            collision_bodies.append(bodyinfo1[1].decode("utf-8"))
            collision_bodies.append(bodyinfo2[1].decode("utf-8"))

        if len(collision_bodies) != 0:
            if "ur5" in collision_bodies:  # robot collision
                return True
            else:
                return False

        return False

    def robot_specific_reset(self, bullet_client, base_position, base_rotation,
                             eef_position=None, eef_orienration=[0, 0.841471, 0, 0.5403023],
                             joint_angle=None):

        if joint_angle is not None:
            print("'reset joint angle", joint_angle)

            self.jdict['shoulder_pan_joint'].reset_position(joint_angle[0], 0)
            self.jdict['shoulder_lift_joint'].reset_position(joint_angle[1], 0)
            self.jdict['elbow_joint'].reset_position(joint_angle[2], 0)
            self.jdict['wrist_1_joint'].reset_position(joint_angle[3], 0)
            self.jdict['wrist_2_joint'].reset_position(joint_angle[4], 0)
            self.jdict['wrist_3_joint'].reset_position(joint_angle[5], 0)

            self.last_position = list(
                self._p.getLinkState(self.robot_body.bodies[0], self.parts['ee_link'].bodyPartIndex)[0])
            return


    def reset(self, bullet_client, client_id, base_position=[0, 0, -1.2], base_rotation=[0, 0, 0, 1],
              eef_pose=None, joint_angle=None):

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

                self.controlled_joints_id = [self.jdict[joint_name].jointIndex for joint_name in self.select_joints]

            else:
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p,
                    self._p.loadURDF(os.path.join(assets.getDataPath(), self.model_urdf),
                                     basePosition=self.basePosition,
                                     baseOrientation=self.baseOrientation,
                                     useFixedBase=self.fixed_base))

        if self.robot_specific_reset(self._p, base_position, base_rotation, eef_pose, joint_angle=joint_angle) is False:
            return False
        self.jdict['shoulder_pan_joint'].max_velocity = 3.15
        self.jdict['shoulder_lift_joint'].max_velocity = 3.15
        self.jdict['elbow_joint'].max_velocity = 3.15
        self.jdict['wrist_1_joint'].max_velocity = 3.2
        self.jdict['wrist_2_joint'].max_velocity = 3.2
        self.jdict['wrist_3_joint'].max_velocity = 3.2

        # self.last_eef_position = self.parts['ee_link'].get_position()

        s = self.calc_state(
        )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
        self.potential = self.calc_potential()

        return s

    def apply_action(self, a):
        # set to position

        assert (np.isfinite(a).all())
        for i in range((self.action_space.shape)[0]):
            # print("i is", i)
            # print("a", a)
            # self.jdict[self.select_joints[i]].set_position(a[i])
            self.jdict[self.select_joints[i]].reset_position(a[i],0.1)



    def calc_state(self):

        joint_position = np.asarray(\
            [self.jdict[i].get_position() for i in self.select_joints if i in self.jdict])  # position

        ee_lin_pos = self.parts['ee_link'].get_position()

        obs = np.concatenate([ee_lin_pos.flatten(), joint_position.flatten()])

        return obs



