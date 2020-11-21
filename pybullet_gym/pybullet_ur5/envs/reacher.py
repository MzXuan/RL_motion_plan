import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
from pybullet_envs.scene_abstract import SingleRobotEmptyScene
from pybullet_envs.env_bases import MJCFBaseBulletEnv
import numpy as np
from pybullet_envs.robot_bases import MJCFBasedRobot
import pybullet
import assets


# Ensure we get the path separator correct on windows

MODEL_XML_PATH = os.path.join(assets.getDataPath(), "reacher12.xml")
print(MODEL_XML_PATH)


class Reacher(MJCFBasedRobot):
  TARG_LIMIT = 0.27

  def __init__(self):
    MJCFBasedRobot.__init__(self, MODEL_XML_PATH, 'body0', action_dim=2, obs_dim=9)

  def reset(self, bullet_client):

    self._p = bullet_client
    #print("Created bullet_client with id=", self._p._client)
    if (self.doneLoading == 0):
      self.ordered_joints = []
      self.doneLoading = 1
      if self.self_collision:
        self.objects = self._p.loadMJCF(self.model_xml,
                                        flags=pybullet.URDF_USE_SELF_COLLISION |
                                        pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
            self._p, self.objects)
      else:
        self.objects = self._p.loadMJCF(
            self.model_xml)
        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
            self._p, self.objects)
    self.robot_specific_reset(self._p)

    s = self.calc_state(
    )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use

    return s

  def robot_specific_reset(self, bullet_client):
    self.jdict["target_x"].reset_current_position(
        self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
    self.jdict["target_y"].reset_current_position(
        self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
    self.fingertip = self.parts["fingertip"]
    self.target = self.parts["target"]
    self.central_joint = self.jdict["joint0"]
    self.elbow_joint = self.jdict["joint1"]
    self.central_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
    self.elbow_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)

  def apply_action(self, a):
    assert (np.isfinite(a).all())
    self.central_joint.set_motor_torque(0.05 * float(np.clip(a[0], -1, +1)))
    self.elbow_joint.set_motor_torque(0.05 * float(np.clip(a[1], -1, +1)))

  def calc_state(self):
    theta, self.theta_dot = self.central_joint.current_relative_position()
    self.gamma, self.gamma_dot = self.elbow_joint.current_relative_position()
    target_x, _ = self.jdict["target_x"].current_position()
    target_y, _ = self.jdict["target_y"].current_position()
    self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())
    return np.array([
        target_x,
        target_y,
        self.to_target_vec[0],
        self.to_target_vec[1],
        np.cos(theta),
        np.sin(theta),
        self.theta_dot,
        self.gamma,
        self.gamma_dot,
    ])

  def calc_potential(self):
    return -100 * np.linalg.norm(self.to_target_vec)


class ReacherBulletEnv(MJCFBaseBulletEnv):

  def __init__(self, render=False):
    self.robot = Reacher()
    MJCFBaseBulletEnv.__init__(self, self.robot, render)

  def create_single_player_scene(self, bullet_client):
    return SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=0.0165, frame_skip=1)

  def step(self, a):
    assert (not self.scene.multiplayer)
    self.robot.apply_action(a)
    self.scene.global_step()

    state = self.robot.calc_state()  # sets self.to_target_vec

    potential_old = self.potential
    self.potential = self.robot.calc_potential()

    electricity_cost = (
        -0.10 * (np.abs(a[0] * self.robot.theta_dot) + np.abs(a[1] * self.robot.gamma_dot)
                )  # work torque*angular_velocity
        - 0.01 * (np.abs(a[0]) + np.abs(a[1]))  # stall torque require some energy
    )
    stuck_joint_cost = -0.1 if np.abs(np.abs(self.robot.gamma) - 1) < 0.01 else 0.0
    self.rewards = [
        float(self.potential - potential_old),
        float(electricity_cost),
        float(stuck_joint_cost)
    ]
    self.HUD(state, a, False)
    return state, sum(self.rewards), False, {}

  def camera_adjust(self):
    x, y, z = self.robot.fingertip.pose().xyz()
    x *= 0.5
    y *= 0.5
    self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)




