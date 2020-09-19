import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, currentdir)

import assets
from scenes.stadium import StadiumScene, PlaneScene
from ur5_rg2 import UR5RG2Robot

import gym, gym.spaces, gym.utils, gym.utils.seeding
from gym.spaces import Tuple
import numpy as np
import pybullet
from pybullet_utils import bullet_client

import utils
import reward_utils

from pkg_resources import parse_version

class L(list):
    def append(self, item):
        list.append(self, item)
        if len(self) > 3: self[:1]=[]

def quaternion_to_euler(w, x, y, z):
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1,
                     np.sign(sinp) * np.pi / 2,
                     np.arcsin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.asarray([roll, pitch, yaw])


class UR5ReachEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 60}

    def __init__(self, render=False, max_episode_steps=1000):
        self.iter_num = 0
        self.max_episode_steps = max_episode_steps

        self.scene = None
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        # self.camera = Camera()
        self.isRender = render
        self.agent = UR5RG2Robot()
        self.seed()
        self._cam_dist = 3
        self._cam_yaw = 0
        self._cam_pitch = 30
        self._render_width = 160
        self._render_height = 120
        self.is_training = True

        self.last_eef_pos = L()

        # Set observation and action spaces

        self.agents_observation_space = self.agent.observation_space

        self.agents_action_space = self.agent.action_space

        self.history_len = 3

        self.image_width = 120
        self.image_height = 80
        # state_dim = self.agent.observation_space.shape[0]+3*(self.history_len-1)+7
        state_dim = self.agent.observation_space.shape[0] + 9
        obs_dim = state_dim+self.image_width*self.image_width
        act_dim = self.agent.action_space.shape[0]

        self.state_dim = state_dim

        high = np.ones([act_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf * np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        high = np.inf * np.ones([state_dim])
        self.state_space = gym.spaces.Box(-high, high)


    def set_training(self, is_training):
        self.is_training = is_training


    def configure(self, args):
        self.agent.args = args


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.agent.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def get_target_value(self):
        box_id_lst = []
        label_lst = []
        for box_id in self.box_ids:
            box_id_lst.append(box_id++((-1+1)<<24))
            label_lst.append(1 if box_id == self.goal_id else 0)


        return {"target_id": box_id_lst,"goal_label": label_lst}

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = PlaneScene(bullet_client, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)

        bullet_client.loadURDF(os.path.join(assets.getDataPath(), "scenes_data", "longtable/longtable.urdf"),
                               [-1, 0, 0],
                               [0.000000, 0.000000, 0.0, 1])

        self.collision_pos = [ [-0.8, 0.8, 1.15], [0.2, 0.1, 1.15]]
        bullet_client.loadURDF(os.path.join(assets.getDataPath(), "scenes_data", "targetbox/targetbox.urdf"),
                               self.collision_pos[0],
                               [0.000000, 0.000000, 0.0, 1])

        bullet_client.loadURDF(os.path.join(assets.getDataPath(), "scenes_data", "targetbox/targetbox.urdf"),
                              self.collision_pos[1],
                              [0.000000, 0.000000, 0.0, 1])

        # add target goal ball
        self.box_ids = []
        for i in range(3):
            id = bullet_client.loadURDF(
                os.path.join(assets.getDataPath(), "scenes_data", "targetball/targetball.urdf"),
                [0.0, 0.5, 0.98],
                [0.000000, 0.000000, 0.0, 1])

            self.box_ids.append(id)
            if i == 0:
                self.goal_id = id
                self._p.changeVisualShape(id, -1, rgbaColor=[0,255,0])

        return self.stadium_scene


    def reset(self):

        self.last_robot_eef = [0, 0, 0]

        self.last_eef_pos = L()

        if (self.physicsClientId < 0):
            self.ownsPhysicsClient = True

            if self.isRender:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            else:
                self._p = bullet_client.BulletClient()

            self.physicsClientId = self._p._client
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

        if self.scene is None:
            self.scene = self.create_single_player_scene(self._p)
        if not self.scene.multiplayer and self.ownsPhysicsClient:
            self.scene.episode_restart(self._p)

        self.camera_adjust()


        # reset env
        self.box_pos = []
        for index, id in enumerate(self.box_ids):
            distance_flag=True
            while(distance_flag):
                r = np.random.uniform(low=0.1, high =0.8)
                theta = np.random.uniform(low = 0, high = 3.14)
                # theta= 0
                # print("r is {} is theta is {} ".format(r, theta))
                x = 0.0 - r*np.cos(theta)
                y = 0.7 - (r-0.1)*np.sin(theta)
                # x = np.random.uniform(low=-0.6, high=0.6)
                # y = np.random.uniform(low=0.2, high=0.6)
                z = 0.98+0.01
                distance_flag = self._is_close([x,y,z])
                # distance_flag = False

                # print("x y z is,", x,y,z)
            if id == self.goal_id:
                self.goal = [x, y, z+0.15]

                self.goal_orient = [0.0, 0.707, 0.0, 0.707]
            self.box_pos.append([x, y, z+0.15])
            self._p.resetBasePositionAndOrientation(id, posObj=[x,y,z], ornObj=[0.0, 0.0, 0.0, 1.0])


        self.agent.scene = self.scene

        self.frame = 0
        self.iter_num = 0

        while True:
            state = self.agent.reset(self._p, base_position=[0.0, 1.0, 0.96], base_rotation=[0, 0, 0, 1])

            dones, obs, infos = self._get_obs()
            self.last_distance = np.linalg.norm((infos['rob_eef_position'] - infos['goal']))
            self.last_angle = abs(utils.quaternion_diff(infos['rob_eef_orient'], infos['goal_ori']))
            self.step(np.zeros(10))
            collisions = self._p.getContactPoints()
            if len(collisions) == 0:
                break

        return obs

    def _is_close(self, p, threshold = 0.4):
        for pc in self.collision_pos:
            dist = np.linalg.norm((p-np.asarray(pc)))
            if dist<0.4:
                return True

        for pp in self.box_pos:
            dist = np.linalg.norm((pp - np.asarray(p)))
            if dist < threshold:
                return True
        else:
            return False


    def render(self, mode='human', close=False):
        if mode == "human":
            self.isRender = True
        if mode != "rgb_array":
            return np.array([])


        base_pos = [0, 0, 10]
        if (hasattr(self, 'robot')):
            if (hasattr(self.robot, 'body_xyz')):
                base_pos = self.robot.body_xyz

        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                distance=self._cam_dist,
                                                                yaw=self._cam_yaw,
                                                                pitch=self._cam_pitch,
                                                                roll=0,
                                                                upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                         aspect=float(self._render_width) /
                                                                self._render_height,
                                                         nearVal=0.1,
                                                         farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(width=self._render_width,
                                                  height=self._render_height,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        if (self.ownsPhysicsClient):
            if (self.physicsClientId >= 0):
                self._p.disconnect()
        self.physicsClientId = -1

    def step(self, actions):
        self.iter_num += 1
        self.agent.apply_action(actions)

        self.scene.global_step()

        dones, obs, infos = self._get_obs()
        rewards = self._comp_reward(infos, obs)
        if self.iter_num > self.max_episode_steps:
            dones = True
            infos['done'] = dones

        return obs, rewards, dones, {}

    def _get_obs(self):
        infos = {}
        dones = False
        ur5_states = self.agent.calc_state()
        ur5_eef_position = ur5_states[:3]
        ur5_eef_oriention = ur5_states[3:7]

        # # debug: add user debug line
        # self._p.addUserDebugLine(self.last_robot_eef, ur5_eef_position, \
        #                          lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=10)
        #
        # self._p.addUserDebugLine(self.box_pos[0], self.box_pos[1], \
        #                          lineColorRGB=[0, 1, 0], lineWidth=2, lifeTime=2)
        # #
        # self._p.addUserDebugLine(self.box_pos[1], self.box_pos[2], \
        #                          lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=2)

        self.last_robot_eef = ur5_eef_position
        #
        # if np.linalg.norm((ur5_eef_position - self.goal)) < 0.06 and \
        #        abs(utils.quaternion_diff(ur5_eef_oriention, self.goal_orient)) < 0.6:
        if np.linalg.norm((ur5_eef_position - self.goal)) < 0.06:
            # if two eef position < threshold, done = True; else done=False
            dones = True
        else:
            dones = False

        infos['succeed'] = dones
        infos['rob_eef_position'] = ur5_eef_position
        infos['rob_eef_orient'] = ur5_eef_oriention
        infos['goal'] = self.goal
        infos['goal_ori'] = self.goal_orient
        infos['alternative_goals'] = self.box_pos[1:]

        collisions = self._p.getContactPoints()
        collision_bodies = []
        for c in collisions:
            bodyinfo1 = self._p.getBodyInfo(c[1])
            bodyinfo2 = self._p.getBodyInfo(c[2])

            collision_bodies.append(bodyinfo1[1].decode("utf-8"))
            collision_bodies.append(bodyinfo2[1].decode("utf-8"))

        infos['collision_bodies'] = collision_bodies

        if len(collisions) > 0:
            dones = True


        # #------ add last n steps to obs--------
        # self.last_eef_pos.append(ur5_states)
        # #calculate delta distance to all goals
        # last_poses = np.asarray(self.last_eef_pos)[:,0:3]
        # delta_d = []
        # if len(last_poses) > 1:
        #     delta_d.extend(reward_utils.delta_distance(last_poses, self.box_pos))
        # while len(delta_d) <self.history_len-1:
        #     delta_d.append(np.zeros(3,))


        # #------------------calculate distance to goals---------------
        # distance = [np.linalg.norm(ur5_eef_position - g) for g in self.box_pos]
        # angle_distance = utils.quaternion_diff(ur5_eef_oriention, self.goal_orient)

        # ------------------calculate xyz distance to goals---------------
        distance = [(ur5_eef_position - g) for g in self.box_pos]
        angle_distance = utils.quaternion_diff(ur5_eef_oriention, self.goal_orient)

        #add depth image
        depth = self.get_depth_img()

        # obs = np.concatenate([depth.flatten(), np.asarray(ur5_states), self.goal, self.goal_orient,
        #                       np.asarray(delta_d).flatten()])

        obs = dict()
        obs['depth'] = depth
        # obs['state'] = np.concatenate([np.asarray(ur5_states), np.asarray(distance).flatten(),
        #                                np.asarray([angle_distance])])
        obs['state'] = np.concatenate([np.asarray(ur5_states), np.asarray(distance).flatten()])

        return dones, obs, infos

    def get_depth_img(self):
        w, h = self.image_width, self.image_height
        far = 3
        near = 0.1
        fov = 60
        img_camera = self.get_camera_img(w, h, fov, near, far)
        depth_buffer_opengl = np.reshape(img_camera[3], [h, w, 1])
        # depth_buffer_opengl = np.reshape(img_camera[3], [1, h, w])
        depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)
        depth = depth_opengl / far
        # depth = np.zeros([self.image_height, self.image_width, 1])
        return depth

    def _comp_reward(self, infos, obs):
        '''
        Computer collision and approach reward
        :return:
        '''
        # success reward
        if infos['succeed'] == True:
            reward = 300
            return reward

        # collision reward
        reward_robot_col = 0
        collision_bodies = infos["collision_bodies"]
        if len(collision_bodies) != 0:

            if "ur5" in collision_bodies:  # robot collision
                reward_robot_col = -10

        # approaching reward
        current_distance = np.linalg.norm((infos['rob_eef_position'] - infos['goal']))
        reward_approach = 50 * (self.last_distance - current_distance)
        self.last_distance = current_distance

        # angle approaching
        current_angle = abs(utils.quaternion_diff(infos['rob_eef_orient'], infos['goal_ori']))
        reward_angle = 10 * (self.last_angle - current_angle)
        self.last_angle = current_angle
        task_reward = reward_robot_col + reward_approach
        # task_reward = reward_robot_col  + reward_approach + reward_angle
        # task_reward = reward_robot_col + reward_angle

        # legible_reward = 500*reward_utils.fast_decrease_distance_reward(obs, infos, self.iter_num)
        legible_reward = 0
        task_ratio = 0.8
        reward = task_ratio*task_reward+(1-task_ratio)*legible_reward
        # print("approach reward: {} and angle reward: {} and legible reward {}".format(reward_approach, reward_angle, legible_reward))
        return reward

    def get_camera_img(self, w, h, fov, nearVal, farVal):
        viewmat = self._p.computeViewMatrixFromYawPitchRoll(
            [0, 0.6, 1.5], 1.2, 0, -60, 0, upAxisIndex=2
        )

        projmat = self._p.computeProjectionMatrixFOV(fov=fov, aspect=w/float(h), nearVal=nearVal, farVal=farVal)

        return self._p.getCameraImage(width=w, height=h,
                                      viewMatrix=viewmat,
                                      projectionMatrix=projmat,
                                      shadow=False,
                                      renderer=self._p.ER_TINY_RENDERER,
                                      flags=self._p.ER_NO_SEGMENTATION_MASK
                                      )

    def camera_adjust(self):
        lookat = [0, 0.3, 1.4]
        distance = 1.0
        yaw = 0
        pitch = -30
        self._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)
        # self._p.getDebugVisualizerCamera(self._p.GUI)
        self.viewmat = \
            self._p.computeViewMatrixFromYawPitchRoll(\
                lookat, distance, yaw, pitch, 0, upAxisIndex=2 )

        # self.projmat = self._p.computeProjectionMatrix(\
        #     left=0, right=500, bottom=500, top=0,\
        #     nearVal = 0.2, farVal =50.0)


        self.projmat = self._p.computeProjectionMatrixFOV(fov=90, aspect=1,nearVal=0.1, farVal=3)

        # self.projmat  = [
        #     1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0, 0.0, 0.0, 0.0,
        #     -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0
        # ]
        # self.projmat = self._p.computeProjectionMatrix()



    def HUD(self, state, a, done):
        pass

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed

