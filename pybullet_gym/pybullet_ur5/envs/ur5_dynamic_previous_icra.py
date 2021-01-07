import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, currentdir)

import assets
from scenes.stadium import StadiumScene, PlaneScene

from ur5eef_previous import UR5EefPreviousRobot
from humanoid import URDFHumanoid, RealHumanoid
from pybullet_ur5.utils.ws_path_gen import WsPathGen

import gym, gym.spaces, gym.utils, gym.utils.seeding

import numpy as np
import pybullet
import colorsys
from pybullet_utils import bullet_client


from pybullet_planning import link_from_name, get_moving_links, get_link_name
from pybullet_planning import get_joint_names, get_movable_joints
from pybullet_planning import multiply, get_collision_fn
from pybullet_planning import sample_tool_ik
from pybullet_planning import set_joint_positions
from pybullet_planning import get_joint_positions, plan_joint_motion, compute_forward_kinematics


import ikfast_ur5
import pyquaternion

import pickle

from pkg_resources import parse_version

from mpi4py import MPI



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

def min_dist_conf(initial_conf, conf_list):

    dist_list = [np.linalg.norm([np.asarray(initial_conf)- np.asarray(conf)]) for conf in conf_list]
    id_min = np.argmin(np.asarray(dist_list))
    return conf_list[id_min]

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class L(list):
     def append(self, item):
         list.append(self, item)
         if len(self) > 3: del self[0]


class Moving_obstacle():
    def __init__(self, arm_id, max_speed=0.008):
        self.id = arm_id
        self.moving_speed = np.asarray([0,0,0])
        self.max_speed = max_speed
        self.range = [1.0, 1.0, 1.0]
        self.rob_base = [0, 0, 0]
        self.pos_list = L()
        self.ori_list = L()

    def set_rob_goal(self, rob_goal):
        self.rob_goal = rob_goal


    def apply_action(self):
        # current_ori = self._p.getBasePositionAndOrientation(bodyUniqueId=self.id)[1]

        # linkinfo = self._p.getLinkStates(bodyUniqueId=self.id, linkIndices=[0, 1])

        # check safety
        state = self.calc_state()

        safe_dist = 0.2

        current_pos = self._p.getBasePositionAndOrientation(bodyUniqueId=self.id)[0]
        if abs(current_pos[0])>self.range[0] or abs(current_pos[1])>self.range[1]\
            or abs(current_pos[2])>self.range[2] \
                or np.linalg.norm(state['current'][0]) < safe_dist \
                or np.linalg.norm(state['current'][1]) < safe_dist:
            self.rob_reset()
            # basepose = self._p.getBasePositionAndOrientation(bodyUniqueId=self.id)
            # current_pos = basepose[0]
            # current_ori = basepose[1]

        # move to next step
        self.create_next_action()
        next_pos = self.pos_list[0]
        next_ori = self.ori_list[0]


        self._p.resetBasePositionAndOrientation(bodyUniqueId = self.id, posObj = next_pos,ornObj = next_ori)

        # print("action self.pos_list", self.pos_list)



    def create_next_action(self):
        pos = self.pos_list[-1]
        ori =self.ori_list[-1]

        noise_vel = self.random_n(max=[0.01, 0.01, 0.01])
        if np.random.choice([1, 2, 3]) == 1:
            velocity = np.zeros(3)
        else:
            velocity = 0.15 * self.velocity * np.random.uniform(0.5, 1.1) + noise_vel


        # #------------for debug--------------------
        # hand_raw = np.asarray([-0.61864441, 1.26849507, 2.79468787])
        # elbow_raw = np.asarray([-0.94411114, 1.12510296, 2.55870706])
        #
        #
        #
        # center = (hand_raw + elbow_raw) / 2
        # next_pos = center
        #
        # hand_trans = (hand_raw - center) / np.linalg.norm([hand_raw - center])
        #
        # alpha = -np.arcsin(hand_trans[1])
        # beta = np.arcsin(hand_trans[0] / np.cos(alpha))
        #
        # next_ori = self._p.getQuaternionFromEuler([alpha, beta, 0])
        # self._p.addUserDebugLine(hand_raw , elbow_raw,
        #                          lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=10)
        # #-----------------done debug----------------------------

        #------origin-----------
        next_pos = np.asarray(pos) + velocity
        next_ori = self._p.getQuaternionFromEuler( \
            self._p.getEulerFromQuaternion(np.asarray(ori)) + self.rot_vel + noise_vel)
        #--------------------------------------

        # next_ori = [0, 0, 0, 1]

        self.pos_list.append(next_pos)
        self.ori_list.append(next_ori)

    def init_action_list(self):
        #create actions at timestep [0,1,2]
        self.pos_list = L()
        self.ori_list = L()
        basepose = self._p.getBasePositionAndOrientation(bodyUniqueId=self.id)
        current_pos = basepose[0]
        self.velocity = 0.15 * (self.human_goal - current_pos) / np.linalg.norm(current_pos - self.human_goal)
        self.rot_vel = self.random_n(max=[0.03, 0.0, 0.03])
        self.pos_list.append(basepose[0])
        self.ori_list.append(basepose[1])
        self.create_next_action()
        self.create_next_action()



    def rob_reset(self):
        success = False
        while not success:
            a = [self.human_goal[0], self.human_goal[1], self.human_goal[2]]
            x_b, y_b = np.random.choice([-1,1]), np.random.choice([-1,1])

            b = [x_b, y_b, -(a[0]*x_b+a[1]*y_b)/a[2]]

            r = np.random.uniform(0.5, 0.7)

            p_r = a + np.asarray(b)/np.linalg.norm(b) * r #p_r = vector a + vector b

            xh = p_r[0]
            yh = p_r[1]
            zh = p_r[2]
            # zh = self.human_goal[2] + np.random.uniform(-0.3, 0.3)

            pos = [xh, yh, zh]
            if np.linalg.norm(pos)>0.3:
                success=True
                ori = self._p.getQuaternionFromEuler(eulerAngles=self.random_n(max=[3.14, 3.14, 3.14]))
                self._p.resetBasePositionAndOrientation(bodyUniqueId=self.id, posObj=pos,
                                                        ornObj=ori)


        # initialize state list with current pos and orientation
        self.init_action_list()


    # def rob_reset(self):
    #     success = False
    #     while not success:
    #         a = [self.human_goal[0], self.human_goal[1]]
    #         x_b = np.random.choice([-1,1])
    #         b = [x_b, -1*a[0]/a[1] * x_b]
    #         d_l = np.random.uniform(0.3, 0.6)
    #
    #         p_r = a + np.asarray(b)/np.linalg.norm(b) * d_l
    #
    #         xh = p_r[0]
    #         yh = p_r[1]
    #         zh = self.human_goal[2] + np.random.uniform(-0.3, 0.3)
    #
    #         pos = [xh, yh, zh]
    #         if np.linalg.norm(pos)>0.3:
    #             success=True
    #
    #
    #     ori = self._p.getQuaternionFromEuler(eulerAngles=self.random_n(max=[3.14, 3.14, 3.14]))
    #     self._p.resetBasePositionAndOrientation(bodyUniqueId=self.id, posObj=pos,
    #                                                 ornObj=ori)
    #
    #     # initialize state list with current pos and orientation
    #     self.init_action_list()



    def reset(self, client, rob_goal):

        self._p = client
        self.rob_goal = rob_goal
        self.human_goal =  rob_goal+self.random_n(max=[0.1,0.1, 0.25], min=[-0.1,-0.1, -0.1])

        self.rob_reset()
        s = self.calc_state(
        )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use


        return s

    def calc_state(self):
        #backup base info
        current_base = self._p.getBasePositionAndOrientation(bodyUniqueId=self.id)
        #1. move link to positions at next n steps, save observations

        states = []

        # print("self.pos_list", self.pos_list)

        for (pos, ori) in zip(self.pos_list, self.ori_list):
            self._p.resetBasePositionAndOrientation(bodyUniqueId=self.id, posObj=pos,
                                                    ornObj=ori)
            linkinfo = self._p.getLinkStates(bodyUniqueId=self.id, linkIndices=[0, 1])
            base = self._p.getBasePositionAndOrientation(bodyUniqueId=self.id)
            #elbow, arm , hand
            # states.append([np.asarray(linkinfo[0][0]), np.asarray(base[0]), np.asarray(linkinfo[1][0])]) #elbow, arm, hand
            states.append(
                [np.asarray(linkinfo[0][0]), np.asarray(linkinfo[1][0])])  # elbow, hand 6

        obs = {"current": states[0],
               "next2": states[2]
        } #12

        # print("obs: ", obs)

        #2. set to current states
        self._p.resetBasePositionAndOrientation(bodyUniqueId=self.id, posObj=current_base[0],
                                                ornObj=current_base[1])

        # self._p.resetBasePositionAndOrientation(bodyUniqueId=self.id, posObj=current_base[0],
        #                                         ornObj=[0, 0, 0, 1])

        return obs


    def random_n(self, max, min=None):
        result = []
        if min is None:
            for m in max:
                result.append(np.random.uniform(-m, m))
        else:
            for s,e in zip(min,max):
                result.append(np.random.uniform(s,e))
        return np.asarray(result)




class UR5DynamicPreviousEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 60}

    def __init__(self, render=False, max_episode_steps=1000,
                 early_stop=False, distance_threshold = 0.05,
                 max_obs_dist = 0.35 ,dist_lowerlimit=0.02, dist_upperlimit=0.2,
                 reward_type="sparse"):
        self.iter_num = 0
        self.max_episode_steps = max_episode_steps

        self.scene = None
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self.isRender = render

        self.hz = 240
        self.sim_dt = 1.0 / self.hz
        self.frame_skip = 8


        self.agent = UR5EefPreviousRobot(dt= self.sim_dt*self.frame_skip)

        self._n_agents = 1
        self.seed()
        self._cam_dist = 1
        self._cam_yaw = 0
        self._cam_pitch = -30

        self.target_off_set=0.2
        self.distance_threshold = distance_threshold #for success

        self.max_obs_dist_threshold = max_obs_dist
        self.safe_dist_lowerlimit= dist_lowerlimit
        self.safe_dist_upperlimit = dist_upperlimit

        # self.distance_threshold = distance_threshold
        self.early_stop=early_stop
        self.reward_type = reward_type

        self.n_actions = 3
        self.action_space = gym.spaces.Box(-1., 1., shape=( self.n_actions,), dtype='float32')
        self.observation_space = gym.spaces.Dict(dict(
            desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            observation=gym.spaces.Box(-np.inf, np.inf, shape=(43,), dtype='float32'),
        ))

        # Set observation and action spaces
        self.agents_observation_space = self.agent.observation_space
        self.agents_action_space = self.agent.action_space



    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = PlaneScene(bullet_client, gravity=0, timestep=self.sim_dt, frame_skip=self.frame_skip)

        # self.long_table_body = bullet_client.loadURDF(os.path.join(assets.getDataPath(), "scenes_data", "longtable/longtable.urdf"),
        #                        [-1, -0.9, -1.0],
        #                        [0.000000, 0.000000, 0.0, 1])


        left_arm_id = bullet_client.loadURDF(os.path.join(assets.getDataPath(),
                                            "scenes_data", "cylinder/cylinder.urdf"),
                               [0, -0.5, 0.1], [0.000000, 0.000000, 0.0, 0.1], useFixedBase=True)

        right_arm_id = bullet_client.loadURDF(os.path.join(assets.getDataPath(),
                                                          "scenes_data", "cylinder/cylinder.urdf"),
                                             [0, -0.5, 0.1], [0.000000, 0.000000, 0.0, 0.1], useFixedBase=True)
        self.move_obstacles = [Moving_obstacle(left_arm_id),Moving_obstacle(right_arm_id) ]


        self.goal_id = bullet_client.loadURDF(
            os.path.join(assets.getDataPath(), "scenes_data", "targetball/targetball.urdf"),
            [0, 0, 0],
            [0.000000, 0.000000, 0.0, 1])

        # self.agents[1].set_goal_position( self.human_pos)
        return self.stadium_scene


    def set_training(self, is_training):
        self.is_training = is_training

    # def configure(self, args):
    #
    #
    #     self.agents[0].args = args[0]
    #     self.agents[1].args = args[1]

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.agent.np_random = self.np_random
        return [seed]

    def set_render(self):
        self.physicsClientId = -1
        self.isRender = True


    def reset(self):
        self.last_human_eef = [0, 0, 0]
        self.last_robot_eef = [0, 0, 0]
        self.last_robot_joint = np.zeros(6)
        self.current_safe_dist = self._set_safe_distance()


        if (self.physicsClientId < 0):
            self.ownsPhysicsClient = True

            if self.isRender:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            else:
                self._p = bullet_client.BulletClient()

            self._p.setGravity(0, 0, -9.81)
            self._p.setTimeStep(self.sim_dt)

            self.physicsClientId = self._p._client
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)


        if self.scene is None:
            self.scene = self.create_single_player_scene(self._p)
        if not self.scene.multiplayer and self.ownsPhysicsClient:
            self.scene.episode_restart(self._p)

        self.camera_adjust()

        self.agent.scene = self.scene


        self.frame = 0
        self.iter_num = 0

        self.robot_base = [0, 0, 0]

        #---- random goal -----#
        self.goal=np.zeros(3)

        collision_flag = True
        while collision_flag:

            x = np.random.uniform(-0.6, 0.6)
            y = np.random.uniform(-0.6, 0.6)
            z = np.random.uniform(0.1, 0.5)

            self.robot_start_eef = [x, y, z]

            self.goal = self.random_set_goal()
            if np.linalg.norm(self.goal) < 0.4:
                continue


            ar = self.agent.reset(self._p, client_id=self.physicsClientId,base_position=self.robot_base,
                                      base_rotation=[0, 0, 0, 1], eef_pose=self.robot_start_eef)
            for obstacles in self.move_obstacles:
                obstacles.reset(self._p, self.goal)
            if ar is False:
                # print("failed to find valid robot solution of pose", robot_eef_pose)
                continue

            self._p.stepSimulation()
            obs = self._get_obs()


            collision_flag = self._contact_detection()
            # print("collision_flag is :", collision_flag)

        s = []
        s.append(ar)
        self._p.resetBasePositionAndOrientation(self.goal_id, posObj=self.goal, ornObj=[0, 0, 0, 1])


        return obs

    def random_set_goal(self):
        max_xyz = [0.7, 0.7, 0.6]
        goal_reset = False
        while not goal_reset:
            goal = np.asarray(self.robot_start_eef.copy())
            goal[0] += np.random.uniform(-0.45, 0.45)
            goal[1] += np.random.uniform(-0.45, 0.45)
            goal[2] += np.random.uniform(-0.2, 0.2)
            if abs(goal[0]) < max_xyz[0] and abs(goal[1]) < max_xyz[1]\
                    and  abs(goal[1]) > 0.2 and abs(goal[2]) < max_xyz[2]:
                goal_reset = True
        return goal


    def _set_safe_distance(self):
        return 0.1
        # return np.random.uniform(self.safe_dist_lowerlimit, self.safe_dist_upperlimit)

    def get_obs(self):
        return self._get_obs()


    def _is_close(self, p, threshold = 0.3):
        dist = np.linalg.norm((np.asarray(self.robot_base)-np.asarray(p)))
        if dist < 0.4 or dist > 1.0:
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

        # self._p.setRealTimeSimulation(0)
        view_matrix = self.viewmat
        proj_matrix = self.projmat


        w = 320
        h = 240
        (_, _, px, _, _) = self._p.getCameraImage(width=w,
                                                  height=h,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.reshape(np.array(px), (h, w, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array


    def close(self):
        if (self.ownsPhysicsClient):
            if (self.physicsClientId >= 0):
                self._p.disconnect()
        self.physicsClientId = -1


    def step(self, action):
        self.iter_num += 1

        self.agent.apply_action(action)
        for obstacles in self.move_obstacles:
            obstacles.apply_action()

        self.scene.global_step()
        obs = self._get_obs()
        done = False


        #------- for debug----------
        # info = {
        #     'is_success': self._is_success(obs['achieved_goal'], self.goal),
        #     'is_collision': self._contact_detection(),
        #     'min_dist': 0.2,
        #     'safe_threshold': self.current_safe_dist
        # }
        #-------------------------------------------


        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'is_collision': self._contact_detection(),
            'min_dist': self.obs_min_dist,
            'safe_threshold': self.current_safe_dist,
            'ee_vel':obs["observation"][3:9]
        }

        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)


        if self.iter_num > self.max_episode_steps:
            done = True
        if self.early_stop:
            if info["is_success"] or info["is_collision"]:
                done = True

        return obs, reward, done, info



    def _get_obs(self):
        '''
        :return:
        obs_robot: [robot states, human states]
        obs_humanoid: [human states, robot states]
        info of every agent: [enf_effector_position, done]
        done of every agent: [True or False]
        '''
        infos = {}
        dones = [False for _ in range(self._n_agents)]
        ur5_states = self.agent.calc_state()
        ur5_eef_position = ur5_states[:3]
        achieved_goal = ur5_eef_position

        arm_states = []
        for obstacles in self.move_obstacles:
            arm_states.append(obstacles.calc_state())

        # arm_state = self.move_obstacle.calc_state()

        infos['succeed'] = dones


        # # ------ drawing ------#
        # self._p.addUserDebugLine(self.last_robot_eef, ur5_eef_position, \
        #                          lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=10)

        obs_human_states = []
        min_dists = []
        for arm_s in arm_states:
            d = [np.linalg.norm([p-ur5_eef_position]) for p in arm_s["current"]]

            min_dist = np.min(np.asarray(d))
            if min_dist >  self.max_obs_dist_threshold:
                min_dist =  self.max_obs_dist_threshold
            min_dists.append(min_dist)



            for p in arm_s["current"]:
                if np.linalg.norm([p-ur5_eef_position]) >  self.max_obs_dist_threshold:
                    obs_human_states.append(np.zeros(3)+self.max_obs_dist_threshold)
                else:
                    obs_human_states.append(p-ur5_eef_position)

            # for p in arm_state["next"]:
            #     if np.linalg.norm([p-ur5_eef_position]) >  self.max_obs_dist_threshold:
            #         obs_human_states.append(np.zeros(3)+self.max_obs_dist_threshold)
            #     else:
            #         obs_human_states.append(p-ur5_eef_position)

            for p in arm_s["next2"]:
                if np.linalg.norm([p-ur5_eef_position]) >  self.max_obs_dist_threshold:
                    obs_human_states.append(np.zeros(3)+self.max_obs_dist_threshold)
                else:
                    obs_human_states.append(p-ur5_eef_position)

        self.obs_min_dist = np.min(np.asarray(min_dists))

        obs = np.concatenate([np.asarray(ur5_states), np.asarray(obs_human_states).flatten(),
                              np.asarray(self.goal).flatten(), np.asarray([self.obs_min_dist])])


        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }



    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)

        #collision
        _is_collision = info['is_collision']
        if isinstance(_is_collision, np.ndarray):
            _is_collision = _is_collision.flatten()

        #safe distance
        min_dist = info['min_dist']
        safe_dist = info ['safe_threshold']

        if isinstance(min_dist, np.ndarray):
            min_dist = min_dist.flatten()
            safe_dist = safe_dist.flatten()

        distance = safe_dist - min_dist
        if isinstance(distance, np.ndarray):
            distance[(distance < 0)] = 0
        else:
            distance = 0 if distance<0 else distance

        smoothness = np.linalg.norm([info['ee_vel'][0:3] - info['ee_vel'][3:6]])

        # sum of reward
        a1 = -1
        a2 = -3
        a3 = -0.1
        asmooth = -0.01

        reward = a1 * (d > self.distance_threshold).astype(np.float32) \
                 + a2 * (_is_collision > 0) + a3 * distance + asmooth*smoothness
        reward_collision = -1.5*(_is_collision > 0) + (-6 * distance)
        # reward_collision = 1 * (_is_collision > 0)
        return [reward,reward_collision]

    def _contact_detection(self):
        # collision detection

        collisions = self._p.getContactPoints()
        collision_bodies = []
        for c in collisions:
            bodyinfo1 = self._p.getBodyInfo(c[1])
            bodyinfo2 = self._p.getBodyInfo(c[2])
            # print("bodyinfo1: ", bodyinfo1, "bodyinfo2: ", bodyinfo2)
            # print("collisions", collisions)
            # print("linkid 1 ", c[3])
            # print("linkid 2", c[4])

            # print("robot parts",self.agents[0].parts)
            if c[3] == 3 and c[4] == 5:
                continue
            if c[3] == 0 or c[4] == 0:
                continue
            # p = self._p.getLinkState(c[1], c[3])[0]
            collision_bodies.append(bodyinfo1[1].decode("utf-8"))
            collision_bodies.append(bodyinfo2[1].decode("utf-8"))

        if len(collision_bodies) != 0:
            if "ur5" in collision_bodies:  # robot collision
                # print("collision_bodies: ", collision_bodies)
                return True
            else:
                return False
        return False


    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)


    def camera_adjust(self):
        lookat = [0, -0.6, 0.8]
        distance = 1.0
        yaw = 0
        pitch = -30
        self._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)
        # self._p.getDebugVisualizerCamera(self._p.GUI)
        self.viewmat = \
            self._p.computeViewMatrixFromYawPitchRoll(lookat, distance, yaw, pitch, 0, upAxisIndex=2)

        self.projmat = self._p.computeProjectionMatrixFOV(fov=90, aspect=1, nearVal=0.1, farVal=3)

    def HUD(self, state, a, done):
        pass

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed



def load_demo_lst():
    # file_lst = ['/home/xuan/demos/task_demo1.pkl','/home/xuan/demos/task_demo2.pkl','/home/xuan/demos/task_demo3.pkl']
    # file_lst = ['/home/xuan/demos/task_demo1.pkl', '/home/xuan/demos/task_demo3.pkl']
    # file_lst = ['/home/xuan/demos/task_demo1.pkl']

    # file_lst = ['/home/xuan/demos/task_cloth_demo1.pkl', '/home/xuan/demos/task_cloth_demo2.pkl']
    # file_lst = ['/home/xuan/demos/task_cloth_demo2.pkl']

    # file_lst = ['/home/xuan/demos/task_task_demo1.pkl']

    # file_lst = ['/home/xuan/demos/plan_2.pkl']


    # file_lst = ['/home/xuan/demos/task_cloth_demo1.pkl', '/home/xuan/demos/task_cloth_demo2.pkl',
    #             '/home/xuan/demos/task_demo1.pkl','/home/xuan/demos/task_demo2.pkl','/home/xuan/demos/task_demo3.pkl',
    #             '/home/xuan/demos/plan_2.pkl', '/home/xuan/demos/task_task_demo1.pkl']

    file_lst = ['/home/xuan/demos/task_cloth_demo1.pkl', '/home/xuan/demos/task_demo1.pkl',
                '/home/xuan/demos/task_demo2.pkl','/home/xuan/demos/task_demo3.pkl',
                '/home/xuan/demos/plan_2.pkl']
    # file_lst = ['/home/xuan/demos/task_demo1.pkl', '/home/xuan/demos/task_demo3.pkl']]
    data_lst = []
    for f in file_lst:
        try:
            with open(f, 'rb') as handle:
                data_lst.append(pickle.load(handle))
            print("load data successfully")
        except:
            print("fail to load data, stop")
            return

    # print("length of data is: ", len(data))
    return data_lst


class L(list):
    def append(self, item):
        list.append(self, item)
        if len(self) > 6: del self[0]

class UR5PreviousTestEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 60}

    def __init__(self, render=False, max_episode_steps=200,
                 early_stop=False, distance_threshold = 0.10,
                 max_obs_dist = 0.35 ,dist_lowerlimit=0.02, dist_upperlimit=0.2,
                 reward_type="sparse"):
        self.iter_num = 0
        self.max_episode_steps = max_episode_steps

        self.scene = None
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self.isRender = render

        self.hz = 240
        self.sim_dt = 1.0 / self.hz
        self.frame_skip = 8


        self.agents = [UR5EefPreviousRobot(dt=self.sim_dt * self.frame_skip),
                       URDFHumanoid(max_obs_dist, load=True, test=True)]

        self._n_agents = 1
        self.seed()
        self._cam_dist = 1
        self._cam_yaw = 0
        self._cam_pitch = -30

        self.target_off_set=0.2
        self.distance_threshold = distance_threshold #for success

        self.max_obs_dist_threshold = max_obs_dist
        self.safe_dist_lowerlimit= dist_lowerlimit
        self.safe_dist_upperlimit = dist_upperlimit

        # self.distance_threshold = distance_threshold
        self.early_stop=early_stop
        self.reward_type = reward_type

        self.n_actions = 3
        self.action_space = gym.spaces.Box(-1., 1., shape=( self.n_actions,), dtype='float32')
        self.observation_space = gym.spaces.Dict(dict(
            desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            observation=gym.spaces.Box(-np.inf, np.inf, shape=(43,), dtype='float32'),
        ))

        # Set observation and action spaces
        # self.agents_observation_space = self.agent.observation_space
        # self.agents_action_space = self.agent.action_space

        # --- following demo----
        self.demo_data_lst = load_demo_lst()
        self.demo_id = 0
        self.default_r = 0.1
        self.sphere_radius = self.default_r
        self.last_collision = False

        self.last_human_obs_list = L(np.zeros((2, 12)))



    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = PlaneScene(bullet_client, gravity=0, timestep=self.sim_dt, frame_skip=self.frame_skip)

        # self.long_table_body = bullet_client.loadURDF(os.path.join(assets.getDataPath(), "scenes_data", "longtable/longtable.urdf"),
        #                        [-1, -0.9, -1.0],
        #                        [0.000000, 0.000000, 0.0, 1])

        self.long_table_body = bullet_client.loadURDF(os.path.join(assets.getDataPath(), "scenes_data", "circletable/circletable.urdf"),
                               [0, 0, -0.005],
                               [0.7068252, 0, 0, -0.7073883])

        self.goal_id = bullet_client.loadURDF(
            os.path.join(assets.getDataPath(), "scenes_data", "targetball/targetball.urdf"),
            [0, 0, 0],
            [0.000000, 0.000000, 0.0, 1])

        # self.agents[1].set_goal_position( self.human_pos)
        return self.stadium_scene


    def set_training(self, is_training):
        self.is_training = is_training


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        for a in self.agents:
            a.np_random = self.np_random
        return [seed]

    def set_render(self):
        self.physicsClientId = -1
        self.isRender = True


    def reset(self):
        self.last_obs_human = np.full(18, self.max_obs_dist_threshold + 0.2)
        self.last_human_eef = [0, 0, 0]
        self.last_robot_eef = [0, 0, 0]
        self.last_robot_joint = np.zeros(6)
        self.current_safe_dist = self._set_safe_distance()


        if (self.physicsClientId < 0):
            self.ownsPhysicsClient = True

            if self.isRender:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            else:
                self._p = bullet_client.BulletClient()

            self._p.setGravity(0, 0, -9.81)
            self._p.setTimeStep(self.sim_dt)

            self.physicsClientId = self._p._client
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)


        if self.scene is None:
            self.scene = self.create_single_player_scene(self._p)
        if not self.scene.multiplayer and self.ownsPhysicsClient:
            self.scene.episode_restart(self._p)

        self.camera_adjust()
        for a in self.agents:
            a.scene = self.scene


        self.frame = 0
        self.iter_num = 0

        self.robot_base = [0, 0, 0]

        #---- random goal -----#

        demo = self.demo_data_lst[self.demo_id]

        start_eef  = demo[0]['toolp']
        start_joint = demo[0]['robjp']
        ar = self.agents[0].reset(self._p, client_id=self.physicsClientId, base_position=self.robot_base,
                              base_rotation=[0, 0, 0, 1], eef_pose=start_eef, joint_angle=start_joint)
        ah = self.agents[1].reset(self._p, client_id=self.physicsClientId,
                                  base_rotation=[0.0005629, 0.707388, 0.706825, 0.0005633])

        #------prepare path----------------
        path = [demo[i]['toolp'] for i in range(len(demo))]
        vel_path = [demo[i]['toolv'] for i in range(len(demo))]
        joint_path = [demo[i]['robjp'] for i in range(len(demo))]

        self.eef_reference_path = path.copy()
        self.reference_path = joint_path.copy()

        self.ws_path_gen = WsPathGen(path, vel_path, joint_path, 0.12)

        #set goal from record demo-------------
        rob_eef = ar[:3]
        self.final_goal = np.array(demo[-1]['toolp'])
        next_goal = self._get_next_goal(rob_eef)
        self.goal, _, self.goal_indices = next_goal[0], next_goal[1], next_goal[2]


        self.demo_id+=1
        if self.demo_id>=len(self.demo_data_lst):
            self.demo_id=0
        #----------------------------------------------

        self._p.stepSimulation()
        obs = self._get_obs()

        s = []
        s.append(ar)
        s.append(ah)
        self._p.resetBasePositionAndOrientation(self.goal_id, posObj=self.goal, ornObj=[0, 0, 0, 1])

        return obs

    def _get_next_goal(self, ur5_eef_position):
        try:
            eef_goal, goal, _, goal_indices = self.ws_path_gen.next_goal(ur5_eef_position, self.sphere_radius)
            eef_goal = np.concatenate([eef_goal,np.array([0,0,0,1])])
            return (eef_goal[:3], np.array(goal), goal_indices)
        except:
            print("!!!!!!!!!!!!!!not exist self.ws_path_gen")
            return False


    def _set_safe_distance(self):
        return 0.1
        # return np.random.uniform(self.safe_dist_lowerlimit, self.safe_dist_upperlimit)

    def get_obs(self):
        return self._get_obs()


    def _is_close(self, p, threshold = 0.3):
        dist = np.linalg.norm((np.asarray(self.robot_base)-np.asarray(p)))
        if dist < 0.4 or dist > 1.0:
            return True

        # for pp in self.box_pos:
        #     dist = np.linalg.norm((pp - np.asarray(p)))
        #     if dist < threshold:
        #         return True
        # else:
        #     return False

    def render(self, mode='human', close=False):
        if mode == "human":
            self.isRender = True

        if mode != "rgb_array":
            return np.array([])

        base_pos = [0, 0, 10]
        if (hasattr(self, 'robot')):
            if (hasattr(self.robot, 'body_xyz')):
                base_pos = self.robot.body_xyz

        # self._p.setRealTimeSimulation(0)
        view_matrix = self.viewmat
        proj_matrix = self.projmat


        w = 320
        h = 240
        (_, _, px, _, _) = self._p.getCameraImage(width=w,
                                                  height=h,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.reshape(np.array(px), (h, w, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array


    def close(self):
        if (self.ownsPhysicsClient):
            if (self.physicsClientId >= 0):
                self._p.disconnect()
        self.physicsClientId = -1

    def draw_path(self):
        color_list = [[0.8, 0.8, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, 0.8],
                      [0.0, 0.0, 0.8], [0.0, 0.0, 0.8], [0.0, 0.0, 0.8],
                      [0.0, 0.0, 0.8], [0.0, 0.0, 0.8], [0.0, 0.0, 0.8], [0.0, 0.0, 0.8]]
        for d_idx, demo in enumerate(self.demo_data_lst):
            idxs = list(np.linspace(0, len(demo) - 2, 20))
            for i in range(len(idxs) - 1):
                self._p.addUserDebugLine(demo[int(idxs[i])]['toolp'],
                                         demo[int(idxs[i + 1])]['toolp'],
                                         lineColorRGB=color_list[d_idx], lineWidth=3)

    def set_sphere(self, r):
        self.sphere_radius = r

    def step(self, action):
        self.iter_num += 1

        self.agents[0].apply_action(action)
        self.agents[1].apply_action(0)


        self.scene.global_step()
        obs = self._get_obs()
        done = False



        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.final_goal),
            'is_collision': self._contact_detection(),
            'min_dist': self.obs_min_dist,
            'safe_threshold': self.current_safe_dist,
            'ee_vel':obs["observation"][9:15]
        }

        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)


        if self.iter_num > self.max_episode_steps:
            done = True
        if self.early_stop:
            if info["is_success"] or info["is_collision"]:
                done = True

        self._p.resetBasePositionAndOrientation(self.goal_id, posObj=self.goal, ornObj=[0, 0, 0, 1])

        return obs, reward, done, info

    def update_robot_obs(self, current_obs, next_rob_state):
        # change robot state to certain state and update obs
        ur5_states = current_obs[0:15].copy()
        # print("shape of next rob state: ", next_rob_state.shape)
        ur5_states[:9] = next_rob_state[:9]

        ur5_eef_position = ur5_states[:3]

        hs = self.agents[1].calc_state()
        self.human_states = [hs[1], hs[2], hs[4], hs[5]]

        delta_p = np.asarray([(p - ur5_eef_position) for p in self.human_states])
        d = np.linalg.norm(delta_p, axis=1)
        min_dist = np.min(d)

        # clip obs
        obs_human = delta_p.copy()
        indices = np.where(d > self.max_obs_dist_threshold)
        obs_human[indices] = np.full((1, 3), self.max_obs_dist_threshold + 0.2)

        # print("shape of obs human", obs_human.shape)

        # human_obs_input = np.asarray(self.last_human_obs_list).flatten()

        obs_human_states = np.asarray(self.last_human_obs_list[-2:])
        obs_human_states[-1:]=obs_human.flatten()


        # print("human_obs_input", self.last_human_obs_list)
        # obs = np.concatenate([np.asarray(ur5_states), human_obs_input,
        #                       np.asarray(self.goal).flatten(), np.asarray([min_dist])])

        obs = np.concatenate([np.asarray(ur5_states), np.asarray(obs_human).flatten(),np.asarray(obs_human).flatten(),
                              np.asarray(self.goal).flatten(), np.asarray([min_dist])])

        # obs = np.concatenate([np.asarray(ur5_states), np.asarray(obs_human_states).flatten(),
        #                       np.asarray(self.goal).flatten(), np.asarray([min_dist])])

        # print("obs: ", obs)


            # human_obs_input = np.asarray(self.last_human_obs_list[-2:]).flatten()
            # obs = np.concatenate([np.asarray(ur5_states), human_obs_input,
            #                       np.asarray(self.goal).flatten(), np.asarray([min_dist])])

        achieved_goal = ur5_states[3:9]  # ur5 joint
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def update_r(self, obs_lst, q_lst, draw=True):
        self.q_thre = 1
        q_lst = np.asarray(q_lst)
        q_sum = np.sum(q_lst)
        # print("!!!!!!q_lst!!!!!!!", q_lst)
        # print("!!!!!!q_sum!!!!!!!", q_sum)

        if q_sum < -0.2:
            # print("min_q", min_q)
            self.last_collision = True
            self.set_sphere(0.5)
            # self.set_sphere(self.default_r)
            # normalize Q for color
            minimum_q = -0.3
            b = 0.6
            k = -b / minimum_q
            color_lst = q_lst * k + b

            if draw:
                self.draw_q(obs_lst, q_lst, color_lst)
        else:
            self.set_sphere(self.default_r)
            return 0

    def draw_q(self, obs_lst, q_lst, color_lst):


        for obs, q, c in zip(obs_lst, q_lst, color_lst):
            if q < self.q_thre:
                # self._p.addUserDebugText(text=str(q)[1:7], textPosition=obs['observation'][:3],
                #                          textSize=1.2, textColorRGB=colorsys.hsv_to_rgb(0.5 - c / 2, c + 0.5, c),
                #                          lifeTime=1)
                self._p.addUserDebugText(text=str(q)[1:7], textPosition=obs['observation'][:3],
                                         textSize=1.2, textColorRGB=colorsys.hsv_to_rgb(c, 1-c, 0.8),
                                         lifeTime=1)


    def _get_obs(self):
        '''
        :return:
        obs_robot: [robot states, human states]
        obs_humanoid: [human states, robot states]
        info of every agent: [enf_effector_position, done]
        done of every agent: [True or False]
        '''
        infos = {}
        dones = [False for _ in range(self._n_agents)]

        ur5_states = self.agents[0].calc_state()
        ur5_eef_position = ur5_states[:3]
        achieved_goal = ur5_eef_position

        # interface for online test (following path)
        next_goal = self._get_next_goal(ur5_eef_position)
        if next_goal is False:
            pass
        else:
            self.goal, _, self.goal_indices = next_goal[0],next_goal[1],next_goal[2]

        # ------human observation-----
        hs = self.agents[1].calc_state()
        self.human_states =[hs[1], hs[2], hs[4], hs[5]]
        infos['succeed'] = dones

        # # ------ drawing ------#
        # self._p.addUserDebugLine(self.last_robot_eef, ur5_eef_position, \
        #                          lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=10)

        obs_human_states = []
        min_dists = []

        delta_p = np.asarray([(p - ur5_eef_position) for p in self.human_states])
        d = np.linalg.norm(delta_p, axis=1)
        min_dist = np.min(d)

        # clip obs
        obs_human = delta_p.copy()
        indices = np.where(d > self.max_obs_dist_threshold)
        obs_human[indices] = np.full((1, 3), self.max_obs_dist_threshold + 0.2)

        self.obs_min_dist = min_dist
        self.last_human_obs_list.append(np.asarray(obs_human.copy()).flatten())

        obs_human_states = np.asarray(self.last_human_obs_list[-2:]).flatten()


        obs = np.concatenate([np.asarray(ur5_states), np.asarray(obs_human_states).flatten(),
                              np.asarray(self.goal).flatten(), np.asarray([self.obs_min_dist])])


        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }



    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)

        #collision
        _is_collision = info['is_collision']
        if isinstance(_is_collision, np.ndarray):
            _is_collision = _is_collision.flatten()

        #safe distance
        min_dist = info['min_dist']
        safe_dist = info ['safe_threshold']

        if isinstance(min_dist, np.ndarray):
            min_dist = min_dist.flatten()
            safe_dist = safe_dist.flatten()

        distance = safe_dist - min_dist
        if isinstance(distance, np.ndarray):
            distance[(distance < 0)] = 0
        else:
            distance = 0 if distance<0 else distance

        smoothness = np.linalg.norm([info['ee_vel'][0:3] - info['ee_vel'][3:6]])

        # sum of reward
        a1 = -1
        a2 = -3
        a3 = -0.1
        asmooth = -0.01

        reward = a1 * (d > self.distance_threshold).astype(np.float32) \
                 + a2 * (_is_collision > 0) + a3 * distance + asmooth*smoothness
        reward_collision = -1.5*(_is_collision > 0) + (-6 * distance)
        # reward_collision = 1 * (_is_collision > 0)
        return [reward,reward_collision]

    def _contact_detection(self):
        # collision detection

        collisions = self._p.getContactPoints()
        collision_bodies = []
        for c in collisions:
            bodyinfo1 = self._p.getBodyInfo(c[1])
            bodyinfo2 = self._p.getBodyInfo(c[2])
            # print("bodyinfo1: ", bodyinfo1, "bodyinfo2: ", bodyinfo2)
            # print("collisions", collisions)
            # print("linkid 1 ", c[3])
            # print("linkid 2", c[4])

            # print("robot parts",self.agents[0].parts)
            if c[3] == 3 and c[4] == 5:
                continue
            if c[3] == 0 or c[4] == 0:
                continue
            # p = self._p.getLinkState(c[1], c[3])[0]
            collision_bodies.append(bodyinfo1[1].decode("utf-8"))
            collision_bodies.append(bodyinfo2[1].decode("utf-8"))

        if len(collision_bodies) != 0:
            if "ur5" in collision_bodies:  # robot collision
                # print("collision_bodies: ", collision_bodies)
                return True
            else:
                return False
        return False


    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)


    def camera_adjust(self):
        lookat = [0, -0.6, 0.8]
        distance = 1.0
        yaw = 0
        pitch = -30
        self._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)
        # self._p.getDebugVisualizerCamera(self._p.GUI)
        self.viewmat = \
            self._p.computeViewMatrixFromYawPitchRoll(lookat, distance, yaw, pitch, 0, upAxisIndex=2)

        self.projmat = self._p.computeProjectionMatrixFOV(fov=90, aspect=1, nearVal=0.1, farVal=3)

    def HUD(self, state, a, done):
        pass

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
