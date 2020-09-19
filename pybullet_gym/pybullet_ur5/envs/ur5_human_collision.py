import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, currentdir)

import assets
from scenes.stadium import StadiumScene, PlaneScene
from humanoid import EefHumanoid, SelfMoveHumanoid, SelfMoveAwayHumanoid
from ur5_rg2 import UR5RG2Robot
from ur5eef import UR5EefRobot

import gym, gym.spaces, gym.utils, gym.utils.seeding
from gym.spaces import Tuple
import numpy as np
import pybullet
from pybullet_utils import bullet_client

import utils
import random

from pybullet_planning import link_from_name, get_moving_links, get_link_name
from pybullet_planning import get_joint_names, get_movable_joints
from pybullet_planning import multiply, get_collision_fn
from pybullet_planning import sample_tool_ik
from pybullet_planning import set_joint_positions
from pybullet_planning import get_joint_positions, plan_joint_motion, compute_forward_kinematics


import ikfast_ur5

from scipy.interpolate import griddata

from pkg_resources import parse_version

from mpi4py import MPI


def calculate_bc_loss(p_rob, reference_traj):
    dist = []
    for p_ref in reference_traj:
        dist.append(np.linalg.norm([p_rob-p_ref]))
    dist = np.asarray(dist)
    min_dist = np.min(dist)
    progress = np.argmin(dist)/len(dist)
    return min_dist, progress

def calculate_bc_ratio(p_rob, reference_traj, ph, pl):
    dist = []
    for p_ref in reference_traj:
        dist.append(np.linalg.norm([p_rob-p_ref]))
    dist = np.asarray(dist)
    min_dist = np.min(dist)
    progress = np.argmin(dist) / len(dist)

    if min_dist < pl:
        ratio = 0
    elif min_dist>ph:
        ratio = 1
    else:
        ratio = ((min_dist-pl)/(ph-pl))**2
    return (1-ratio), progress

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


class UR5HumanCollisionEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 60}

    def __init__(self, render=False, max_episode_steps=1000):
        self.distance_close = 0.3

        self.iter_num = 0
        self.max_episode_steps = max_episode_steps

        self.scene = None
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self.isRender = render
        self.agents = [UR5RG2Robot(), SelfMoveHumanoid(0, 12)]
        # self.agents = [UR5RG2Robot(), SelfMoveAwayHumanoid(0, 12)]
        # self.agents = [UR5Robot(), SelfMoveAwayHumanoid(0, 12)]
        # self.agents = [UR5EefRobot(3,), SelfMoveHumanoid(0, 12)]
        # self.agents = [UR5EefRobot(3, ), SelfMoveAwayHumanoid(0, 12)]
        self._n_agents = 2
        self.seed()
        self._cam_dist = 3
        self._cam_yaw = 0
        self._cam_pitch = 30
        self._render_width = 320
        self._render_height = 240
        self.is_training = True

        self.success_range = 0.1

        # Set observation and action spaces
        self.agents_observation_space = Tuple([
            agent.observation_space for agent in self.agents
        ])
        self.agents_action_space = Tuple([
            agent.action_space for agent in self.agents
        ])

        obs_dim, act_dim = 0, 0
        for agent in self.agents:
            obs_dim += agent.observation_space.shape[0]
            act_dim += agent.action_space.shape[0]

        obs_dim+=22

        high = np.ones([act_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf * np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)


        self.target_off_set=0.2
        self.safe_dist_threshold = 0.6
        self.dpath = 0
        self.progress = 0


    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = PlaneScene(bullet_client, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)

        # self.long_table_body = bullet_client.loadURDF(os.path.join(assets.getDataPath(), "scenes_data", "longtable/longtable.urdf"),
        #                        [-1, -0.9, -1.2],
        #                        [0.000000, 0.000000, 0.0, 1])

        # add target box goals
        self.box_ids = []
        self.box_pos = []

        self.human_pos = []
        for i in range(6):
            for j in range(2):
                x = (i - 3.0) / 5
                y = (j - 4) / 6
                z = -0.2

                id_temp = bullet_client.loadURDF(os.path.join(assets.getDataPath(),
                                                              "scenes_data", "targetbox/targetbox.urdf"),
                                                 [x, y, z], [0.000000, 0.000000, 0.0, 1])
                if j == 1:
                    self.box_ids.append(id_temp)
                    self.box_pos.append([x, y, z])
                # if j == 0:
                #     bullet_client.changeVisualShape(id_temp, -1, rgbaColor=[1,1,0,1])
                #     self.human_pos.append([x,y,z])

                bullet_client.changeVisualShape(id_temp, -1, rgbaColor=[1,1,0,1])
                self.human_pos.append([x,y,z])



        self.goal_id = bullet_client.loadURDF(
            os.path.join(assets.getDataPath(), "scenes_data", "targetball/targetball.urdf"),
            [0, 0, 0],
            [0.000000, 0.000000, 0.0, 1])

        self.agents[1].set_goal_position( self.human_pos)

        return self.stadium_scene

    def set_training(self, is_training):
        self.is_training = is_training

    def configure(self, args):
        self.agents[0].args = args[0]
        self.agents[1].args = args[1]

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        for a in self.agents:
            a.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def reset(self):
        self.last_human_eef = [0, 0, 0]
        self.last_robot_eef = [0, 0, 0]
        self.last_robot_joint = np.zeros(6)

        if (self.physicsClientId < 0):
            self.ownsPhysicsClient = True

            if self.isRender:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            else:
                self._p = bullet_client.BulletClient()

            self.physicsClientId = self._p._client
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

            # print("self.physicsClientId: ", self.physicsClientId)

        # #write id
        # rank = MPI.COMM_WORLD.Get_rank()
        # path = "/home/xuan/Client_id"
        # fname = "ClientId" + str(rank) + ".txt"
        # file = os.path.join(path, fname)
        # f = open(file, "w")
        # f.write("ClientId: {}".format(self.physicsClientId))
        # f.close()


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

        # ---select goal from boxes---#
        self.goal = np.asarray(random.choice(self.box_pos)) #robot goal
        self.goal[2]+=0.1
        self.goal_orient = [0.0, 0.707, 0.0, 0.707]


        reset_success_flag = False
        while not reset_success_flag:
            # # ---random goal in space---#
            # self.goal = np.zeros(3)
            # while np.linalg.norm(self.goal) < 0.3:
            #     self.goal = np.asarray([np.random.uniform(-0.6, 0.6),
            #                             np.random.uniform(-0.5, -0.1),
            #                             np.random.uniform(-0.2, 0) + self.target_off_set])
            #
            # self.goal_orient = [0.0, 0.707, 0.0, 0.707]



            #
            x = np.random.uniform(0.3, 0.6)
            y = np.random.uniform(-0.7, -0.5)
            z = np.random.uniform(0.2, 0.6)

            robot_eef_pose = [np.random.choice([-1, 1])*x, y, z]

            self.robot_start_eef = robot_eef_pose

            s = []
            ar = self.agents[0].reset(self._p, base_position=self.robot_base, base_rotation=[0, 0, 0, 1], eef_pose=self.robot_start_eef)
            ah = self.agents[1].reset(self._p, base_position=[0.0, -1.2, -1.3], \
                                      base_rotation=[0, 0, 0.7068252, 0.7073883], is_training=self.is_training)
            s.append(ar)
            s.append(ah)

            dones, obs, infos = self._get_obs()
            # obs = np.concatenate([obs, self._get_next_ref_traj()])
            self.last_distance = np.linalg.norm((infos['rob_eef_position'] - infos['goal']))
            self.last_angle = abs(utils.quaternion_diff(infos['rob_eef_orient'], infos['goal_ori']))
            self.last_progress = 0
            self.step(np.zeros(10))

            # print("obs: ", obs)


        #     # --- no motion planner---#
        #     collisions = self._p.getContactPoints()
        #     if len(collisions) == 0:
        #         reset_success_flag = True
        #     else:
        #         reset_success_flag = False
        #     start = infos['rob_eef_position']
        #     goal = infos['goal']
        #     self.reference_traj = np.linspace(start, goal, num=50, endpoint=True)
        # self._p.resetBasePositionAndOrientation(self.goal_id, posObj=infos['goal'], ornObj=[0.0, 0.0, 0.0, 1.0])
        # for i in range(len(self.reference_traj)-1):
        #     self._p.addUserDebugLine(self.reference_traj[i], self.reference_traj[i+1], \
        #                              lineColorRGB=[0, 1, 1], lineWidth=4, lifeTime=20)

            #---- for motion planner---#
            collisions = self._p.getContactPoints()
            if len(collisions) != 0:
                reset_success_flag = False
                continue

            result = self.motion_planner()
            if result is not None:
                (joint_path, cartesian_path) = result
                normal_joint_path = [self.agents[0].normalize_joint(j) for j in joint_path]
                self.reference_traj = np.asarray(normal_joint_path)

                reset_success_flag = True

        self._p.resetBasePositionAndOrientation(self.goal_id, posObj=infos['goal'], ornObj=[0.0, 0.0, 0.0, 1.0])
        for i in range(len(cartesian_path)-1):
            self._p.addUserDebugLine(cartesian_path[i], cartesian_path[i+1], \
                                     lineColorRGB=[0, 1, 1], lineWidth=4, lifeTime=20)



        # #--- way point line ----#
        # start = infos['rob_eef_position']
        # goal = infos['goal']
        # middle_point = (start + goal)/2
        # middle_point[2] += 0.2
        # traj_1 = np.linspace(start, middle_point, 50)
        # traj_2 = np.linspace(middle_point, goal, 50)
        # self.reference_traj = np.concatenate([traj_1, traj_2])
        #
        # for i in range(len(self.reference_traj)-1):
        #     self._p.addUserDebugLine(self.reference_traj[i], self.reference_traj[i+1], \
        #                              lineColorRGB=[0, 1, 1], lineWidth=4, lifeTime=20)

        return obs

    def motion_planner(self):
        robot, ik_joints = self._test_moving_links_joints()
        workspace = self.agents[1].objects[0]
        robot_base_link_name = 'base_link'
        robot_base_link = link_from_name(robot, robot_base_link_name)

        ik_fn = ikfast_ur5.get_ik
        fk_fn = ikfast_ur5.get_fk

        # we have to specify ik fn wrapper and feed it into pychoreo
        def get_sample_ik_fn(robot, ik_fn, robot_base_link, ik_joints,tool_from_root=None):
            def sample_ik_fn(world_from_tcp):
                if tool_from_root:
                    world_from_tcp = multiply(world_from_tcp, tool_from_root)
                return sample_tool_ik(ik_fn, robot, ik_joints, world_from_tcp, robot_base_link, get_all=True)

            return sample_ik_fn




        collision_fn = get_collision_fn(robot, ik_joints,
                                        obstacles=[workspace], attachments=[],
                                        self_collisions=True,
                                        #    disabled_collisions=disabled_collisions,
                                        #    extra_disabled_collisions=extra_disabled_collisions,
                                        custom_limits={})
        # Let's check if our ik sampler is working properly
        sample_ik_fn = get_sample_ik_fn(robot, ik_fn, robot_base_link, ik_joints)
        p_end = (self.goal, self.goal_orient)

        # calculate initial ik and end ik
        initial_conf = get_joint_positions(robot, ik_joints)

        #calculate end ik
        qs = sample_ik_fn(p_end)
        if collision_fn is not None:
            conf_list = [conf for conf in qs if conf and not collision_fn(conf, diagnosis=False)]
        try:
            conf_list[0]
        except:
            return None
        n_conf_list = [normalize_conf(np.asarray(initial_conf), conf) for conf in conf_list]
        final_conf = min_dist_conf(initial_conf, n_conf_list)

        # print("initial conf: ", initial_conf)
        # print("goal is: ", self.goal)
        # print("fina_conf: ", final_conf)

        # set robot to initial configuration
        set_joint_positions(robot, ik_joints, initial_conf)

        # start planning
        path = plan_joint_motion(robot, ik_joints, final_conf, obstacles=[workspace],
                                 self_collisions=True, diagnosis=False)

        #set robot to initial configuration
        set_joint_positions(robot, ik_joints, initial_conf)

        if path is None:
            return None
        else:
            cartesion_path = [compute_forward_kinematics(fk_fn, conf)[0] for conf in path]

        return (path, cartesion_path)


    def _test_moving_links_joints(self):
        robot = self.agents[0].robot_id
        workspace = self.agents[1].objects[0]
        assert isinstance(robot, int) and isinstance(workspace, int)

        movable_joints = get_movable_joints(robot)
        assert isinstance(movable_joints, list) and all([isinstance(mj, int) for mj in movable_joints])
        assert 6 == len(movable_joints)
        assert [b'shoulder_pan_joint', b'shoulder_lift_joint', b'elbow_joint', b'wrist_1_joint', b'wrist_2_joint',
                b'wrist_3_joint'] == \
               get_joint_names(robot, movable_joints)

        return robot, movable_joints


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

        for i, agent in enumerate(self.agents):
            act_dim = self.agents_action_space[i].shape[0]
            if i == 0:
                act = actions[:act_dim]
            elif i == 1:
                act = actions[-act_dim:]
            else:
                print("ERROR: Cannot support more than two agents!")
                return False

            agent.apply_action(act)

        self.scene.global_step()

        dones, obs, infos = self._get_obs()
        rewards = self._comp_reward(infos)
        # obs = np.concatenate([obs, self._get_next_ref_traj()])

        self.last_human_eef = infos['human_eef_position']
        self.last_robot_eef = infos['rob_eef_position']
        self.last_robot_joint = infos['rob_joint_state']

        if self.iter_num > self.max_episode_steps:
            dones = True
            infos['done'] = dones

        return obs, rewards, dones, infos

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
        ur5_eef_oriention = ur5_states[3:7]
        humanoid_states = self.agents[1].calc_state()
        # print("humanoid state is: ", humanoid_states)
        humanoid_eef_position = humanoid_states[3:6]
        infos['succeed'] = dones
        infos['rob_eef_position'] = ur5_eef_position
        infos['rob_eef_orient'] = ur5_eef_oriention
        infos['rob_joint_state'] = ur5_states[7:13]
        infos['human_eef_position'] = humanoid_eef_position
        infos['goal'] = self.goal
        infos['goal_ori'] = self.goal_orient
        infos['alternative_goals'] = self.box_pos[1:]


        # check human hand and elbow position
        self._p.addUserDebugLine(humanoid_states[-3:], humanoid_states[-6:-3], \
                                 lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=10)

        self._p.addUserDebugLine(self.last_human_eef, humanoid_eef_position, \
                                 lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=10)
        self._p.addUserDebugLine(self.last_robot_eef, ur5_eef_position, \
                                 lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=10)


        # if np.linalg.norm((ur5_eef_position - self.goal)) < 0.05 and \
        #        abs(utils.quaternion_diff(ur5_eef_oriention, self.goal_orient)) < 0.6:
        if np.linalg.norm((ur5_eef_position - self.goal)) < self.success_range:
            dones = True
        else:
            dones = False


        ############## change human links to robot base ##############
        # human_obs_origin = humanoid_states.copy() # max distance between human & robot; 1m
        # human_obs_origin[0:3] = self.trans_points(human_obs_origin[0:3])
        # human_obs_origin[3:6] = self.trans_points(human_obs_origin[3:6])

        #############################################################



        self.robot_current_p = ((ur5_eef_position), (ur5_eef_oriention))

        # human to robot base minimum distance

        pts = self._p.getClosestPoints(self.agents[0].robot_body.bodies[0],
                                       self.agents[1].robot_body.bodies[0], self.safe_dist_threshold)
        min_dist = self.safe_dist_threshold
        for i in range(len(pts)):
            if pts[i][8] < min_dist:
                min_dist = pts[i][8]
        if len(pts) == 0:
            # normalize
            obs_human_states = np.ones(len(humanoid_states))
            # obs_human_states[0:6] = humanoid_states[0:6] / np.linalg.norm(humanoid_states[0:6]) * self.safe_dist_threshold
        else:
            # add reward according to min_distance
            obs_human_states = humanoid_states

        infos['min_dist'] = min_dist




        collisions = self._p.getContactPoints()
        collision_bodies = []
        for c in collisions:
            bodyinfo1 = self._p.getBodyInfo(c[1])
            bodyinfo2 = self._p.getBodyInfo(c[2])
            # print("bodyinfo1: ", bodyinfo1, "bodyinfo2: ", bodyinfo2)
            collision_bodies.append(bodyinfo1[1].decode("utf-8"))
            collision_bodies.append(bodyinfo2[1].decode("utf-8"))

        infos['collision_bodies'] = collision_bodies

        # #early stop
        # if len(collisions) > 0:
        #     dones = True
        # if "ur5" in collision_bodies:
        #     dones = True

        # ------------------calculate xyz distance to goals---------------
        # distance = [(ur5_eef_position - g) for g in self.box_pos]
        distance = ur5_eef_position-self.goal
        angle_distance = utils.quaternion_diff(ur5_eef_oriention, self.goal_orient)

        try:
            # # eef bc loss
            # dpath, progress = calculate_bc_loss(infos['rob_eef_position'], self.reference_traj)

            # joint angle bc loss
            dpath, progress = calculate_bc_loss(infos['rob_joint_state'], self.reference_traj)

        except:
            dpath = 0
            progress = 0

        self.dpath = dpath
        self.progress = progress

        ref_traj = self._get_next_ref_traj()
        infos['ref_traj'] = ref_traj

        obs = np.concatenate([np.asarray(ur5_states), np.asarray(obs_human_states),
                              np.asarray(self.goal).flatten(), np.asarray([min_dist]), np.asarray(ref_traj)])

        return dones, obs, infos

    def _get_next_ref_traj(self):
        # --- cartesian traj ---#
        step_len = 3
        dof = 6
        ref_traj = np.zeros(dof * step_len)

        try:
            start_index = int(self.last_progress * len(self.reference_traj))
        except:
            pass
        else:
            end_index = start_index + step_len
            if end_index >= len(self.reference_traj):
                last_traj = np.full((end_index - len(self.reference_traj), dof), (self.reference_traj[-1, :]))
                end_index = len(self.reference_traj)
                ref_traj[(end_index - start_index) * dof:] = last_traj.ravel()

            # print("start index is {} and end index is {} ".format(start_index, end_index))
            ref_traj[:(end_index - start_index) * dof] = (self.reference_traj[start_index:end_index, :]).ravel()


        return ref_traj



    def trans_points(self, p):
        # p_new = p-self.robot_base
        p_new = p.copy()
        # if np.linalg.norm(p_new) > 1.0:
        #     p_new = p_new/np.linalg.norm(p_new)
        # print("p_new,", p_new)
        return p_new


    def _comp_reward(self, infos):
        '''
        Computer collision and approach reward
        :return:
        '''
        # task reward
        success_r = 20
        collision_success_ratio = 0.3

        if infos['succeed'] == True:
            task_reward = success_r
        else:
            task_reward = 0
            # # #-----------approaching reward------------#
            current_distance = np.linalg.norm((infos['rob_eef_position'] - infos['goal']))
            last_distance = np.linalg.norm([self.last_robot_eef - infos['goal']])
            # current_angle = abs(utils.quaternion_diff(infos['rob_eef_orient'], infos['goal_ori']))
            # task_reward = - (1*current_distance + 1* current_angle)
            task_reward = last_distance-current_distance

        # -----collision reward------#
        reward_robot_col = 0
        collision_bodies = infos["collision_bodies"]
        if len(collision_bodies) != 0:
            if "ur5" in collision_bodies:  # robot collision
                reward_robot_col = -10
            if "ur5" and "human" in collision_bodies:
                reward_robot_col = -(success_r)*collision_success_ratio
        # print("reward_robot_col", reward_robot_col)

        #----- human safe distance reward---#
        safe_distance_reward = -(self.safe_dist_threshold - infos['min_dist'])

        #-------behavior clone reward------#
        #behavior distance
        target_point = infos['ref_traj'][-6:]
        bc_reward =  np.linalg.norm([self.last_robot_joint - target_point]) - np.linalg.norm([infos['rob_joint_state'] - target_point])


        reward = 0*reward_robot_col + 0.1*safe_distance_reward+ 30*bc_reward + 10*task_reward


        # print("dpath reward {} and progrss reward {}".format(-1*dpath,  10*(progress-self.last_progress)))

        # print("reward col {} and safe_distance {}".format(0*reward_robot_col, 0.1*safe_distance_reward))
        # print("bc reward {} and task_reward {}".format(30*bc_reward,10*task_reward ))

        # #----ratio bc loss---#
        # try:
        #     bc_ratio, progress = calculate_bc_ratio(infos['rob_eef_position'], self.reference_traj, ph=0.2, pl=0.01)
        # except:
        #     bc_ratio = 1
        #     progress = 0
        #
        #
        # self.last_progress = progress
        #
        # print(" current distance is {}, current angle is {} and bc_ratio is {} and progress is {}"\
        #       .format(current_distance, current_angle, bc_ratio, progress))
        # print("col reward is {}, task reward is {} and bc_reward is {}".format(reward_robot_col, task_reward, bc_ratio))
        #
        # reward = reward_robot_col + task_reward*bc_ratio

        return reward



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




class UR5HumanSharedEnv(UR5HumanCollisionEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 60}
    def __init__(self, render=False, max_episode_steps=1000):
        self.distance_close = 0.3

        self.iter_num = 0
        self.max_episode_steps = max_episode_steps

        self.scene = None
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self.isRender = render
        self.agents = [UR5RG2Robot(), SelfMoveAwayHumanoid(0, 12)]
        self._n_agents = 2
        self.seed()
        self._cam_dist = 3
        self._cam_yaw = 0
        self._cam_pitch = 30
        self._render_width = 320
        self._render_height = 240
        self.is_training = True

        # Set observation and action spaces
        self.agents_observation_space = Tuple([
            agent.observation_space for agent in self.agents
        ])
        self.agents_action_space = Tuple([
            agent.action_space for agent in self.agents
        ])

        obs_dim, act_dim = 0, 0
        for agent in self.agents:
            obs_dim += agent.observation_space.shape[0]
            act_dim += agent.action_space.shape[0]

        obs_dim+=4

        high = np.ones([act_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf * np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        # super(UR5HumanSharedEnv, self).__init__(render=render, max_episode_steps=max_episode_steps)
        # self.agents = [UR5RG2Robot(), SelfMoveAwayHumanoid(0, 12)]

    def _comp_reward(self, infos):
        '''
        Computer collision and approach reward
        :return:
        '''
        # success reward
        success_r = 200
        collision_success_ratio = 0.5

        if infos['succeed'] == True:
            reward = success_r
            return reward

        # reward_uncomfortable = 0
        #
        # pts = self._p.getClosestPoints(self.agents[0].robot_body.bodies[0], self.agents[1].robot_body.bodies[0], 0.15)
        # if len(pts) > 0:
        #     reward_uncomfortable = -2
        #     # reward_uncomfortable = - np.log(abs(1 / pts[0][8])+1e-10) * 0.5


        # collision reward
        reward_robot_col = 0
        collision_bodies = infos["collision_bodies"]
        if len(collision_bodies) != 0:
            if "ur5" in collision_bodies:  # robot collision
                reward_robot_col = -10
            if "ur5" and "human" in collision_bodies:
                reward_robot_col = -(success_r)*collision_success_ratio


        # approaching reward
        current_distance = np.linalg.norm((infos['rob_eef_position'] - infos['goal']))
        reward_approach = 50 * (self.last_distance - current_distance)
        self.last_distance = current_distance

        # angle approaching
        current_angle = abs(utils.quaternion_diff(infos['rob_eef_orient'], infos['goal_ori']))
        reward_angle = 10 * (self.last_angle - current_angle)
        self.last_angle = current_angle
        task_reward = reward_robot_col + reward_approach + reward_angle
        # task_reward = reward_robot_col  + reward_approach + reward_angle + reward_uncomfortable
        # task_reward = reward_robot_col + reward_approach
        # task_reward = reward_robot_col + reward_angle

        # legible_reward = 500*reward_utils.fast_decrease_distance_reward(obs, infos, self.iter_num)
        legible_reward = 0
        task_ratio = 1.0
        reward = task_ratio * task_reward + (1 - task_ratio) * legible_reward
        # print("approach reward: {} and angle reward: {} and legible reward {}".format(reward_approach, reward_angle, legible_reward))
        return reward
