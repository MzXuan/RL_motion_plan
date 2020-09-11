import numpy as np
import os
import mujoco_py

from gym.envs.robotics import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

MODEL_XML_PATH = os.path.join('fetch', 'reach_xz.xml')

INITIAL_QPOS = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }


class Moving_obstacle():
    def __init__(self, name):
        self.name = name
        self.moving_speed = np.asarray([0,0,0])

    def move_obstacle(self,sim):
        try:
            self.goal
        except:
            self.goal = self.random_obstacle_goal(sim)

        body_num = sim.model.body_name2id( self.name)
        obstable_center = sim.model.body_pos[body_num]
        if np.linalg.norm(obstable_center-self.goal)<0.01:
            self.goal = self.random_obstacle_goal(sim)
        self.moving_speed = 0.008*(self.goal-obstable_center)/np.linalg.norm(obstable_center-self.goal)
        sim.model.body_pos[body_num] = obstable_center+ self.moving_speed

    def random_obstacle_goal(self, sim):
        body_num = sim.model.body_name2id('table0')
        box_center = sim.model.body_pos[body_num]
        obs_goal = box_center.copy()
        # reset mocap to some place

        if np.random.choice([0, 1]) == 0:
            obs_goal[0] += np.random.choice([-0.15, 0.15])
            obs_goal[1] += np.random.uniform(-0.25, 0.25)
        else:
            obs_goal[0] += np.random.uniform(-0.15, 0.15)
            obs_goal[1] += np.random.choice([-0.25, 0.25])

        obs_goal[2] += np.random.uniform(0.25, 0.4)
        return obs_goal


class FetchReachV2Env(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path=MODEL_XML_PATH, n_substeps=20, gripper_extra_height=0.2, block_gripper=True,
        has_object=False, target_in_the_air=True, target_offset=0.0, obj_range=0.15, target_range=0.4,
        distance_threshold=0.05, initial_qpos=INITIAL_QPOS, reward_type='sparse', early_stop=False):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        self.early_stop = early_stop
        self.obstacle_name_list = ['obstacle0', 'obstacle1', 'obstacle2']
        self.obstacles_cls = [Moving_obstacle(obs_name) for obs_name in self.obstacle_name_list]

        super(FetchReachV2Env, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=3,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        force_reward = self._contact_force()
        a1 = -1
        a2 = -15
        reward = a1*(d > self.distance_threshold).astype(np.float32)+a2*force_reward
        return reward



    def step(self, action):
        self.time_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        for obstacle in self.obstacles_cls:
            obstacle.move_obstacle(self.sim)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()


        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'is_collision': self._contact_dection(),
            'alternative_goals': obs['observation'][-6:]
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

        if self.early_stop:
            if info["is_success"] or info["is_collision"]:
                done = True
        return obs, reward, done, info





    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (3,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl = action[:3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion

        action = np.concatenate([pos_ctrl, rot_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        # if self.has_object:
        #     object_pos = self.sim.data.get_site_xpos('object0')
        #     # rotations
        #     object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
        #     # velocities
        #     object_velp = self.sim.data.get_site_xvelp('object0') * dt
        #     object_velr = self.sim.data.get_site_xvelr('object0') * dt
        #     # gripper state
        #     object_rel_pos = object_pos - grip_pos
        #     object_velp -= grip_velp
        # else:
        #     object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obstable_center_lst = []
        obstacle_rel_pos_lst = []
        obstacle_v_lst = []

        for obstacle in self.obstacles_cls:
            body_num = self.sim.model.body_name2id(obstacle.name)
            obstable_center = self.sim.model.body_pos[body_num]
            obstable_center_lst.append(obstable_center)
            obstacle_rel_pos_lst.append(obstable_center - grip_pos)
            obstacle_v_lst.append(obstacle.moving_speed)

        achieved_goal = grip_pos.copy()

        obs = np.concatenate([
            grip_pos, np.asarray(obstable_center_lst).ravel(), np.asarray(obstacle_rel_pos_lst).ravel(), gripper_state,
            np.asarray(obstacle_v_lst).ravel(), grip_velp, gripper_vel,
        ])

        # obs = np.concatenate([
        #     grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
        #     object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        # ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _contact_force(self):
        for i in range(self.sim.data.ncon):
            # Note that the contact array has more than `ncon` entries,
            # so be careful to only read the valid entries.
            contact = self.sim.data.contact[i]
            # print('geom1', contact.geom1, self.sim.model.geom_id2name(contact.geom1))
            # print('geom2', contact.geom2, self.sim.model.geom_id2name(contact.geom2))
            if self.sim.model.geom_id2name(contact.geom1) is None or \
                    self.sim.model.geom_id2name(contact.geom2) is None:
                continue

            if "obstacle" in self.sim.model.geom_id2name(contact.geom1) and \
                "robot0" in self.sim.model.geom_id2name(contact.geom2):
                return 1.0

            elif "robot0" in self.sim.model.geom_id2name(contact.geom1) and \
                "obstacle" in self.sim.model.geom_id2name(contact.geom2):
                return 1.0

        return 0.0


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
            elif "obstacle" in self.sim.model.geom_id2name(contact.geom1) and \
                "obstacle" in self.sim.model.geom_id2name(contact.geom2):
                pass
            else:
                # print('geom1', contact.geom1, self.sim.model.geom_id2name(contact.geom1))
                # print('geom2', contact.geom2, self.sim.model.geom_id2name(contact.geom2))
                # print('--------------------------')
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



    def _reset_arm(self):
        collision_flag = True
        while collision_flag:
            body_num = self.sim.model.body_name2id('table0')
            box_center = self.sim.model.body_pos[body_num]
            grip_pos = box_center.copy()
            # reset mocap to some place
            grip_pos[0] += np.random.uniform(-0.1, 0.1)
            grip_pos[1] += np.random.uniform(-0.15, 0.15)
            grip_pos[2] += np.random.uniform(0.25, 0.41)

            # gripper_rotation = np.array([1., 1., 0., 0.])
            gripper_rotation = np.array([1., 0., 1., 0.])

            self.sim.data.set_mocap_pos('robot0:mocap', grip_pos)
            self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
            for _ in range(10):
                self.sim.step()
            collision_flag = self._contact_dection()
            # collision_flag = False


    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        self._reset_arm()

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            body_num = self.sim.model.body_name2id('table0')
            box_center = self.sim.model.body_pos[body_num]
            # reset mocap to some place
            goal = np.zeros(3)
            goal[0] = box_center[0] + np.random.uniform(-0.2, 0.2)
            goal[1] = box_center[1] + np.random.uniform(-0.25, 0.25)
            goal[2] = box_center[2] + np.random.uniform(0.25, 0.41)
        return goal.copy()


    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(FetchReachV2Env, self).render(mode, width, height)
