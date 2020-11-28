import os, glob
import pybullet_ur5
import gym
import pybullet

import numpy as np
import tensorflow as tf
import random
from scipy.interpolate import CubicSpline


import time
import threading



from pybullet_planning import link_from_name, get_moving_links, get_link_name
from pybullet_planning import get_joint_names, get_movable_joints
from pybullet_planning import multiply, get_collision_fn
from pybullet_planning import inverse_kinematics, sample_tool_ik
from pybullet_planning import set_joint_positions, wait_for_duration
from pybullet_planning import get_joint_positions, plan_waypoints_joint_motion, plan_joint_motion, compute_forward_kinematics
from pybullet_planning.motion_planners.stomp import STOMP

import ikfast_ur5
import pyquaternion



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


class UR5Planner():
    def __init__(self, robot, workspace, ik_joints, env):
        self.robot = robot
        self.env = env
        self.clientid = env.physicsClientId
        self.ik_joints = ik_joints
        ee_link_name = self.env.agents[0].ee_link
        self.tool_link = link_from_name(self.robot, ee_link_name)
        # self.select_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", \
        #                       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.ur5_collision_fn(robot, ik_joints, workspace)
        self.reset_stomp()
        self.steps_count =[]



    def reset_stomp(self, n=40 ):
        D = 6
        self.N = n
        dt = 0.05
        max_joint_v = 0.5
        K = 10
        self.stomp = STOMP(D, self.N, K, dt, max_joint_v, collision_fn=self.collision_fn, ik_fn=self.calculate_ur5_ik)


    def cubic_spline(self, path, n):
        '''
        interpolate line longer than 40 time steps to 40 steps
        '''
        # path_length = np.zeros(self.D)
        # for i in range(self.N-1):
        #     path_length += abs(path[i+1,:]-path[i,:])
        # min_t =np.max(path_length/self.max_joint_v)

        total_steps = len(path)
        x = np.linspace(0,total_steps,total_steps)
        cs = CubicSpline(x, path)
        xs = np.linspace(0,total_steps, n)
        splined_path = cs(xs)
        return splined_path


    def calculate_ur5_ik(self, p):
        #-------------bullet ik------------------------------

        robot_base_link_name = 'base_link'
        ori = [0, 0.841471, 0, 0.5403023]
        tool_link = self.tool_link

        pb_q = pybullet.calculateInverseKinematics(self.robot, tool_link,
                                                        p, ori, physicsClientId = self.clientid
                                                        )
        n_conf_list = [normalize_conf(np.asarray([0, 0, 0, 0, 0, 0]), conf) for conf in [pb_q]]
        conf = n_conf_list[0]
        return conf


    def get_eef_from_conf(self, q):
        for i in range(6):
            pybullet.resetJointState(self.robot, self.ik_joints[i], q[i], 0, self.clientid)
        pos = pybullet.getLinkState(self.robot, self.tool_link, physicsClientId = self.clientid)[0]
        return pos


    def get_current_conf(self):
        jointstates = pybullet.getJointStates(self.robot, self.ik_joints, physicsClientId=self.clientid)
        joint_pos = [js[0] for js in jointstates]
        return joint_pos

    def set_conf(self,q):
        for i in range(6):
            pybullet.resetJointState(self.robot, self.ik_joints[i], q[i], 0, self.clientid)


    def stomp_planning(self, initial_conf, end_conf, initial_trajectory=None):
        #---configuration space result---#

        if initial_trajectory is None:
            self.reset_stomp()
        else:
            if len(initial_trajectory)>40:
                initial_trajectory = self.cubic_spline(initial_trajectory, 40)
                #todo: spline
            else:
                self.reset_stomp(len(initial_trajectory))
        conf_result, step = self.stomp.plan(initial_conf, end_conf, initial_trajectory)
        if step >1:
            self.steps_count.append(step-1)
        print("steps count", self.steps_count)

        #-----for visual debug----
        ca_result = [self.get_eef_from_conf(q) for q in conf_result]
        for i in range(len(ca_result)-1):
            pybullet.addUserDebugLine(ca_result[i] , ca_result[i+1], physicsClientId=self.clientid)

        #------reset to beginning----
        self.set_conf(initial_conf)
        return conf_result


    def ur5_collision_fn(self, robot, ik_joints, workspace):
        self.collision_fn = get_collision_fn(robot, ik_joints, obstacles=[workspace],
                                        attachments=[], self_collisions=True,
                                        #    disabled_collisions=disabled_collisions,
                                        #    extra_disabled_collisions=extra_disabled_collisions,
                                        custom_limits={})



class UR5Mover:
    def __init__(self, robot, ik_joints, env):
        self.robot = robot
        self.env = env
        self.clientid = env.physicsClientId
        self.ik_joints = ik_joints
        ee_link_name = self.env.agents[0].ee_link
        self.tool_link = link_from_name(self.robot, ee_link_name)
        self.n=0
        self.dt = 0.2

    def reset_conf_traj(self, conf_traj):
        self.n = 0
        self.conf_traj = conf_traj
        # self.move_step(cancel=False)

    # def move(self, conf_traj):
    #     self.conf_traj = conf_traj
    #
    #     self.move_step()

        # for i in range(6):
        #     pybullet.setJointMotorControl2(self.robot, self.ik_joints[i], pybullet.POSITION_CONTROL,
        #                             target_p[i], target_v[i], positionGain=1, velocityGain=0.5)
        # scheduler = sched.scheduler(time.time, time.sleep)
        # self.schedule_it(scheduler, 1/self.dt, 10, self.move_step, conf_traj)
        # scheduler.run()

        # self.schedule_it(1 / self.dt, 10, self.move_step, conf_traj)


    def move_step(self, cancel=False):
        # if cancel:
        #     print("cancel....")
        #     # self.move_timer.cancel()
        #     return

        print("self.n is", self.n)

        conf_traj = self.conf_traj

        # print("length of conf is: ", len(conf_traj))
        if self.n>=len(conf_traj)-1:
            print("cancel due to max n ")
            # self.move_timer.cancel()
            return
        target_p = conf_traj[self.n,:]
        target_v = (conf_traj[self.n+1,:] - conf_traj[self.n,:])/self.dt

        print("target v: ", target_v)
        for i in range(6):
            pybullet.setJointMotorControl2(self.robot, self.ik_joints[i], pybullet.POSITION_CONTROL,
                                    target_p[i], target_v[i],
                                           positionGain=1, velocityGain=0.5, maxVelocity=1.5, physicsClientId=self.clientid)
        pybullet.stepSimulation(physicsClientId=self.clientid)
        self.n+=1


        self.move_timer = threading.Timer(self.dt, self.move_step).start()

        # print("target P", target_p)
        # print("end conf is", conf_traj[-1,:])


def move_human(env):
    env.agents[1].apply_action(0)



def main(env, test):
    seed = np.random.randint(1,200)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)

    pybullet.connect(pybullet.DIRECT)
    env = gym.make(env)
    env.set_training(not test)
    env.seed(seed)
    print(" ---------------------------- ")
    print(env.observation_space.shape)
    print(env.action_space.shape)
    print(" ---------------------------- ")

    get_action = lambda obs: [0.1, 0.1, 0]
    env.render(mode="human")
    obs = env.reset()
    initial_ref_path = env.reference_path


    #---------prepare planning------------------
    # ik_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    #             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    ik_joints = [1, 2, 3, 4, 5, 6]

    ur5_planner = UR5Planner(robot = env.agents[0].robot_body.bodies[0], workspace = env.agents[1].robot_body.bodies[0],
                             ik_joints = ik_joints ,env=env)
    ur5_mover = UR5Mover(robot = env.agents[0].robot_body.bodies[0], ik_joints = ik_joints ,env=env)

    def update_plan(initial_conf, end_conf, initial_trajectory):
        conf_result = ur5_planner.stomp_planning(initial_conf=initial_conf, end_conf=end_conf, initial_trajectory=initial_trajectory)
        ur5_mover.reset_conf_traj(conf_result)
        print("conf result, ", conf_result)
        ur5_mover.move_step()



    #-------------start planning------------------------------

    conf_result = ur5_planner.stomp_planning(initial_conf = initial_ref_path[0], end_conf = initial_ref_path[-1], initial_trajectory=initial_ref_path)

    #debug

    #-----start moving------
    print("length of conf result", len(conf_result))
    ur5_mover.reset_conf_traj(conf_result)
    ur5_mover.move_step()


    #--------joint planning-----------#

    # dt = 0.2
    # replan_t = 2
    # # moving and replanning
    # for _ in range(50000):
    #     try:
    #         for n in range(int(replan_t / dt)):
    #             move_human(env)
    #             pybullet.stepSimulation(physicsClientId=env.physicsClientId)
    #             time.sleep(dt)
    #
    #         initial_conf = ur5_planner.get_current_conf()
    #         if np.linalg.norm(np.array(initial_conf) - np.array(end_conf)) < 0.1:
    #             print("env reset")
    #             obs = env.reset()
    #             initial_conf = ur5_planner.calculate_ur5_ik(obs['achieved_goal'])
    #             end_conf = ur5_planner.calculate_ur5_ik(obs['desired_goal'])
    #             t1 = threading.Thread(target=update_plan, args=[initial_conf, end_conf, None])
    #             t1.start()
    #         else:
    #             result = ur5_mover.conf_traj - initial_conf
    #             error = np.linalg.norm(result, axis=1)
    #             id_min = np.argmin(error)
    #             ref_traj = ur5_mover.conf_traj[id_min:, :]
    #             print("ref traj shape is: ", ref_traj.shape)
    #             if len(ref_traj) <= 5:
    #                 pass
    #
    #             else:
    #                 t1 = threading.Thread(target=update_plan, args=[initial_conf, end_conf, ref_traj])
    #                 t1.start()
    #
    #     except KeyboardInterrupt:
    #         print(ur5_planner.steps_count)
    #         return

    # #------------ cartesian start and end planning---------#
    # dt = 0.2
    # replan_t = 2
    #
    # for _ in range(50000):
    #     try:
    #         for n in range(int(replan_t/dt)):
    #             move_human(env)
    #             pybullet.stepSimulation(physicsClientId=env.physicsClientId)
    #             time.sleep(dt)
    #
    #         initial_conf = ur5_planner.get_current_conf()
    #         if np.linalg.norm(np.array(initial_conf)-np.array(end_conf)) < 0.1:
    #             print("env reset")
    #             obs = env.reset()
    #             initial_conf = ur5_planner.calculate_ur5_ik(obs['achieved_goal'])
    #             end_conf = ur5_planner.calculate_ur5_ik(obs['desired_goal'])
    #             t1 = threading.Thread(target=update_plan, args=[initial_conf, end_conf, None])
    #             t1.start()
    #         else:
    #             result = ur5_mover.conf_traj - initial_conf
    #             error = np.linalg.norm(result, axis=1)
    #             id_min = np.argmin(error)
    #             ref_traj = ur5_mover.conf_traj[id_min:,:]
    #             print("ref traj shape is: ", ref_traj.shape)
    #             if len(ref_traj) <=5:
    #                 pass
    #
    #             else:
    #                 t1 = threading.Thread(target=update_plan, args=[initial_conf, end_conf, ref_traj])
    #                 t1.start()
    #
    #     except KeyboardInterrupt:
    #         print(ur5_planner.steps_count)
    #         return
    #




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="UR5HumanPlanEnv-v0")

    # parser.add_argument("--env", type=str, default="UR5DynamicReachPlannerEnv-v0")

    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    main(args.env, args.test)