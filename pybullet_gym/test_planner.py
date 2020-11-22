import os
import joblib
import pybullet
import os, glob
import time
import pybullet_ur5
import gym

import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import pickle
from termcolor import cprint


from pybullet_planning import link_from_name, get_moving_links, get_link_name
from pybullet_planning import get_joint_names, get_movable_joints
from pybullet_planning import multiply, get_collision_fn
from pybullet_planning import inverse_kinematics, sample_tool_ik
from pybullet_planning import set_joint_positions, wait_for_duration
from pybullet_planning import get_joint_positions, plan_waypoints_joint_motion, plan_joint_motion, compute_forward_kinematics
from pybullet_planning.motion_planners import stomp

import ikfast_ur5



def test_moving_links_joints(viewer, env):
    # connect(use_gui=viewer)
    # sim_id = connect(use_gui=False)
    client_id = env.physicsClientId
    print("client id should be: ", client_id)
    robot = env.agents[0].robot_id
    workspace = env.agents[1].objects[0]
    print("robot id is {} and workspace id is: {} ". format(robot, workspace))

    assert isinstance(robot, int) and isinstance(workspace, int)


    movable_joints = get_movable_joints(robot)
    assert isinstance(movable_joints, list) and all([isinstance(mj, int) for mj in movable_joints])
    assert 6 == len(movable_joints)
    assert [b'shoulder_pan_joint', b'shoulder_lift_joint', b'elbow_joint', b'wrist_1_joint', b'wrist_2_joint',
            b'wrist_3_joint'] == \
           get_joint_names(robot, movable_joints)

    moving_links = get_moving_links(robot, movable_joints)
    assert isinstance(moving_links, list) and all([isinstance(ml, int) for ml in moving_links])
    assert 10== len(moving_links)
    link_names = [get_link_name(robot, link) for link in moving_links]
    assert ['shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link',
            'ee_link', 'rg2_body_link', 'rg2_eef_link', 'tool0'] == \
           link_names


    return robot, workspace, movable_joints


def test_ur5_ik(robot, workspace, ik_joints, env):

    ee_link_name = env.agents[0].ee_link
    robot_base_link_name = 'base_link'

    tool_link = link_from_name(robot, ee_link_name)
    robot_base_link = link_from_name(robot, robot_base_link_name)


    ik_fn = ikfast_ur5.get_ik
    fk_fn = ikfast_ur5.get_fk

    # we have to specify ik fn wrapper and feed it into pychoreo
    def get_sample_ik_fn(robot, ik_fn, robot_base_link, ik_joints, tool_from_root=None):
        def sample_ik_fn(world_from_tcp):
            if tool_from_root:
                world_from_tcp = multiply(world_from_tcp, tool_from_root)
            return sample_tool_ik(ik_fn, robot, ik_joints, world_from_tcp, robot_base_link, get_all=True)

        return sample_ik_fn

    def get_bullet_ik_fn(robot, tool_link):
        def bullet_ik_fn(p):
            return inverse_kinematics(robot, tool_link, p)
        return bullet_ik_fn


    # #bullet ik#
    # bullet_ik_fn = get_bullet_ik_fn(robot, ik_fn)
    # p = (env.goal, None)
    # p_current = env.robot_current_p
    # # p = (env.goal, env.goal_orient)
    # p = (env.goal, None)
    # initial_conf = get_joint_positions(robot, ik_joints)
    #
    # print("p start is {} and p end is {}  ".format(p_current, p))
    #
    # pb_q = inverse_kinematics(robot, tool_link, p)
    # if pb_q is None:
    #     cprint('pb ik can\'t find an ik solution', 'red')

    collision_fn = get_collision_fn(robot, ik_joints, obstacles=[workspace],
                                    attachments=[], self_collisions=True,
                                    #    disabled_collisions=disabled_collisions,
                                    #    extra_disabled_collisions=extra_disabled_collisions,
                                    custom_limits={})
    # Let's check if our ik sampler is working properly
    sample_ik_fn = get_sample_ik_fn(robot, ik_fn, robot_base_link, ik_joints)
    p_current = env.robot_current_p
    p = (env.goal, env.goal_orient)

    initial_conf = get_joint_positions(robot, ik_joints)

    print("p start is {} and p end is {}  ".format(p_current, p))

    # print('-' * 5)
    # initial_qs = sample_ik_fn(p_current)
    # print("initial_qs: ", initial_qs)
    # if collision_fn is not None:
    #     initial_conf_list = [conf for conf in initial_qs if conf and not collision_fn(conf, diagnosis=False)]
    # initial_conf = initial_conf_list[0]

    qs = sample_ik_fn(p)
    print(qs)

    if qs is not None:
        cprint('But Ikfast does find one! {}'.format(qs[0]), 'green')
    else:
        cprint('ikfast can\'t find an ik solution', 'red')

    # we ignore self collision in this tutorial, the collision_fn only considers joint limit now
    # See : https://github.com/yijiangh/pybullet_planning/blob/dev/tests/test_collisions.py
    # for more info on examples on using collision function

    if collision_fn is not None:
        conf_list = [conf for conf in qs if conf and not collision_fn(conf, diagnosis=False)]


    set_joint_positions(robot, ik_joints, initial_conf)

    n_conf_list= [normalize_conf(np.asarray(initial_conf),conf) for conf in conf_list]

    final_conf = min_dist_conf(initial_conf, n_conf_list)

    print("start_cont is {} and final_conf is {}: ".format(initial_conf, final_conf))
    path = plan_joint_motion(robot, ik_joints, final_conf, obstacles=[workspace],
                             self_collisions=True, diagnosis=True)

    # for final_conf in n_conf_list:
    #     print("start_cont is {} and final_conf is {}: ".format(initial_conf, final_conf))
    #     path = plan_joint_motion(robot, ik_joints, final_conf, obstacles=[workspace],
    #                              self_collisions=True, diagnosis=True)

    # path = plan_waypoints_joint_motion(robot, ik_joints, [final_conf], obstacles=[workspace],
    #                          self_collisions=True)
    if path is None:
        print('se3 planning fails!')
        return

    # print("path is: ", path)
    time_step = 0.1


    cartesion_path = [compute_forward_kinematics(fk_fn, conf)[0] for conf in path]
    print("cartesion path: ", cartesion_path)


    for i, conf in enumerate(path):

        env.render(mode="human")
        cprint('conf: {}'.format(conf))
        set_joint_positions(robot, ik_joints, conf)
        wait_for_duration(time_step)

    ##---------test cartesian motion planner-------##
    # ee_poses=[p_current, p]
    # path, cost = plan_cartesian_motion_lg(robot, ik_joints, ee_poses, sample_ik_fn, collision_fn)
    # # path, cost = plan_cartesian_motion_lg(robot, ik_joints, ee_poses, bullet_ik_fn, collision_fn)
    # print("path is: ", path)
    #
    # if path is None:
    #     cprint('ladder graph (w/o releasing dof) cartesian planning cannot find a plan!', 'red')
    # else:
    #     cprint('ladder graph (w/o releasing dof) cartesian planning find a plan!', 'green')
    #     cprint('Cost: {}'.format(cost), 'yellow')
    #     time_step =5
    #     for conf in path:
    #         env.render(mode="human")
    #         cprint('conf: {}'.format(conf))
    #         set_joint_positions(robot, ik_joints, conf)
    #         wait_for_duration(time_step)

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

def stomp_planning(initial, end):
    return 0


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
    #todo: start planning
    stomp_planning(obs['achieved_goal'], obs['desired_goal'])

    time.sleep(10)




    # robot, workspace, movable_joints = test_moving_links_joints(False, env)

    # test_ur5_ik(robot, workspace,movable_joints, env)
    # obs = env.reset()
    # print("obs initial", obs)
    # id=0
    #
    # env.render()
    # ## start simulation loop ##
    #
    # obs = env.get_obs()
    # while(id<300):
    #     try:
    #         time.sleep(0.1)
    #         action = get_action(obs)
    #         obs, rew, done, info = env.step(action)
    #
    #         if done == True:
    #             #reset all
    #             print("reset")
    #             env.reset()
    #             id+=1
    #             seed+=1
    #             np.random.seed(seed)
    #             tf.set_random_seed(seed)
    #             random.seed(seed)
    #
    #         env.render()
    #
    #     except KeyboardInterrupt:
    #         env.close()
    #         raise




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env", type=str, default="UR5HumanCollisionEnv-v0")
    # parser.add_argument("--env", type=str, default="PyUR5ReachEnv-v2")
    # parser.add_argument("--env", type=str, default="UR5HumanSharedEnv-v0")
    parser.add_argument("--env", type=str, default="UR5DynamicReachPlannerEnv-v0")

    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    main(args.env, args.test)