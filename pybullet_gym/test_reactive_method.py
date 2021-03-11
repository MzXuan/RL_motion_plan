import os
import joblib
import pybullet
import os, glob
import time
import pybullet_ur5
import gym
import cv2
import cam_utils

import os.path as osp
import numpy as np
import tensorflow as tf
import random

import matplotlib.pyplot as plt
import pickle


# def get_action(obs, rob_eef_goal, human2robot, env):
#     print("rob_eef_goal: ", rob_eef_goal)
#     rob_goal = rob_eef_goal[:3]
#     current_eef= obs['observation'][:3]
#     # attract to goal:
#     # goal_v = (rob_goal-current_eef)/np.linalg.norm(rob_goal-current_eef)
#     goal_v = 2.5*(rob_goal - current_eef)
#     print("norm of goal v is: ", np.linalg.norm(goal_v))
#     if np.linalg.norm(goal_v) > 0.6:
#         goal_v = 0.6*goal_v/np.linalg.norm(goal_v)
#
#     #repulsive force from human:
#     human_v = []
#     thre = 0.4
#     for p in human2robot:
#         d = p
#         d_norm = np.linalg.norm(p)
#         if d_norm<thre:
#             # v =  (np.exp(1/(2*thre))-1)*d/d_norm
#             v = -3*d* np.exp((d_norm-thre)/thre)
#             human_v.append(v)
#     rob_v = np.array(goal_v)
#     #todo: support pose changing
#     for v in human_v:
#         rob_v += v
#
#     print("human_v", human_v)
#     print("rob_v", rob_v)
#     print("goal_v", goal_v)
#
#     target_jp = env.get_robot_ik(current_eef+rob_v)
#     current_jp = obs['observation'][3:9]
#
#     joint_v = (target_jp-current_jp)
#     print("joint_v", joint_v) #similar to action, max is 1
#     #todo: calculate ik and send to joint xxx
#     return joint_v


def get_action(obs, rob_joint_goal, human2robot, env):

    current_eef= obs['observation'][:3]
    current_joint = obs['observation'][3:9]
    # attract to goal:
    # goal_v = (rob_goal-current_eef)/np.linalg.norm(rob_goal-current_eef)
    goal_v = 5*(rob_joint_goal - current_joint)

    # print("norm of goal v is: ", np.linalg.norm(goal_v))
    if np.linalg.norm(goal_v) > 1.0:
        goal_v = 1.0*goal_v/np.linalg.norm(goal_v)

    #repulsive force from human:
    human_v = []
    thre = 0.4
    for p in human2robot:
        d = p
        d_norm = np.linalg.norm(p)
        if d_norm<thre:
            # v =  (np.exp(1/(2*thre))-1)*d/d_norm
            v = -3*d* np.exp((d_norm-thre)/thre)
            human_v.append(v)
    human_v_sum = 0
    for v in human_v:
        human_v_sum+=v



    # print("human_v", human_v)
    # print("rob_v", rob_v)
    # print("goal_v", goal_v)

    human_jp = env.get_robot_ik(current_eef+human_v_sum)
    current_jp = obs['observation'][3:9]

    human_delta_v = (human_jp - current_jp)
    rob_joint_v = goal_v+human_delta_v

    # for v in human_v:
    #     rob_v += v


    # print("joint_v", rob_joint_v) #similar to action, max is 1

    return rob_joint_v




def main(env):
    seed = np.random.randint(0,100)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)

    pybullet.connect(pybullet.DIRECT)
    env = gym.make(env)
    env.seed(seed)
    print(" ---------------------------- ")
    print(env.observation_space.shape)
    print(env.action_space.shape)
    print(" ---------------------------- ")
    env.render(mode="human")

    env.reset()
    env.draw_path()
    traj_count = 1

    env.set_sphere(0.15)

    time_lst= []
    collision_lst = []
    success_count = 0
    s=0
    traj_len_lst = []
    traj_len = 0

    success_steps = []
    ## start simulation loop ##
    obs, rew, done, info = env.step(np.array([0, 0, 0,0,0,0]))
    last_obs = obs
    while traj_count < 300:
        try:
            # time.sleep(0.03)
            # obs = env.get_obs()
            time.sleep(0.01)

            human2robot = env.last_obs_human
            # human2robot = [obs_human[2], obs_human[5]]
            # rob_eef_goal = env.eef_goal
            # action = get_action(obs, rob_eef_goal, human2robot,env)


            rob_joint_goal = env.goal
            start_time = time.time()
            action = get_action(obs, rob_joint_goal, human2robot, env)
            time_lst.append(time.time() - start_time)

            obs, rew, done, info = env.step(action)
            s += 1
            collision_lst.append(info['is_collision'])
            traj_len += np.linalg.norm(obs['observation'][:3] - last_obs['observation'][:3])
            last_obs = obs

            if done == True:

                if sum(collision_lst) <= 3 and info['is_success']:
                    success_count += 1
                    success_steps.append(s)
                    traj_len_lst.append(traj_len)

                print("current mean of traj len is: ", np.array(traj_len_lst).mean())
                print("current std of traj len is: ", np.array(traj_len_lst).std())
                print("current mean reach time is: ", np.array(time_lst).mean())
                print("current std of reach time is: ", np.array(time_lst).std())
                print("current success rate is: ", success_count / traj_count)
                print("current mean success steps is: ", np.array(success_steps).mean())
                print("current std success steps is: ", np.array(success_steps).std())
                s = 0
                traj_count += 1
                traj_len = 0
                collision_lst = []
                print("reset")
                env.agents[0].stop()
                env.reset()
                obs, rew, done, info = env.step(np.array([0, 0, 0, 0, 0, 0]))
                last_obs = obs
            env.render()

        except KeyboardInterrupt:
            print("current mean reach time is: ", np.array(time_lst).mean())
            print("current std of reach time is: ", np.array(time_lst).std())
            env.agents[0].stop()
            env.close()
            raise



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env", type=str, default="UR5HumanRealEnv-v0")
    parser.add_argument("--env", type=str, default="UR5HumanEnv-v0")
    args = parser.parse_args()
    main(args.env)

