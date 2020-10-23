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


def get_action(obs, info, hand_v):
    goal = obs["desired_goal"]
    ee_pos = obs["achieved_goal"]
    ee_vel = info['ee_vel'][:3]

    dist = info['min_dist']
    # human hand velocity - ee_vel / distance

    rob_max_v = 1.5
    scale = 5
    print("dist is:", dist)
    if dist < 0.35:
        print("hand!!!!")
        ee_next_v = (hand_v)/dist*scale
    else:
        ee_next_v = (goal - ee_pos)*scale


    print("vel range is: ", np.linalg.norm(ee_next_v))
    if np.linalg.norm(ee_next_v) >rob_max_v:
        ee_next_v =  ee_next_v/np.linalg.norm(ee_next_v)

    print("ee_next_v", ee_next_v)

    return ee_next_v

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


    obs = env.reset()
    # print("obs initial", obs)

    traj_count = 0
    env.render()

    # env.set_sphere(0.2)

    ## start simulation loop ##
    obs, rew, done, info = env.step([0, 0, 0])
    while traj_count < 300:
        try:
            time.sleep(0.03)
            # obs = env.get_obs()

            # hand_v = env.get_hand_velocity()
            # action = get_action(obs, info, hand_v)

            action = [0,0,0]
            # print("action: ", action)
            obs, rew, done, info = env.step(action)
            #

            # print("obs, ", obs)
            # print("reward: ", rew)
            # print("obs is: ", obs['observation'][7:13])
            # print("info is:", info)

            if done == True:
                #reset all
                traj_count += 1
                print("reset")
                env.reset()
                obs, rew, done, info = env.step([0, 0, 0])

            env.render()

        except KeyboardInterrupt:
            env.close()
            raise



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="UR5DynamicReachPlannerEnv-v0")
    args = parser.parse_args()
    main(args.env)

