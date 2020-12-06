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

import pickle


def load_traj():
    try:
        with open('/home/xuan/demos/plan_joint_3.pkl', 'rb') as handle:
            data = pickle.load(handle)
        print("load data successfully")
    except:
        print("fail to load data, stop")
        return

    print("length of data is: ", len(data))
    return data

def get_action(obs, rob_joint_goal, human2robot, env):

    current_eef= obs['observation'][:3]
    current_joint = obs['observation'][3:9]
    # attract to goal:
    # goal_v = (rob_goal-current_eef)/np.linalg.norm(rob_goal-current_eef)
    goal_v = 5*(rob_joint_goal - current_joint)



    print("norm of goal v is: ", np.linalg.norm(goal_v))
    if np.linalg.norm(goal_v) > 2:
        goal_v = 2.0*goal_v/np.linalg.norm(goal_v)

    # #repulsive force from human:
    # human_v = []
    # thre = 0.4
    # for p in human2robot:
    #     d = p
    #     d_norm = np.linalg.norm(p)
    #     if d_norm<thre:
    #         # v =  (np.exp(1/(2*thre))-1)*d/d_norm
    #         v = -3*d* np.exp((d_norm-thre)/thre)
    #         human_v.append(v)
    # human_v_sum = 0
    # for v in human_v:
    #     human_v_sum+=v
    # print("human_v", human_v)
    # # print("rob_v", rob_v)
    # print("goal_v", goal_v)
    # human_jp = env.get_robot_ik(current_eef+human_v_sum)
    # current_jp = obs['observation'][3:9]
    #
    # human_delta_v = (human_jp - current_jp)
    # rob_joint_v = goal_v+human_delta_v

    rob_joint_v= goal_v
    # for v in human_v:
    #     rob_v += v


    print("joint_v", rob_joint_v) #similar to action, max is 1

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
    traj_count = 0

    env.set_sphere(0.15)


    ref_joint_traj = load_traj()
    del ref_joint_traj[50:90]
    del ref_joint_traj[100:180]
    ## start simulation loop ##
    obs, rew, done, info = env.step(np.array([0, 0, 0,0,0,0]))
    waypoint_id = 0
    while traj_count < 300:
        try:
            if waypoint_id > len(ref_joint_traj)-1:
                waypoint_id = len(ref_joint_traj)-1
            time.sleep(0.06)
            # obs = env.get_obs()
            print("waypointid: ", waypoint_id)


            if waypoint_id == 50:
                env.agents[0].stop()
                time.sleep(0.5)
            human2robot = env.last_obs_human
            rob_joint_goal = ref_joint_traj[waypoint_id]['robjp']
            if waypoint_id >50 and waypoint_id<80:
                rob_joint_goal[1]  =rob_joint_goal[1]-0.15
                rob_joint_goal[2] = rob_joint_goal[2] - 0.2
            action = get_action(obs, rob_joint_goal, human2robot, env)
            waypoint_id+=1

            obs, rew, done, info = env.step(action)


            if done == True:

                waypoint_id=0
                #reset all
                traj_count += 1
                print("reset")
                env.agents[0].stop()
                time.sleep(5)
                env.reset()

                obs, rew, done, info = env.step(np.array([0, 0, 0, 0, 0, 0]))

            env.render()

        except KeyboardInterrupt:
            env.agents[0].stop()
            env.close()
            raise



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="UR5HumanRealEnv-v0")
    # parser.add_argument("--env", type=str, default="UR5HumanEnv-v0")
    args = parser.parse_args()
    main(args.env)

