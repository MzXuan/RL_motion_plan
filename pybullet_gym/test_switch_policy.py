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



def restore_tf_graph(sess, fpath):
    """
    Loads graphs saved by Logger.

    Will output a dictionary whose keys and values are from the 'inputs' 
    and 'outputs' dict you specified with logger.setup_tf_saver().

    Args:
        sess: A Tensorflow session.
        fpath: Filepath to save directory.

    Returns:
        A dictionary mapping from keys to tensors in the computation graph
        loaded from ``fpath``. 
    """
    tf.saved_model.loader.load(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                fpath
            )
    model_info = joblib.load(osp.join(fpath, 'model_info.pkl'))
    graph = tf.get_default_graph()
    model = dict()
    model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['inputs'].items()})
    model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['outputs'].items()})
    return model

def get_fname(fpath):
    saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x) > 8]
    itr = '%d' % max(saves) if len(saves) > 0 else ''
    fname = osp.join(fpath, 'tf1_save' + itr)
    print('\n\nLoading from %s.\n\n' % fname)
    return fname


def load_tf_policy(fpath1, fpath2, itr, deterministic=True):
    """ Load a tensorflow policy saved with Spinning Up Logger."""
    fname1 = get_fname(fpath1)
    fname2 = get_fname(fpath2)

    # load the things!
    # sess = tf.Session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    model1 = restore_tf_graph(sess, fname1)
    sess2 = tf.Session(config=config)
    model2 = restore_tf_graph(sess2, fname2)

    # # get the correct op for executing actions
    # if deterministic and 'mu' in model.keys():
    #     # 'deterministic' is only a valid option for SAC policies
    #     print('Using deterministic action op.')
    #     action_op = model['mu']
    # else:
    #     print('Using default action op.')
    #     action_op = model['pi']

    # make function for producing an action given a single state
    get_action_1 = lambda x : sess.run(model1['mu'], feed_dict={model1['x']: x[None,:]})[0]
    get_action_2 = lambda x: sess.run(model2['mu'], feed_dict={model2['x']: x[None, :]})[0]

    return get_action_1, get_action_2

# def save_set(dir, obs_lst, action_lst, id):
#     print("save data...")
#     filename = os.path.join(dir,"demo_"+str(id)+".pkl")
#     with open(filename, 'wb') as handle:
#         pickle.dump({'obs_lst': obs_lst, 'action_lst': action_lst}, \
#                     handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print("save dataset successfully at {}; data size is {}".format(filename, len(obs_lst)))

def main(fpath1, fpath2, env, itr, collect, test):
    seed = 50
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

    get_action_1, get_action_2 = load_tf_policy(fpath1, fpath2, itr)

    if collect is False:
        env.render(mode="human")

    #prepare collect buffer list
    obs_lst = []
    center_lst = []
    goal_label = []

    obs = env.reset()
    print("obs initial", obs)
    id=0
    if collect is False:
        env.render(mode="human")
    info={}
    info['human_robot_distance'] = 1
    ## start simulation loop ##
    while(id<300):
        time.sleep(0.03)

        print("closet distance:", info['human_robot_distance'])
        if info['human_robot_distance'] > 0.1:
            action = get_action_1(obs)
        if info['human_robot_distance']:
            action = get_action_2(obs)


        obs, rew, done, info = env.step(action)

        if done == True:
            #reset all
            time.sleep(1)
            env.reset()
            # save data
            if collect is True:
                cam_utils.save_set(id, obs_lst, center_lst, goal_label)

            #reset
            obs_lst = []
            center_lst = []
            goal_label = []
            id+=1
            seed+=1
            np.random.seed(seed)
            tf.set_random_seed(seed)
            random.seed(seed)

        if collect is False:
            env.render()
            # w, h = 112, 112
            # far =3
            # near =0.1
            # img_camera = env.get_camera_img(w, h)
            # depth_buffer_opengl = np.reshape(img_camera[3], [w, h])
            # depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)
            # cv2.imshow("img", depth_opengl/far)
            # cv2.waitKey(1)

        else:
            w, h = 112, 112
            img_camera = env.get_camera_img(w, h)
            img = np.asarray(img_camera[2]).reshape((w, h, 4))
            # find pixels with goal value in img_camera[4] if reset, than find once
            if center_lst == [] or goal_label == []:
                target_value = env.get_target_value()
                # print("target value", target_value)
                goal_label = target_value["goal_label"]
                for t_id in target_value["target_id"]:
                    center_lst.append(cam_utils.find_link_px(img_camera[4], t_id, w, h))
                c = list(zip(center_lst, goal_label))
                random.shuffle(c)
                center_lst, goal_label = zip(*c)
                center_lst = list(center_lst)
                goal_label = list(goal_label)
                print("center_ lst {} goal_label {}".format(center_lst, goal_label))
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            obs_lst.append(img)

            # # #for debug
            # # print("center is {}".format(c))
            # for idx, c in enumerate(center_lst):
            #     if goal_label[idx] == 1:
            #         img = cv2.circle(img, (c[0], c[1]), 5, (0, 0, 255), 2)
            #     else:
            #         img = cv2.circle(img, (c[0], c[1]), 5, (0, 255, 0), 2)
            # cv2.imshow("img", img)
            # cv2.waitKey(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath1', type=str, default=None)
    parser.add_argument('--fpath2', type=str, default=None)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument("--env", type=str, default="UR5HumanCollisionEnv-v0")
    # parser.add_argument("--env", type=str, default="UR5HumanSharedEnv-v0")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    main(args.fpath1, args.fpath2, args.env, args.itr, args.collect, args.test)

