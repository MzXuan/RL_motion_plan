import os
import joblib
import os, glob
import time

import gym
import gym_rlmp
import cv2


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

def load_tf_policy(fpath, itr, deterministic=True):
    """ Load a tensorflow policy saved with Spinning Up Logger."""
    saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]
    itr = '%d'%max(saves) if len(saves) > 0 else ''
    fname = osp.join(fpath, 'tf1_save'+itr)
    print('\n\nLoading from %s.\n\n'%fname)

    # load the things!
    # sess = tf.Session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    model = restore_tf_graph(sess, fname)

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    return get_action

# def save_set(dir, obs_lst, action_lst, id):
#     print("save data...")
#     filename = os.path.join(dir,"demo_"+str(id)+".pkl")
#     with open(filename, 'wb') as handle:
#         pickle.dump({'obs_lst': obs_lst, 'action_lst': action_lst}, \
#                     handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print("save dataset successfully at {}; data size is {}".format(filename, len(obs_lst)))



def main(fpath, env, itr, collect):
    seed = 50
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)

    env = gym.make(env)
    env = gym.wrappers.FlattenObservation(env)
    
    numItr = 100
    initStateSpace = "random"
    env.reset()
    env.render(mode="human")

    print("Reset!")
    actions = []
    observations = []
    infos = []

    if fpath is None:
        # get_action=lambda obs: [-0.1, 0.1, 0.1, 0, 0, 0]
        get_action = lambda obs: 0.6*env.action_space.sample()
    else:
        get_action = load_tf_policy(fpath, itr)

    while len(actions) < numItr:
        obs = env.reset()
        print("ITERATION NUMBER ", len(actions))
        episodeAcs, episodeInfo, episodeObs = goToGoal(env, obs, get_action)
        
        actions.append(episodeAcs)
        observations.append(episodeObs)
        infos.append(episodeInfo)


    fileName = "data_fetch"
    fileName += "_" + initStateSpace
    fileName += "_" + str(numItr)
    fileName += ".npz"

    np.savez_compressed(fileName, acs=actions, obs=observations, info=infos) # save the file


def goToGoal(env, obs, get_action):
    episodeAcs = []
    episodeObs = []
    episodeInfo = []

    done = False
    while not done:
        action = get_action(obs)
        obs, rew, done, info = env.step(action)

        time.sleep(0.03)
        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obs)
        env.render()
        

    return episodeAcs, episodeInfo, episodeObs



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default=None)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument("--env", type=str, default="FetchMotionPlan-v0")
    parser.add_argument("--collect", action="store_true")

    args = parser.parse_args()
    main(args.fpath, args.env, args.itr, args.collect)
