import os
import joblib
import pybullet
import os, glob
import time
import pybullet_ur5
import gym

import os.path as osp
import numpy as np
import tensorflow as tf
import random

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

def load_tf_policy(sess, fpath, itr, deterministic=True):
    """ Load a tensorflow policy saved with Spinning Up Logger."""
    saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]
    itr = '%d'%max(saves) if len(saves) > 0 else ''
    fname = osp.join(fpath, 'tf1_save'+itr)
    print('\n\nLoading from %s.\n\n'%fname)

    # load the things!
    # sess = tf.Session()

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

def save_set(dir, obs_lst, action_lst, id):
    print("save data...")
    filename = os.path.join(dir,"demo_"+str(id)+".pkl")
    with open(filename, 'wb') as handle:
        pickle.dump({'obs_lst': obs_lst, 'action_lst': action_lst}, \
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("save dataset successfully at {}; data size is {}".format(filename, len(obs_lst)))

def main(robot_path, human_path, env, itr, collect):
    seed = 50
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)

    pybullet.connect(pybullet.DIRECT)
    # env = gym.make("PyFetchReachEnv-v0")
    # env = gym.make("PyReacherEnv-v0")
    # env = gym.make("PyUR5ReachEnv-v0")
    # env = gym.make("HumanoidReachEnv-v0")
    env = gym.make(env)
    env.seed(seed)
    print(" ---------------------------- ")
    print(env.observation_space.shape)
    print(env.action_space.shape)
    print(" ---------------------------- ")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    get_robot_action = load_tf_policy(sess, robot_path, itr)
    get_human_action = load_tf_policy(sess, human_path, itr)

    env.render(mode="human")

    #prepare collect buffer list
    obs_lst = []
    action_lst = []
    obs = env.reset()
    print("obs initial", obs)
    id=0
    env.camera_adjust()
    env.render(mode="human")

    # time.sleep((3))

    # remove all previous set if collect is true
    if collect is True:
        dir = os.path.join(fpath,"demo_sac_s"+fpath.split("sac_s")[-1])
        if not os.path.exists(dir):
            os.mkdir(dir)

        filelist = glob.glob(os.path.join(dir, "data_*.pkl"))
        print("removed file list {}".format(filelist))
        for f in filelist:
            os.remove(f)


    while(id<100):
        time.sleep(0.03)
        robot_a = get_robot_action(obs)
        human_a = get_human_action(obs)
        action = np.concatenate([robot_a, human_a], axis=-1)
        if collect is True:
            #save obs and actions
            obs_lst.append(obs)
            action_lst.append(action)

        obs, rew, done, info = env.step(action)
        if done == True:
            env.reset()
            # print("reset")
            if collect is True:
                save_set(dir, obs_lst, action_lst, id)
                obs_lst = []
                action_lst = []
            id+=1

            # time.sleep(10)
        env.render()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('robot_path', type=str)
    parser.add_argument('human_path', type=str)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument("--env", type=str, default="UR5HumanHandoverEnv-v0")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    main(args.robot_path, args.human_path, args.env, args.itr, args.collect)

