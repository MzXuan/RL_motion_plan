import os
import joblib
import pybullet
import os, glob
import time
import pybullet_ur5
import gym
import copy

import os.path as osp
import numpy as np
import tensorflow as tf
import random

import pickle
from spinup.algos.tf1.safe_rl.core import get_vars


def save_network(sess, save_path):
    params = get_vars("pred")
    ps = sess.run(params)
    # print(ps)
    joblib.dump(ps, save_path)

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

def load_normal_policy(sess, fpath, itr, deterministic=True):
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

def load_safe_policy(sess, fpath, itr, deterministic=True):
    """ Load a tensorflow policy saved with Spinning Up Logger."""
    saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]
    itr = '%d'%max(saves) if len(saves) > 0 else ''
    fname = osp.join(fpath, 'tf1_save'+itr)
    print('\n\nLoading from %s.\n\n'%fname)

    # load the things!
    # sess = tf.Session()

    model = restore_tf_graph(sess, fname)

    # save_network(sess, "pred_model")

    class GetAction(object):
        def __init__(self, sess, model, pred_dim, max_timestep, deterministic=True):
            self.sess = sess
            self.model = model
            self.pred_dim = pred_dim
            self.max_timestep = max_timestep

            self.x_ph = model['x']
            self.x_seq_ph = model['x_seq']
            self.seq_len_ph = model['seq_len']

            self.pred_x = model['pred_x']
            self.uncer = model['uncer']
            
            if deterministic and 'mu' in model.keys():
                # 'deterministic' is only a valid option for SAC policies
                print('Using deterministic action op.')
                self.action_op = model['mu']
            else:
                print('Using default action op.')
                self.action_op = model['pi']

            self.reset()

        def reset(self):
            self.o_seq = []

        def compute_mixed_o(self, o):
            o = copy.deepcopy(o)
            # pred_o = o[-self.pred_dim:]
            pred_o = o[13:13 + self.pred_dim]

            if len(self.o_seq) >= self.max_timestep:
                self.o_seq = self.o_seq[1:]

            self.o_seq.append(np.expand_dims(pred_o.reshape(1, -1), axis=1))
            o_seq = np.concatenate(self.o_seq, axis=1)
            seq_len = [o_seq.shape[1]]


            x_p, uncer = self.sess.run(
                [self.pred_x, self.uncer], 
                feed_dict={
                    self.x_seq_ph: o_seq, 
                    self.seq_len_ph: seq_len
                }
            )
            print("uncer is: ", uncer)

            # delta_x = (np.squeeze(x_p) - pred_o)
            # uncer = np.zeros_like(np.squeeze(uncer))
            # o = np.concatenate([o, delta_x, uncer])

            delta_x = (np.squeeze(x_p) - pred_o)


            o = np.concatenate([o, delta_x, np.squeeze(uncer)])

            return o

        def __call__(self, o):
            o = self.compute_mixed_o(o)

            a = sess.run(self.action_op, feed_dict={self.model['x']: o.reshape(1,-1)})[0]        
            return a 

    get_action = GetAction(sess, model, 9, 10, deterministic)

    return get_action


def save_set(dir, obs_lst, action_lst, id):
    print("save data...")
    filename = os.path.join(dir,"demo_"+str(id)+".pkl")
    with open(filename, 'wb') as handle:
        pickle.dump({'obs_lst': obs_lst, 'action_lst': action_lst}, \
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("save dataset successfully at {}; data size is {}".format(filename, len(obs_lst)))


def main(robot_path, env, itr, test):
    seed = 60
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

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    get_robot_action = load_safe_policy(sess, robot_path, itr)

    env.render(mode="human")

    #prepare collect buffer list
    obs_lst = []
    action_lst = []
    obs = env.reset()
    get_robot_action.reset()
    print("obs initial", obs)
    id=0
    env.camera_adjust()
    env.render(mode="human")

    # time.sleep((3))
    ret = 0
    while(id<100):
        time.sleep(0.03)
        robot_a = get_robot_action(obs)

        obs, r, done, info = env.step(robot_a)
        ret += r
        if done == True:
            print("ret: ", ret)
            ret = 0
            env.reset()
            get_robot_action.reset()
            # print("reset")
            id+=1

            # time.sleep(10)
        env.render()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default=None)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument("--env", type=str, default="UR5HumanCollisionEnv-v0")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    main(args.fpath, args.env, args.itr, args.test)

