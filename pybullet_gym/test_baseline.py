import gym
import tensorflow as tf

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import make_session
from baselines.ppo2 import ppo2
from baselines.a2c import a2c

import pybullet_ur5
#!/usr/bin/env python
import argparse
import os, sys
from baselines import bench, logger
from mpi4py import MPI


def train(env_id, num_timesteps, num_env, seed, d_targ, load_path, point):
    env = make_vec_env(env_id, None, num_env, seed)
    env = VecNormalize(env, use_tf=True)

    model = ppo2.learn(env=env,
                       network='mlp',
                       lr=1e-3,
                       total_timesteps=5e5,
                       save_interval=2,
                       load_path=load_path)
    # model.save("baselines-a2c-two-finger.pkl")


def test(env_id, seed, load_path):
    num_env = 1
    env = make_vec_env(env_id, None, num_env, seed)
    env = VecNormalize(env, use_tf=True)
    ppo2.test(env=env,
           network='mlp',
           lr=1e-3,
           total_timesteps=300000,
           save_interval=5,
           load_path=load_path)
    # model.save("baselines-a2c-two-finger.pkl")



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='UR5HumanCollisionEnv-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--num-env', type=int, default=2)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--d_targ', type=float, default=0.012)
    parser.add_argument('--point', type=str, default='00050')
    args = parser.parse_args()
    curr_path = sys.path[0]
    if args.test is False:
        logger.configure(dir='{}/log'.format(curr_path))
        train(args.env, num_timesteps=args.num_timesteps, num_env=args.num_env, seed=args.seed,
              d_targ=args.d_targ, load_path=args.load_path, point=args.point)
    else:
        test(args.env, seed=args.seed, load_path=args.load_path)



if __name__ == '__main__':
    main()
