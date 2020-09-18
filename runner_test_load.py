import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np
import random

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module

from spinup.algos.tf1.value_pi.value_pi import value_pi as value_pi_tf1
from spinup.algos.tf1.value_pi.value_pi import VPGBuffer as vpg_buffer

import pandas as pd
import scipy.signal

import gym_rlmp

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))


    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    # env_type, env_id = get_env_type(args)

    env_id = args.env
    env_type = "robotics"

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)

    return env


def get_env_type(args):
    env_id = args.env

    # print("env id", env_id)
    # print("env_type: ", args.env_type)

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


class StateBuffer:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.val_buf = []

    def store(self, obs, act, rew):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)

    def finish_path(self):
        # self.val_buf[:] = self.discount_cumsum(self.rew_buf, self.gamma)[:-1]
        rews = np.append(self.rew_buf, 0)
        self.val_buf[:] = self.discount_cumsum(rews, self.gamma)[:-1]

        # print("len of obs {}, act {}, rew {} , val {}".\
        #       format(len(self.obs_buf), len(self.act_buf), len(self.rew_buf), len(self.val_buf)))
        # print("rew list is: ", self.rew_buf)
        # print("\n")
        # print("val list is: ", self.val_buf)

    def clear(self):
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.val_buf = []

    def discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def flatten_obs(obs):
    flat_obs = np.concatenate([obs['observation'].ravel(),obs['achieved_goal'].ravel(), obs['desired_goal'].ravel()])
    return flat_obs


def collect_data(model, env, total, name, seed):
    logger.log("Running trained model")
    seed = seed
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)
    obs = env.reset()

    dfObj = pd.DataFrame(columns=['obs', 'action', 'rew', 'val'])

    # state = model.initial_state if hasattr(model, 'initial_state') else None
    # dones = np.zeros((1,))
    #
    # fail_count = 0
    # success_count = 0

    episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)

    buf = StateBuffer()
    trajs_count = 0

    while trajs_count < total:
        actions, Q, _, _ = model.step_with_q(obs)
        # print("q is: ", Q)
        obs2, rew, done, info = env.step(actions)

        flat_obs = flatten_obs(obs)
        buf.store(obs=flat_obs, act=actions, rew=rew)
        # dfObj = dfObj.append({'obs': obs, 'action': actions, 'rew': rew}, ignore_index=True)
        obs = obs2

        episode_rew += rew
        # env.render()
        done_any = done.any() if isinstance(done, np.ndarray) else done
        if done_any:
            trajs_count +=1
            print("-------finish path {}----------".format(trajs_count))
            for i in np.nonzero(done)[0]:
                print('episode_rew={}'.format(episode_rew[i]))
                episode_rew[i] = 0

            # calculate value and append values to pandas
            buf.finish_path()
            xtra = {'obs': buf.obs_buf, 'action': buf.act_buf, 'rew':buf.rew_buf,'val':buf.val_buf}
            dfObj = dfObj.append(pd.DataFrame(xtra))
            print("-------------------------")
            #reset
            buf.clear()
            obs = env.reset()

    try:
        file_name = "./data/"+name+".pkl"
        dfObj.to_pickle(file_name)
        print("save data {} successfully".format(file_name))
    except:
        print("fail to save data")



def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    model, env = train(args, extra_args)


    if args.play:
        collect_data(model, env, total = 30000, name="mydata",seed=3000)
        collect_data(model, env, total=2000, name="valid_mydata", seed=0)

    env.close()

if __name__ == '__main__':
    main(sys.argv)
