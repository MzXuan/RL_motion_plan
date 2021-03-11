import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np
import random
import time
import pybullet
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import pandas as pd

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module

import pybullet_ur5
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

def random_n(max, min=None):
    result = []
    if min is None:
        for m in max:
            result.append(np.random.uniform(-m, m))
    else:
        for s,e in zip(min,max):
            result.append(np.random.uniform(s,e))
    return np.asarray(result)

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

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    env.close()


    if args.play:
        pybullet.connect(pybullet.DIRECT)
        # env = gym.make("UR5PreviousTestEnv-v0")
        env = gym.make("UR5HumanEnv-v0")
        # env = gym.make("UR5HumanRealEnv-v0")
        # env.render("human")

        logger.log("Running trained model")
        seed = 0
        np.random.seed(seed)
        tf.set_random_seed(seed)
        random.seed(seed)
        obs = env.reset()
        env.draw_path()
        last_obs = obs
        # env.render("rgb_array")

        episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)


        #--------------------------- draw Q function for different robot state-------------------------------------#
        eef_current = obs['observation'][:3]
        goal = obs['desired_goal']

        collision_lst = []
        success_count = 0
        success_steps = []
        traj_len_lst = []
        traj_count = 0
        s=0
        traj_len = 0
        robot_state_lst = []
        time_lst = []
        dynamic_time = []
        r_change_flag = True

        all_joint_err_list = []
        all_eef_err_list = []
        while traj_count < 300: #500 test trajectories
            try:
                # time.sleep(0.05)
                # update env for several steps (let obstacle move)
                # print("shape of obs is:", obs['observation'].shape)

                actions, Q, q, _ = model.step_with_q(obs)
                obs, rew, done, info = env.step(actions)
                s+=1
                traj_len += np.linalg.norm(obs['observation'][:3] - last_obs['observation'][:3])
                robot_state_lst.append(obs['observation'][:9])
                last_obs = obs
                collision_lst.append(info['is_collision'])

                done_any = done.any() if isinstance(done, np.ndarray) else done
                if info['is_success'] or done_any:

                    # print("info", info)
                    # print("collision_lst is: ", collision_lst)
                    print("number of collision steps: ", sum(collision_lst))
                    print("success info",info["is_success"])
                    print("steps is: ", s)
                    if sum(collision_lst) <=1 and info['is_success'] and s<250:
                        success_count+=1
                        success_steps.append(s)
                        time_lst.append(s*0.033)
                        traj_len_lst.append(traj_len)
                    env.agents[0].stop()

                    # ------plot result-----#
                    # plot_path(env.reference_path, env.eef_reference_path, robot_state_lst)
                    joint_err_list = calculate_joint_error(env.reference_path, robot_state_lst)
                    #
                    eef_error_list = calculate_carte_error(env.eef_reference_path, robot_state_lst)
                    avg_eef_error = np.asarray(eef_error_list).mean()
                    avg_joint_error = np.asarray(joint_err_list).mean()
                    all_joint_err_list.append(avg_joint_error)
                    all_eef_err_list.append(avg_eef_error)

                    print("-------------end step {}---------".format(traj_count))
                    s=0
                    traj_count+=1
                    seed +=1
                    traj_len = 0
                    time.sleep(1)
                    obs = env.reset()
                    last_obs = obs
                    collision_lst = []
                    robot_state_lst = []
                    # start_time = time.time()
                    print("current mean of traj len is: ", np.array(traj_len_lst).mean())
                    print("current std of traj len is: ", np.array(traj_len_lst).std())
                    print("current mean reach time is: ", np.array(time_lst).mean())
                    print("current std of reach time is: ", np.array(time_lst).std())

                    print("current mean joint error is: ", np.array(all_joint_err_list).mean())
                    print("current std of joint error is: ", np.array(all_joint_err_list).std())

                    print("current mean end_effector error is: ", np.array(all_eef_err_list).mean())
                    print("current std of end_effector error is: ", np.array(all_eef_err_list).std())

                    print("current mean dynamic time is: ", np.array(dynamic_time).mean())
                    print("current std of dynamic time is: ", np.array(dynamic_time).std())
                    print("current success rate is: ", success_count / traj_count)
                    print("current mean success steps is: ", np.array(success_steps).mean())
                    print("current std success steps is: ", np.array(success_steps).std())
                obs = env.get_obs()
                #
                # #----generate batch obs and dynamic select goal---#
                # #
                start_time = time.time()
                path_remain = env.ws_path_gen.path_remain.copy()
                joint_path_remain = env.ws_path_gen.joint_path_remain.copy()
                _,joint_goal,_,goal_indices = env.ws_path_gen.next_goal(center=obs['observation'][:3],r=0.3, remove=False, test=True)

                # linear interpolate robot states between current to future
                rob_current = obs['observation'][:9]
                goal_state = np.concatenate([path_remain[goal_indices],joint_path_remain[goal_indices]])
                next_states = np.linspace(rob_current, goal_state, num=10)

                # rob_current = obs['observation'][:3]
                # goal_state = np.array(path_remain[goal_indices])
                # next_states = np.linspace(rob_current, goal_state, num=10)

                line_traj = [env.update_robot_obs(obs['observation'], ns) for ns in next_states]
                q_lst = model.get_collision_q(line_traj)
                r_change_flag = env.update_r(line_traj, q_lst, draw=False)
                dynamic_time.append(time.time()-start_time)

            except KeyboardInterrupt:
                env.agents[0].stop()
                print("success rate is: ", success_count / traj_count)
                print("mean success steps is: ", np.array(success_steps).mean())
                print("std success steps is: ", np.array(success_steps).std())
                env.close()
                raise
        print("success rate is: ", success_count / traj_count)
        print("mean success steps is: ", np.array(success_steps).mean())
        print("std success steps is: ", np.array(success_steps).std())

    env.close()
    return model


def calculate_joint_error(ref, real):
    error_lst = []
    for p in real:
        # print("ref", ref.shape)
        # print("pshape",p.shape)
        jp = p[3:]
        # result = np.sum(np.abs(ref - jp), axis=1)
        result = np.linalg.norm(ref - jp, axis=1)
        error_lst.append(min(result))
        # print("shape of ref", len(ref))
        # print("shape of result", result.shape)
    return smoothing(error_lst)


def calculate_carte_error(ref, real):
    error_lst = []
    for p in real:
        jp = p[:3]
        result = np.linalg.norm(ref-jp,axis=1)
        error_lst.append(min(result))
        # print("shape of ref", len(ref))
        # print("shape of result", result.shape)

    return smoothing(error_lst)


def smoothing(dataset, smoothingWeight=0.2):
	set_smoothing =[]
	for idx, d in enumerate(dataset):
		if idx==0:
			last = d
		else:
			d_smoothed = last * smoothingWeight + (1 - smoothingWeight) * d
			last=d_smoothed
		set_smoothing.append(last)
	return set_smoothing


def plot_path(joint_ref_path, cat_ref_path,  real_path):
    # joint error
    j_error_lst = calculate_joint_error(joint_ref_path, real_path)
    plt.plot(range(len(j_error_lst)), j_error_lst)
    cat_error_lst = calculate_carte_error(cat_ref_path, real_path)
    plt.plot(range(len(cat_error_lst)), cat_error_lst)
    plt.show()

    df = pd.DataFrame()
    df['joint'] = j_error_lst
    df['end_effector'] = cat_error_lst
    df.to_csv(path_or_buf="/home/xuan/Code/motion_style/pybullet_gym/pybullet_ur5/utils/joint_human.csv")


if __name__ == '__main__':
    main(sys.argv)
