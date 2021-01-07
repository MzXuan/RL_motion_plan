import os


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


    her_path = "/home/xuan/her_models"
    try:
        os.makedirs(her_path)
    except:
        print("makedir error")
    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        save_path=her_path,
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

def random_n(max, min=None):
    result = []
    if min is None:
        for m in max:
            result.append(np.random.uniform(-m, m))
    else:
        for s,e in zip(min,max):
            result.append(np.random.uniform(s,e))
    return np.asarray(result)

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



    if args.play:
        pybullet.connect(pybullet.DIRECT)
        env = gym.make(args.env)
        env.render("human")

        logger.log("Running trained model")
        seed = 80
        np.random.seed(seed)
        tf.set_random_seed(seed)
        random.seed(seed)
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))



        episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)

        #
        #
        # #--------------------------- draw Q function for different robot state-------------------------------------#
        # # todo: manual update obs, test different Q, draw in pybullet simulator
        # # write an obs updater in environment
        # # get Q from model.step with q
        # # save Q value and corressponding obs
        # # normalize Q value and draw in pybullet gui
        # eef_current = obs['observation'][:3]
        # goal = obs['desired_goal']
        #
        # n = 20
        # x = np.linspace(goal[0]-0.6, goal[0]+0.6, num=n)
        # y = goal[1]
        # z = np.linspace(goal[2]-0.6, goal[2]+0.6, num=n)
        # xv, zv = np.meshgrid(x, z, sparse=False)
        #
        #
        # line_traj = []
        # for i in range(n):
        #     for j in range(n):
        #         line_traj.append([xv[i,j], y, zv[i,j]])
        #
        #
        # # line_traj = np.concatenate([np.linspace(eef_current, goal, num=20),
        # #                             np.linspace(eef_current, goal+random_n(max=[0.2, 0.2,0.2]), num=20),
        # #                             np.linspace(eef_current, goal+random_n(max=[0.2, 0.2,0.2]), num=20),
        # #                             np.linspace(eef_current, goal+random_n(max=[0.2, 0.2,0.2]), num=20)])
        #
        #
        #
        # line_traj = np.asarray(line_traj)
        # print("shape of line_traj: ", line_traj.shape)
        # obs_lst = []
        # q_lst = []
        # for i in range(60):
        #     # update env for several steps (let obstacle move)
        #     actions, Q, q, _ = model.step_with_q(obs)
        #     obs, rew, done, info = env.step(actions)
        #
        #
        # for next_eef in line_traj:
        #     #todo: batch operation
        #     actions, Q, q, _ = model.step_with_q(obs)
        #     # obs_lst.append(obs['observation'][:3])
        #     obs_lst.append(obs['desired_goal'][:3])
        #     q_lst.append(q)
        #
        #     # obs = env.update_robot_obs(next_eef)
        #     obs = env.update_goal_obs(next_eef)
        #     env.render()
        #
        # print("Q_list: ", q_lst)
        #
        #
        # env.draw_Q(obs_lst, q_lst)
        # time.sleep(200)
        #
        #
        #
        # #----------------------- draw Q for different goal state ---------------------------------#




        # ------------------------------ normal test ----------------------#
        min_dist_list = []
        env.render(mode="human")
        traj_count = 0
        success_rg_dist=[]
        fail_rg_dist=[]

        data_list = []

        fail_count = 0
        success_count = 0
        not_reach_count =0




        time_list = []
        while traj_count < 100:

            if state is not None:
                actions, _, state, _ = model.step(obs,S=state, M=dones)
            else:
                # start_time = time.time()
                actions, Q, q, _ = model.step_with_q(obs)
                # print("solving time is: ", time.time()-start_time)
                # time_list.append(time.time()-start_time)
                # data_list.append(np.concatenate([obs['observation'],obs['achieved_goal'], obs['desired_goal'], np.asarray([999]), actions]))

            # print("actions", actions)
            if len(time_list) > 3000:
                time_list = np.array(time_list[1:])
                print("mean of solving time: ", np.mean(time_list))
                print("std of solving time ", np.std(time_list))
                return

            obs, rew, done, info = env.step(actions)

            print("info: ", info['is_success'])
            # print("info: ", info['is_collision'])
            #
            # print("actions:", actions)
            # print("achieved goal", obs['achieved_goal'])
            # print("desired goal", obs['desired_goal'])


            safe_distance = info['safe_threshold']
            # print("min dist is: ", info['min_dist'])
            if info['min_dist']<0.6:
                min_dist_list.append(info['min_dist'])

            r_g_dist = np.linalg.norm(obs['achieved_goal']-obs['desired_goal'])
            if info['min_dist'] < 0.2:
                if info['is_collision'] is True:
                    fail_rg_dist.append(r_g_dist)
                else:
                    success_rg_dist.append(r_g_dist)



            # env.render()
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                np.savetxt("./simulate_data.csv", np.asarray(data_list), delimiter=",")
                data_list = []

                # break

                print("random seed is: ", seed)

                if info['is_collision'] is True:
                    print("min dist", info['min_dist'])
                    # robot goal distance
                    print("fail")
                    fail_count+=1
                elif info['is_success'] == 1.0:
                    print('success')
                    success_count+=1
                else:
                    not_reach_count+=1
                    print("fail with unknown reason")

                # for i in np.nonzero(done)[0]:
                #     print('episode_rew={}'.format(episode_rew[i]))
                #     episode_rew[i] = 0


                print("-------------end step {}----------".format(traj_count))
                traj_count+=1
                seed +=1
                np.random.seed(seed)
                tf.set_random_seed(seed)
                random.seed(seed)
                obs = env.reset()
                # time.sleep(1)
                # env.render()
                print("--------start----------")

        print("mean of minimum distance is: ", np.asarray(min_dist_list).mean())

        print("for minimum dist < 0.1, success rg dist mean is: ", np.asarray(success_rg_dist).mean())
        print("for minimum dist < 0.1, fail rg dist mean is: ", np.asarray(fail_rg_dist).mean())
        print("success rate is: ", (success_count+not_reach_count)/(fail_count+success_count))
        print("success count {} , collision  count {} not reach count {}: ".format(success_count, fail_count, not_reach_count))

    env.close()

    return model

if __name__ == '__main__':
    main(sys.argv)
