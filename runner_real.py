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
        # env = gym.make("UR5RealTestEnv-v0")
        env = gym.make("UR5HumanEnv-v0")
        env.render("human")

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

        time_lst = []
        start_time = time.time()
        while traj_count < 100:
            try:

                # update env for several steps (let obstacle move)
                actions, Q, q, _ = model.step_with_q(obs)
                obs, rew, done, info = env.step(actions)
                s+=1
                traj_len += np.linalg.norm(obs['observation'][:3] - last_obs['observation'][:3])
                last_obs = obs
                collision_lst.append(info['is_collision'])


                done_any = done.any() if isinstance(done, np.ndarray) else done
                if done_any:

                    # print("info", info)
                    # print("collision_lst is: ", collision_lst)
                    print("number of collision steps: ", sum(collision_lst))
                    print("success info",info["is_success"])
                    if sum(collision_lst) <=3 and info['is_success']:
                        success_count+=1
                        success_steps.append(s)
                        time_lst.append(time.time() - start_time)
                        traj_len_lst.append(traj_len)
                    env.agents[0].stop()

                    print("-------------end step {}---------".format(traj_count))
                    s=0
                    traj_count+=1
                    seed +=1
                    traj_len = 0
                    obs = env.reset()
                    last_obs = obs
                    collision_lst = []
                    start_time = time.time()
                    print("current mean of traj len is: ", np.array(traj_len_lst).mean())
                    print("current std of traj len is: ", np.array(traj_len_lst).std())
                    print("current mean reach time is: ", np.array(time_lst).mean())
                    print("current std of reach time is: ", np.array(time_lst).std())
                    print("current success rate is: ", success_count / traj_count)
                    print("current mean success steps is: ", np.array(success_steps).mean())
                    print("current std success steps is: ", np.array(success_steps).std())


                #----todo: generate batch obs---#
                time.sleep(0.1)
                # start_time = time.time()
                line_traj = []
                q_lst=[]

                path_remain = env.ws_path_gen.path_remain.copy()
                joint_path_remain = env.ws_path_gen.joint_path_remain.copy()
                _,_,_,goal_indices = env.ws_path_gen.next_goal(center=obs['observation'][:3],r=0.4, remove=False)

                # get trajectory after current indices
                if goal_indices>15:
                    for i in range(0, goal_indices, int(goal_indices/15)):
                        try:
                            p = path_remain[i]
                            jp = joint_path_remain[i]
                            next_state = np.concatenate([p,jp])
                            # print("next state is: ", next_state)
                        except:
                            continue
                        line_traj.append(env.update_robot_obs(obs['observation'], next_state)) #0.0016s for one obs if print; if not print, 0.0002s for one obs
                        # line_traj.append(env.update_robot_obs(p + random_n(max=[0.05, 0.05, 0.05])))
                        # line_traj.append(env.update_robot_obs(p + random_n(max=[0.05, 0.05, 0.05])))


                    # print("time cost 1 is: ", time.time() - start_time)
                    q_lst = model.get_collision_q(line_traj)
                    # print("time cost 3 is: ", time.time() - start_time)
                env.update_r(line_traj, q_lst, draw=False)


                # time.sleep(200)
            except KeyboardInterrupt:
                print("success rate is: ", success_count / traj_count)
                print("mean success steps is: ", np.array(success_steps).mean())
                print("std success steps is: ", np.array(success_steps).std())
                env.close()
                raise
        print("success rate is: ", success_count / traj_count)
        print("mean success steps is: ", np.array(success_steps).mean())
        print("std success steps is: ", np.array(success_steps).std())





        # #--------------------------- original test----------------------------------------#
        # traj_count = 0
        # total_steps = 1
        # data_list = []
        # last_time = time.time()
        # actions_list = []
        # env.set_sphere(0.1)
        # while traj_count < 300: #run at about 120 hz without gui; 30hz with gui
        #     try:
        #         print(time.time())
        #
        #         obs = env.get_obs() #0.001
        #         # print("obs",obs)
        #         actions, _, _, info = model.step(obs)  #0.002s
        #         data_list.append(np.concatenate([obs['observation'],obs['achieved_goal'], obs['desired_goal'], np.asarray([999]), actions]))
        #
        #         actions_list.append(actions)
        #
        #         obs, rew, done, info = env.step(actions) #0.01s
        #
        #         # img = env.render("rgb_array")
        #         # cv2.imshow("image", img)
        #         # cv2.waitKey(1)
        #         print("actions:", actions)
        #         print("achieved goal", obs['achieved_goal'])
        #         print("desired goal", obs['desired_goal'])
        #
        #         episode_rew += rew
        #         total_steps+=1
        #         # env.render()
        #         done_any = done.any() if isinstance(done, np.ndarray) else done
        #
        #         if done_any:
        #             print("info", info)
        #             print("total steps", total_steps)
        #             np.savetxt("./real_data.csv", np.asarray(data_list), delimiter=",")
        #             data_list= []
        #
        #             np.savetxt("./actions_data.csv", np.asarray(actions_list), delimiter=",")
        #             actions_list = []
        #
        #             env.agents[0].stop()
        #             time.sleep(2)
        #             # break
        #
        #             for i in np.nonzero(done)[0]:
        #                 print('episode_rew={}'.format(episode_rew[i]))
        #                 episode_rew[i] = 0
        #
        #
        #             print("-------------end step {}---------".format(traj_count))
        #             traj_count+=1
        #             seed +=1
        #             obs = env.reset()
        #
        #         last_time = time.time()
        #
        #     except KeyboardInterrupt:
        #         env.close()
        #         raise


    env.close()

    return model

if __name__ == '__main__':
    main(sys.argv)
