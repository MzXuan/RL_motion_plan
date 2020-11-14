import os
import subprocess
import sys
import importlib
import inspect
import functools

import tensorflow as tf
import numpy as np

from baselines.common import tf_util as U


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def import_function(spec):
    """Import a function identified by a string like "pkg.module:fn_name".
    """
    mod_name, fn_name = spec.split(':')
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_name)
    return fn


def flatten_grads(var_list, grads):
    """Flattens a variables and their gradients.
    """
    # print("varlist:", var_list)
    # print("grad: ", grads)
    return tf.concat([tf.reshape(grad, [U.numel(v)])
                      for (v, grad) in zip(var_list, grads)], 0)


def nn(input, layers_sizes, reuse=None, flatten=False, name=""):
    """Creates a simple neural network
    """
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                reuse=reuse,
                                name=name + '_' + str(i))
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input


def rnn(input, layers_sizes, reuse=False, flatten=False, name=""):
    """Creates a simple neural network
    """
    #todo: add gru cell

    # print("shape of gru input: ", gru_input.shape)

    # out,states = tf.keras.layers.GRU(32, return_sequences=True, return_state=True)(
    #     tf.stack([input[14:14+18],input[14:14+18]], axis=1)) #[batch, timesteps, feature]

    # out = tf.nn.rnn_cell.GRUCell(32, reuse=reuse)(
    #     tf.stack([input[14:-4],input[14:-4]], axis=1))  # [batch, timesteps, feature]

    # gru_cell = tf.nn.rnn_cell.GRUCell(32)
    #
    # outputs, state = tf.nn.dynamic_rnn(gru_cell, tf.stack([input[14:-4],input[14:-4]], axis=1),
    #                                              dtype=tf.float32)
    # print("states shape", outputs.shape)

    def gru_cell(reuse):
        return tf.nn.rnn_cell.GRUCell(32, reuse=reuse)
        # return tf.keras.layers.GRUCell(32)
        # return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)



    # input_exp_dim = tf.stack([input[:,14:14+180],input[:,14:14+180]], axis=1)

    human_dim = 18
    human_step = 10
    batch_size = input.shape[0]

    in_human_flat = tf.expand_dims(input[:,14:14+human_dim*human_step],axis=2)
    print("shape of in flat", in_human_flat.shape)
    input_human =tf.reshape(in_human_flat, shape = [-1,10, 18])

    # input_human = tf.keras.layers.Reshape(target_shape=(batch_size,human_step, human_dim))(
    #     input[:,14:14+human_dim*human_step])

    out,state = tf.keras.layers.RNN(gru_cell(reuse), return_sequences=True, return_state=True)(input_human)

    input = tf.concat([input[:,:14],input[:,14+human_dim*human_step:], state], axis=1)




    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None



        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                reuse=reuse,
                                name=name + '_' + str(i))
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input



def install_mpi_excepthook():
    import sys
    from mpi4py import MPI
    old_hook = sys.excepthook

    def new_hook(a, b, c):
        old_hook(a, b, c)
        sys.stdout.flush()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort()
    sys.excepthook = new_hook


def mpi_fork(n, extra_mpi_args=[]):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    """
    if n <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        # "-bind-to core" is crucial for good performance
        args = ["mpirun", "-np", str(n)] + \
            extra_mpi_args + \
            [sys.executable]

        args += sys.argv
        subprocess.check_call(args, env=env)
        return "parent"
    else:
        install_mpi_excepthook()
        return "child"


def convert_episode_to_batch_major(episode):
    """Converts an episode to have the batch dimension in the major (first)
    dimension.
    """
    episode_batch = {}
    for key in episode.keys():
        val = np.array(episode[key]).copy()
        # make inputs batch-major instead of time-major
        episode_batch[key] = val.swapaxes(0, 1)

    return episode_batch


def transitions_in_episode_batch(episode_batch):
    """Number of transitions in a given episode batch.
    """
    shape = episode_batch['u'].shape
    return shape[0] * shape[1]


def reshape_for_broadcasting(source, target):
    """Reshapes a tensor (source) to have the correct shape and dtype of the target
    before broadcasting it with MPI.
    """
    dim = len(target.get_shape())
    shape = ([1] * (dim - 1)) + [-1]
    return tf.reshape(tf.cast(source, target.dtype), shape)
