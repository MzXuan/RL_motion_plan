import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError

def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def conv_mlp(depth, x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    conv_output = depth_conv(depth)
    x = tf.concat([conv_output, x], axis=1)
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]

def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


"""
CNNs
"""

def depth_conv(x):
    x = tf.layers.conv2d(
        x,
        filters=32,
        kernel_size=8,
        strides=(4, 4),
        activation=tf.nn.relu
    )
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)
    x = tf.layers.conv2d(
        x,
        filters=64,
        kernel_size=3,
        strides=(1, 1),
        activation=tf.nn.relu
    )
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(
        x, 128, tf.nn.tanh
    )
    return x


#
# def depth_conv(x):
#     x = tf.layers.conv2d(
#         x,
#         filters=16,
#         kernel_size=8,
#         strides=(4, 4),
#         activation=tf.nn.relu
#     )
#     print("!!!!!!x shape is {}".format(x))
#     x = tf.layers.conv2d(
#         x,
#         filters=32,
#         kernel_size=4,
#         strides=(2, 2),
#         activation=tf.nn.relu
#     )
#     print("!!!!!!x shape is {}".format(x))
#     x = tf.layers.conv2d(
#         x,
#         filters=32,
#         kernel_size=3,
#         strides=(1, 1),
#         activation=tf.nn.relu
#     )
#     print("!!!!!!x shape is {}".format(x))
#     x = tf.layers.flatten(x)
#     print("!!!!!!x shape is {}".format(x))
#
#     x = tf.layers.dense(
#         x, 128, tf.nn.tanh
#     )
#
#     return x

"""
Policies
"""

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space.n
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp, logp_pi

def conv_gaussian_policy(depth, x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = a.shape.as_list()[-1]
    mu = conv_mlp(depth, x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp, logp_pi


"""
Actor-Critics
"""
def conv_actor_critic(depth, x, a, hidden_sizes=(64,64), activation=tf.tanh,
                     output_activation=None, policy=None, action_space=None):

    # default policy builder depends on action space
    if policy is None and isinstance(action_space, Box):
        policy = conv_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        print("No implementation of pf discrete situation!!!")


    with tf.variable_scope('pi'):
        mu, pi, logp, logp_pi = policy(depth, x, a, hidden_sizes, activation, output_activation, action_space)
    with tf.variable_scope('v'):
        v = tf.squeeze(conv_mlp(depth, x, list(hidden_sizes)+[1], activation, None), axis=1)
    return mu, pi, logp, logp_pi, v


def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh, 
                     output_activation=None, policy=None, action_space=None):

    # default policy builder depends on action space
    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy


    with tf.variable_scope('pi'):
        mu, pi, logp, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, action_space)
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return mu, pi, logp, logp_pi, v
