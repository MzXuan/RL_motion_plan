import tensorflow as tf
from baselines.her.util import store_args, nn, rnn


class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        print("!!!!!!!!!!!!!!!!!!!USE MLP MODEL!!!!!!!!!!!!!!!!!!!!!1")
        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                input_pi, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)
        with tf.variable_scope('qc'): #Q for collision checking
            # for collision checking, not use for policy training
            # input_Qc = tf.concat(axis=1, values=[o, g])
            # self._input_Qc = input_Qc  # exposed for tests
            # self.Qc_tf = nn(input_Qc, [self.hidden] * self.layers + [1])

            # input_Qc = tf.concat(axis=1, values=[o[:, 0:3], o[:, -40:], g])
            # self._input_Qc = input_Qc  # exposed for tests
            # self.Qc_tf = nn(input_Qc, [self.hidden] * self.layers + [1])

            input_Qc = tf.concat(axis=1, values=[o[:, 0:9], o[:, -22:-4], g])
            self._input_Qc = input_Qc  # exposed for tests
            # self.Qc_tf = nn(input_Qc, [self.hidden] * self.layers + [1], acti_fun=tf.nn.sigmoid)
            self.Qc_tf = nn(input_Qc, [self.hidden] * self.layers + [1])



class ActorCriticRNN:
    #todo: add rnn feature extration
    #todo: store hidden variable
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """

        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        print("!!!!!!!!!!!!!!!!!!!USE RNN POLICY!!!!!!!!!!!!!!!!!!!!!1")
        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor


        # Networks.
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(rnn(
                input_pi, [self.hidden] * self.layers + [self.dimu]))


        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = rnn(input_Q, [self.hidden] * self.layers + [1])


            # for critic training
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = rnn(input_Q, [self.hidden] * self.layers + [1], reuse=True)
        with tf.variable_scope('qc'): #Q for collision checking
            # for collision checking, not use for policy training
            # input_Qc = tf.concat(axis=1, values=[o[:,0:14],o[:,-40:], g])
            # self._input_Qc = input_Qc  # exposed for tests
            # self.Qc_tf = nn(input_Qc, [self.hidden] * self.layers + [1])

            # input_Qc = tf.concat(axis=1, values=[o[:, 0:3], o[:, -40:], g])
            # self._input_Qc = input_Qc  # exposed for tests
            # self.Qc_tf = nn(input_Qc, [self.hidden] * self.layers + [1])

            input_Qc = tf.concat(axis=1, values=[o[:, 0:9], o[:, -22:-4], g])
            # input_Qc = tf.concat(axis=1, values=[o[:, 0:9], o[:, -18:], g])
            self._input_Qc = input_Qc  # exposed for tests
            # self.Qc_tf = nn(input_Qc, [self.hidden] * self.layers + [1], acti_fun=tf.nn.sigmoid)
            self.Qc_tf = nn(input_Qc, [self.hidden] * self.layers + [1])
