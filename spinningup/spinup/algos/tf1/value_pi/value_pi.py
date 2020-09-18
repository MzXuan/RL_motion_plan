import numpy as np
import tensorflow as tf
import gym
import time
import spinup.algos.tf1.value_pi.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

import pandas as pd
import os

class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        # self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        # self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val

        self.ptr += 1
    #
    # def finish_path(self, last_val=0):
    #     """
    #     Call this at the end of a trajectory, or when one gets cut off
    #     by an epoch ending. This looks back in the buffer to where the
    #     trajectory started, and uses rewards and value estimates from
    #     the whole trajectory to compute advantage estimates with GAE-Lambda,
    #     as well as compute the rewards-to-go for each state, to use as
    #     the targets for the value function.
    #
    #     The "last_val" argument should be 0 if the trajectory ended
    #     because the agent reached a terminal state (died), and otherwise
    #     should be V(s_T), the value function estimated for the last state.
    #     This allows us to bootstrap the reward-to-go calculation to account
    #     for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
    #     """
    #
    #     # path_slice = slice(self.path_start_idx, self.ptr)
    #     # rews = np.append(self.rew_buf[path_slice], last_val)
    #     # # vals = np.append(self.val_buf[path_slice], last_val)
    #     #
    #     # # the next two lines implement GAE-Lambda advantage calculation
    #     # # deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
    #     # # self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
    #     #
    #     # # the next line computes rewards-to-go, to be targets for the value function
    #     # self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
    #     #
    #     self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get

        self.ptr, self.path_start_idx = 0, 0
        #my get function
        return [self.obs_buf, self.val_buf]

    def valid_get(self):
        #not reset

        assert self.ptr == self.max_size  # buffer has to be full before you can get
        return [self.obs_buf, self.val_buf]

        # # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        # self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        # return [self.obs_buf, self.act_buf, self.adv_buf,
        #         self.ret_buf, self.logp_buf]



def value_pi(env_fn, critic=core.value_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=10,
        chk_save_freq=50, load=False, start_itr=0, fpath=None):
    """
    Vanilla Policy Gradient 

    (with GAE-Lambda for advantage estimation)

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a_ph``
                                           | in states ``x_ph``.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``.
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. (Critical: make sure 
                                           | to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to VPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    env = gym.wrappers.FlattenObservation(env)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    
    # Share information about action space with policy architecture
    # ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    # x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
    x_ph = core.placeholder_from_space(env.observation_space)
    val_ph = core.placeholder(None)
    # adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    # Main outputs from computation graph
    # pi, logp, logp_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)

    v = critic(x_ph, **ac_kwargs)



    # Need all placeholders in *this* order later (to zip with data from buffer)
    # all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    all_phs = [x_ph, val_ph]

    # Every step, get: action, value, and logprob
    # get_action_ops = [pi, v, logp_pi]

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = VPGBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    valid_buf = VPGBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['v'])
    logger.log('\nNumber of parameters: \t v: %d\n'%var_counts)
    saver = tf.train.Saver(name="saver")

    # VPG objectives
    # pi_loss = -tf.reduce_mean(logp * adv_ph)
    v_loss = tf.reduce_mean((val_ph - v)**2)

    # Info (useful to watch during learning)
    # approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
    # approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute

    # Optimizers
    # train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # if load is True, load saved model
    if load is True:
        chk_file = os.path.join(fpath, 'chk', 'checkpoint%d.chk' % start_itr)
        try:
            saver.restore(sess, chk_file)

            # # re-initialized pi/log_std
            # pi_log_std = tf.get_default_graph().get_tensor_by_name("pi/log_std:0")
            # pi_log_std_assign = tf.assign(pi_log_std, -0.5 * np.ones(act_dim, dtype=np.float32))
            # sess.run(pi_log_std_assign)
            # print(pi_log_std.eval(session=sess))
            print("load check point from {}".format(chk_file))
        except:
            print("fail to load check point from {}".format(chk_file))
            return

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'v': v})

    def update():
        inputs = {k:v for k,v in zip(all_phs, buf.get())}
        valid_inputs = {k:v for k,v in zip(all_phs, valid_buf.valid_get())}

        v_l_old = sess.run([v_loss], feed_dict=inputs)

        valid_loss = sess.run([v_loss], feed_dict=valid_inputs)



        # pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)

        # Policy gradient step
        # sess.run(train_pi, feed_dict=inputs)

        # Value function learning
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update
        # pi_l_new, v_l_new, kl = sess.run([pi_loss, v_loss, approx_kl], feed_dict=inputs)
        v_l_new = sess.run([v_loss], feed_dict=inputs)

        # print("v_l_old: {} and v_l_new {}", v_l_old, v_l_new)
        logger.store(LossV=v_l_old,
                     ValidLossV = valid_loss[0],
                     DeltaLossV=(v_l_new[0] - v_l_old[0]))


    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    # prepare validate data
    valid_df = pd.read_pickle("/home/xuan/Code/motion_style/data/validate_mydata.pkl")
    valid_row_gen = valid_df.iterrows()
    for t in range(local_steps_per_epoch):
        try:
            row = next(valid_row_gen)[1]
        except:
            print("iterate to data end, re-sample from beginning")
            valid_row_gen = valid_df.iterrows()
            row = next(valid_row_gen)[1]
        o, a, r, v = row.obs, row.action, row.rew, row.val

        # save and log
        valid_buf.store(o, a, r, v)



    saved_df = pd.read_pickle("/home/xuan/Code/motion_style/data/mydata.pkl")
    print("local steps per epoch is: ", local_steps_per_epoch)
    print("row count is: ", saved_df.shape[0])
    row_gen = saved_df.iterrows()
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        #todo: load saved buffer
        for t in range(local_steps_per_epoch):
            # a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1,-1)})
            #
            # o2, r, d, _ = env.step(a[0])
            # ep_ret += r
            # ep_len += 1

            try:
                row = next(row_gen)[1]
            except:
                print("iterate to data end, re-sample from beginning")
                row_gen = saved_df.iterrows()
                row = next(row_gen)[1]

            o, a, r, v = row.obs, row.action, row.rew, row.val

            # save and log
            buf.store(o, a, r, v)

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # save checkpoints
        if (epoch % chk_save_freq == 0) or (epoch == epochs - 1):
            print("saving checkpoints, iteration is: {} and start itr is {}".format(epoch, start_itr))
            logger.save_weights(saver, sess, epoch + start_itr)

        # Perform VPG update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('ValidLossV', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='FetchDynamicCollectReach-v2')
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=8)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--exp_name', type=str, default='value_pi')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--fpath', type=str, default=None)
    parser.add_argument('--itr', type=int, default=0)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    value_pi(lambda : gym.make(args.env), critic=core.value_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs,  load=args.load, start_itr=args.itr, fpath=args.fpath)
