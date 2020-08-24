import joblib
import os
import os.path as osp


import numpy as np
import tensorflow as tf
import gym
import pybullet_ur5

import time
import spinup.algos.tf1.legible_ppo.depth_core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, img_h, img_w, obs_dim, state_dim, act_dim, size, gamma=0.99, lam=0.95):

        self.state_dim = state_dim

        self.depth_buf = np.zeros([size, img_h, img_w, 1], dtype=np.float32)
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, depth, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.depth_buf[self.ptr] = depth
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr


    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.depth_buf, 
                self.obs_buf, 
                self.act_buf, 
                self.adv_buf, 
                self.ret_buf, 
                self.logp_buf]


def ppo(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=16000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=200,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10,
        chk_save_freq=50, load=False, start_itr=0, fpath=None):
    """
    Proximal Policy Optimization (by clipping),

    with early stopping based on approximate KL

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
            function you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`.

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

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
    obs_dim = env.state_space.shape
    act_dim = env.action_space.shape
    state_dim = env.state_space.shape

    img_w = env.image_width
    img_h = env.image_height

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    depth_ph = tf.placeholder(tf.float32, [None, img_h, img_w, 1], name="depth")
    x_ph, a_ph = core.placeholders_from_spaces(env.state_space, env.action_space)
    adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    # Main outputs from computation graph
    mu, pi, logp, logp_pi, v = actor_critic(depth_ph, x_ph, a_ph, **ac_kwargs)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [depth_ph, x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(img_h, img_w, obs_dim, state_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)
    # params = tuple(core.get_vars(scope) for scope in ['pi', 'v']) #for weight saving
    saver = tf.train.Saver(name="saver")

    # PPO objectives
    ratio = tf.exp(logp - logp_old_ph)  # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(adv_ph > 0, (1 + clip_ratio) * adv_ph, (1 - clip_ratio) * adv_ph)
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    v_loss = tf.reduce_mean((ret_ph - v) ** 2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)  # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)  # a sample estimate for entropy, also easy to compute
    clipped = tf.logical_or(ratio > (1 + clip_ratio), ratio < (1 - clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))


    # Optimizers
    train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    sess = tf.Session(config=conf)
    sess.run(tf.global_variables_initializer())

    #if load is True, load saved model
    if load is True:
        chk_file = os.path.join(fpath,'checkpoint%d.chk' % start_itr)
        try:
            saver.restore(sess, chk_file)
            print("load check point from {}".format(chk_file))
        except:
            print("fail to load check point from {}".format(chk_file))
            return


    # Sync params across processes
    sess.run(sync_all_params())
    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'depth': depth_ph, 'x': x_ph}, outputs={'mu': mu, 'pi': pi, 'v': v})


    def update():
        inputs = {k: v for k, v in zip(all_phs, buf.get())}
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)
        
        # Training
        for i in range(train_pi_iters):
            _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)
        
        # Log changes from update
        pi_l_new, v_l_new, kl, cf, adv_new = sess.run([pi_loss, v_loss, approx_kl, clipfrac, min_adv], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old, MinAdv=adv_new,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

        # add batch loop reading
        # max_batch = 50
        # for i in range(0, local_steps_per_epoch-max_batch+1, max_batch):
        #     start = i
        #     end = i + max_batch
        #     # print("start is {} and end is {}".format(start, end))
        #     if end > local_steps_per_epoch or start > local_steps_per_epoch:
        #         break
        #     inputs = {k: v for k, v in zip(all_phs, buf.get(start, end))}

        #     pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)

        #     # Training
        #     for i in range(train_pi_iters):
        #         _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
        #         kl = mpi_avg(kl)
        #         if kl > 1.5 * target_kl:
        #             logger.log('Early stopping at step %d due to reaching max kl.' % i)
        #             break
        #     logger.store(StopIter=i)
        #     for _ in range(train_v_iters):
        #         sess.run(train_v, feed_dict=inputs)

        #     # Log changes from update
        #     pi_l_new, v_l_new, kl, cf, adv_new = sess.run([pi_loss, v_loss, approx_kl, clipfrac, min_adv], feed_dict=inputs)
        #     logger.store(LossPi=pi_l_old, LossV=v_l_old, MinAdv=adv_new,
        #                  KL=kl, Entropy=ent, ClipFrac=cf,
        #                  DeltaLossPi=(pi_l_new - pi_l_old),
        #                  DeltaLossV=(v_l_new - v_l_old))

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            depth = o['depth']
            x = o['state']

            a, v_t, logp_t = sess.run(get_action_ops, feed_dict={depth_ph: [depth],
                                                                 x_ph: x.reshape(1, -1)})

            o2, r, d, _ = env.step(a[0])

            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o['depth'], o['state'], a, r, v_t, logp_t)
            logger.store(VVals=v_t)

            # Update obs (critical!)
            o = o2

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t == local_steps_per_epoch - 1):
                if not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = 0 if d else sess.run(v, feed_dict={depth_ph: [depth],
                                                              x_ph: x.reshape(1, -1)})
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        #save checkpoints
        if (epoch % chk_save_freq == 0) or (epoch == epochs - 1):
            logger.save_weights(saver, sess, epoch+start_itr)

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('MinAdv', average_only=True)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PyUR5ReachEnv-v1')
    parser.add_argument('--hid', type=int, default=80)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=100)
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--steps', type=int, default=3200)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--exp_name', type=str, default='legible_depth_ppo')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--fpath', type=str, default=None)
    parser.add_argument('--itr', type=int, default=0)

    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda: gym.make(args.env), actor_critic=core.conv_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, load=args.load, start_itr=args.itr, fpath=args.fpath)
