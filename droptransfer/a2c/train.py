import os
import time

import gym
import numpy as np
import tensorflow as tf
from droptransfer.a2c.estimators import PolicyEstimator, ValueEstimator

from droptransfer import CONFIG

STABILITY = 1e-8


def train():

    env, policy_net, value_net = _setup(CONFIG.env, CONFIG.seed)

    writer = tf.summary.FileWriter(os.path.join(CONFIG.dpath_model, 'train'))
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.0, max_to_keep=10)

    # Maximum length for episodes
    max_path_length = CONFIG.ep_len if CONFIG.ep_len > 0 else None
    max_path_length = max_path_length or env.spec.max_episode_steps

    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        total_timesteps = 0

        # Initialize from other run
        if CONFIG.init_from:
            latest_checkpoint = tf.train.latest_checkpoint(CONFIG.init_from)
            if not latest_checkpoint:
                raise ValueError('Could not initialize from {}'.format(CONFIG.init_from))
        else:
            latest_checkpoint = tf.train.latest_checkpoint(CONFIG.dpath_checkpoint)

        if latest_checkpoint:
            print("Loading model checkpoint: {}".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)

        reset_ops = policy_net.reset_ops + value_net.reset_ops

        sess.run(reset_ops)

        for itr in range(CONFIG.n_iter):

            # Collect paths until we have enough timesteps
            timesteps_this_batch = 0
            paths = []
            while True:
                obs = env.reset()
                observations, actions, rewards = [], [], []
                render_this_episode = (not paths and (itr % 10 == 0) and CONFIG.render)
                steps = 0
                while True:
                    if render_this_episode:
                        env.render()
                        time.sleep(0.05)
                    observations.append(obs)
                    sampled_action = sess.run(policy_net.sampled, feed_dict={policy_net.observations: obs[None]})[0]
                    actions.append(sampled_action)
                    obs, rew, done, _ = env.step(sampled_action)
                    rewards.append(rew)
                    steps += 1
                    if done or steps > max_path_length:
                        break
                path = {"observation": np.array(observations),
                        "reward": np.array(rewards),
                        "action": np.array(actions)}
                paths.append(path)
                timesteps_this_batch += len(path["reward"])
                if timesteps_this_batch > CONFIG.batch_size:
                    break
            total_timesteps += timesteps_this_batch

            # Build arrays for observation, action for the policy gradient update by concatenating
            # across paths
            ob_no = np.concatenate([path["observation"] for path in paths])
            ac_na = np.concatenate([path["action"] for path in paths])

            q_n = []

            for path in paths:
                n = path["reward"].shape[0]
                discounts = CONFIG.discount ** np.arange(n)
                discounted_rew_seq = discounts * path["reward"]
                q_path = np.cumsum(discounted_rew_seq[::-1])[::-1] / discounts

                q_n.extend(q_path)

            q_n = np.array(q_n)

            val_predicted = sess.run(value_net.predicted_values, {value_net.observations: ob_no})
            val_predicted_norm = _normalize(val_predicted, q_n.mean(), q_n.std())
            adv_n = q_n - val_predicted_norm

            if not CONFIG.dont_normalize_advantages:
                adv_n = _normalize(adv_n)

            feed_dict = {
                policy_net.observations: ob_no,
                policy_net.actions: ac_na,
                policy_net.targets: adv_n,
                value_net.observations: ob_no,
                value_net.targets: _normalize(q_n),
            }

            summaries = tf.summary.merge_all(tf.GraphKeys.SUMMARIES)

            policy_loss, _, value_loss, _, summaries_all = sess.run([
                policy_net.loss,
                policy_net.update_op,
                value_net.loss,
                value_net.update_op,
                summaries,
            ], feed_dict=feed_dict)

            writer.add_summary(summaries_all, global_step=itr + 1)

            add_path_summaries(itr, paths, writer)

            writer.flush()

            if itr % 10 == 0:
                print(policy_loss, value_loss)

        saver.save(sess, os.path.join(CONFIG.dpath_model, 'checkpoints', 'model'))


def _setup(env_name, seed):

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)

    for name, type_, val in zip(*[iter(CONFIG.modify_env)] * 3):

        if type_ not in ['str', 'int', 'float']:
            raise ValueError('Cannot parse type %s' % type_)

        setattr(env.env, name, eval(type_)(val))

    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    policy_net = PolicyEstimator(ob_dim, ac_dim)
    value_net = ValueEstimator(ob_dim)

    return env, policy_net, value_net


def add_path_summaries(itr, paths, writer):

    data = {
        'returns': [path["reward"].sum() for path in paths],
        'length': [len(path["reward"]) for path in paths],
    }

    funcs = {
        'mean': np.mean,
        'std': np.std,
        'max': np.max,
        'min': np.min,
    }

    for data_name, func_name in [
        ('returns', 'mean'),
        ('returns', 'std'),
        ('returns', 'max'),
        ('returns', 'min'),
        ('length', 'mean'),
        ('length', 'std'),
    ]:
        name = 'path/%s/%s' % (data_name, func_name)
        val = funcs[func_name](data[data_name])
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=val)])
        writer.add_summary(summary, global_step=itr + 1)


def _normalize(a, u=0, s=1):
    a_norm = (a - np.mean(a)) / (np.std(a) + STABILITY)
    a_rescaled = a_norm * s + u

    return a_rescaled
