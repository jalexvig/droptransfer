import itertools
import threading

import os
import tensorflow as tf

from droptransfer import CONFIG
from droptransfer.a3c.estimators import ValueEstimator, PolicyEstimator
from droptransfer.a3c.policy_monitor import PolicyMonitor
from droptransfer.a3c.worker import Worker
from droptransfer.utils import make_env


def train():

    summary_writer = tf.summary.FileWriter(os.path.join(CONFIG.dpath_model, "train"))

    with tf.device("/cpu:0"):
        # Keeps track of the number of updates we've performed
        tf.Variable(0, name="global_step", trainable=False)

        # Global policy and value nets that share first part of network
        with tf.variable_scope("global"):
            policy_net = PolicyEstimator(num_outputs=len(CONFIG.valid_actions))
            value_net = ValueEstimator()

        # Note this is okay in CPython since it's thread safe
        # https://www.reddit.com/r/Python/comments/52mlee/a_threadsafe_incrementing_counter_very_simple_but/
        # Global step iterator
        global_counter = itertools.count()
        next(global_counter)

        # Create worker graphs
        workers = []
        for worker_id in range(CONFIG.num_workers):
            # We only write summaries in one of the workers because they're
            # pretty much identical and writing them on all workers
            # would be a waste of space
            worker_summary_writer = summary_writer if worker_id == 0 else None

            worker = Worker(
                name="worker_{}".format(worker_id),
                env=make_env(CONFIG.env),
                policy_net=policy_net,
                value_net=value_net,
                global_counter=global_counter,
                discount_factor=0.99,
                summary_writer=worker_summary_writer,
                max_global_steps=CONFIG.max_global_steps)
            workers.append(worker)

        saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.0, max_to_keep=10)

        # Used to occasionally save videos for our policy net
        # and write episode rewards to Tensorboard
        pe = PolicyMonitor(
            env=make_env(CONFIG.env, wrap=False),
            policy_net=policy_net,
            summary_writer=summary_writer,
            saver=saver)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()

        # Load a previous checkpoint if it exists
        latest_checkpoint = tf.train.latest_checkpoint(CONFIG.init_from or CONFIG.dpath_checkpoint)
        if latest_checkpoint:
            print("Loading model checkpoint: {}".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)

        # Start worker threads
        worker_threads = []
        for worker in workers:
            worker_fn = lambda worker=worker: worker.run(sess, coord, CONFIG.t_max)
            t = threading.Thread(target=worker_fn)
            t.start()
            worker_threads.append(t)

        # Start a thread for policy eval task
        monitor_thread = threading.Thread(target=lambda: pe.continuous_eval(CONFIG.eval_every, sess, coord))
        monitor_thread.start()

        # Wait for all workers to finish
        coord.join(worker_threads)
