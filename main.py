import os
import shutil

import argparse

from droptransfer import a3c, CONFIG
from droptransfer.utils import make_env


def parse_flags():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='a3c', choices=['a3c', 'dqn'])
    parser.add_argument('--model_dir', default='./saved')
    parser.add_argument('--env', default='Breakout-V0', help='Name of gym Atari environment, e.g. Breakout-v0')
    parser.add_argument("--t_max", default=5, help="Number of steps before performing an update", type=int)
    parser.add_argument("--max_global_steps", type=int,
                        help="Stop training after this many steps in the environment. Defaults to running indefinitely.")
    parser.add_argument("--eval_every", default=300, help="Evaluate the policy every N seconds", type=int)
    parser.add_argument("--reset", action='store_true',
                        help="If set, delete the existing model directory and start training from scratch.")
    parser.add_argument("--num_workers", help="Number of threads to run. If not set we run [num_cpu_cores] threads.",
                        type=int)
    parser.add_argument("--keep_prob", default=1.0, help="Probability to keep elements in dropout layers.", type=float)
    parser.add_argument("--run_name", default="default", help="Name of run.")
    parser.add_argument("--init_from", help="Directory to initialize model from.")

    parser.parse_args(namespace=CONFIG)


def proc_flags():

    # Depending on the game we may have a limited action space
    if CONFIG.env == "Pong-v0" or CONFIG.env == "Breakout-v0":
        CONFIG.valid_actions = list(range(4))
    else:
        env_ = make_env(CONFIG.env)
        CONFIG.valid_actions = list(range(env_.action_space.n))
        env_.close()

    # Set the number of workers
    if not CONFIG.num_workers:
        import multiprocessing
        CONFIG.num_workers = multiprocessing.cpu_count()

    run_name_components = [CONFIG.run_name,
                           CONFIG.env,
                           str(CONFIG.max_global_steps),
                           str(bool(CONFIG.init_from)),
                           str(CONFIG.keep_prob)]
    CONFIG.dpath_model = os.path.join(CONFIG.model_dir, '_'.join(run_name_components))

    # Optionally empty model directory
    if CONFIG.reset:
        shutil.rmtree(CONFIG.dpath_model, ignore_errors=True)

    CONFIG.dpath_checkpoint = os.path.join(CONFIG.dpath_model, 'checkpoints')
    if not os.path.exists(CONFIG.dpath_checkpoint):
        os.makedirs(CONFIG.dpath_checkpoint)

    if CONFIG.init_from:
        with open(os.path.join(CONFIG.dpath_model, 'init_from.txt'), 'w') as f:
            f.write(CONFIG.init_from)


if __name__ == '__main__':

    parse_flags()
    proc_flags()

    if CONFIG.model == 'a3c':
        a3c.train()
