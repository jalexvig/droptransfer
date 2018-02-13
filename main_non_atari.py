import argparse
from multiprocessing import Process
import json
import os
import shutil

from droptransfer import CONFIG
from droptransfer import a2c


def parse_flags():
    """
    Parse CLI flags

    e.g. python train_pg.py CartPole-v0 -n 100 -b 1000 -e 5 -dna --exp_name sb_no_rtg_dna
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='CartPole-v0', help='Name of gym environment, e.g. CartPole-v0')
    parser.add_argument('--model', default='a2c', choices=['a2c'])
    parser.add_argument('--dpath_model', default='./saved')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument("--reset", action='store_true',
                        help="If set, delete the existing model directory and start training from scratch.")
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--normalize_advantages', '-na', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument("--dropout_keep", default=1.0, help="Probability to keep elements in dropout layers.", type=float)
    parser.add_argument("--dropconnect_keep", default=1.0, help="Probability to keep elements in dropconnect.", type=float)
    parser.add_argument("--run_name", default="default", help="Name of run.")
    parser.add_argument("--init_from", help="Directory to initialize model from.")
    parser.add_argument("--modify_env", help="Ways to modify gym environment (attrs to be set on env).", nargs='*')
    parser.parse_args(namespace=CONFIG)

    CONFIG.seed_orig = CONFIG.seed
    CONFIG.init_from_orig = CONFIG.init_from
    CONFIG.dpath_model_orig = CONFIG.dpath_model


def proc_flags():

    # Depending on the game we may have a limited action space
    run_name_components = [CONFIG.run_name,
                           CONFIG.env,
                           str(CONFIG.n_iter),
                           str(bool(CONFIG.init_from)),
                           str(CONFIG.dropout_keep),
                           str(CONFIG.dropconnect_keep)]
    CONFIG.dpath_model = os.path.join(CONFIG.dpath_model_orig, '_'.join(run_name_components), str(CONFIG.seed))

    # Optionally empty model directory
    if CONFIG.reset:
        shutil.rmtree(CONFIG.dpath_model, ignore_errors=True)

    CONFIG.dpath_checkpoint = os.path.join(CONFIG.dpath_model, 'checkpoints')
    if not os.path.exists(CONFIG.dpath_checkpoint):
        os.makedirs(CONFIG.dpath_checkpoint)

    if CONFIG.init_from_orig:
        CONFIG.init_from = os.path.join(CONFIG.init_from_orig, str(CONFIG.seed))

    with open(os.path.join(CONFIG.dpath_model, 'config.txt'), 'w') as f:
        json.dump(vars(CONFIG), f, indent=4)


if __name__ == '__main__':

    parse_flags()

    if CONFIG.model == 'a2c':
        for e in range(CONFIG.n_experiments):
            CONFIG.seed = CONFIG.seed_orig + 10 * e

            proc_flags()

            # Awkward hacky process runs, because Tensorflow does not like
            # repeatedly calling train in the same thread.
            p = Process(target=a2c.train, args=tuple())
            p.start()
            p.join()
