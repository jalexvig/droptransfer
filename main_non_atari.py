import argparse
from multiprocessing import Process
import json
import os
import shutil

from droptransfer import CONFIG
from droptransfer import a2c
from droptransfer.utils import combine_events


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
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument("--dropout_keep", default=1.0, help="Probability to keep elements in dropout layers.", type=float)
    parser.add_argument("--permanent_keep", default=1.0,
                        help="Probability to keep each weight - otherwise it will be reset.", type=float)
    parser.add_argument('--num_units', '-nu', nargs='*', type=int, default=[64, 64])
    parser.add_argument("--run_name", default="default", help="Name of run.")
    parser.add_argument("--init_from", help="Directory to initialize model from.")
    parser.add_argument("--modify_env", help="Ways to modify gym environment (attrs to be set on env).", nargs='*', default=[])
    parser.add_argument('--dont_combine', '-dc', action='store_true')
    parser.parse_args(namespace=CONFIG)


def proc_flags():

    CONFIG.seed_orig = CONFIG.seed
    CONFIG.init_from_orig = CONFIG.init_from

    run_name_components = [
        CONFIG.run_name,
        CONFIG.env,
        str(CONFIG.n_iter),
        str(bool(CONFIG.init_from)),
        str(CONFIG.dropout_keep),
        str(CONFIG.permanent_keep)]

    CONFIG.dpath_model_orig = os.path.join(CONFIG.dpath_model, '_'.join(run_name_components))

    if CONFIG.reset:
        shutil.rmtree(CONFIG.dpath_model, ignore_errors=True)


def proc_flags_with_seed():

    CONFIG.dpath_model = os.path.join(CONFIG.dpath_model_orig, str(CONFIG.seed))

    CONFIG.dpath_checkpoint = os.path.join(CONFIG.dpath_model, 'checkpoints')
    if not os.path.exists(CONFIG.dpath_checkpoint):
        os.makedirs(CONFIG.dpath_checkpoint)

    if CONFIG.init_from_orig:
        CONFIG.init_from = os.path.join(CONFIG.init_from_orig, str(CONFIG.seed), 'checkpoints')

    with open(os.path.join(CONFIG.dpath_model, 'config.txt'), 'w') as f:
        json.dump(vars(CONFIG), f, indent=4)


if __name__ == '__main__':

    parse_flags()
    proc_flags()

    if CONFIG.model == 'a2c':
        for e in range(CONFIG.n_experiments):
            CONFIG.seed = CONFIG.seed_orig + 10 * e

            proc_flags_with_seed()

            # Awkward hacky process runs, because Tensorflow does not like
            # repeatedly calling train in the same thread.
            p = Process(target=a2c.train, args=tuple())
            p.start()
            p.join()

    if not CONFIG.dont_combine:
        combine_events(CONFIG.dpath_model_orig)

    print(CONFIG.dpath_model_orig)
