from benchmark_utils import experiment_runner
from plotting import plot_experiments_results
from print_time import time_printer
import numpy as np

from pathlib import Path
import argparse
import time
import pickle as pkl


class Store_as_array(argparse._StoreAction):
    """
    Helper class to convert data stream into numpy array.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)


parser = argparse.ArgumentParser('Flex GNM')
parser.add_argument('--N_iter', type=int, default=1000, help='The number of iterations to run Gauss-Newton method.')
parser.add_argument('--seed', type=int, default=617, help='The random seed.')
parser.add_argument(
    '--n_starts', type=int, default=5, help='The number of random samples for each combination of hyperparameters.')
parser.add_argument(
    '--verbose', type=bool, default=False, help='Whether to print auxiliary messages throughout the whole experiment.')
parser.add_argument(
    '--store_dir', type=str, default='./figures', help="The directory to store experiments' results.")
parser.add_argument(
    '--n_dims', action=Store_as_array, type=int, nargs='+', default=np.array([100]),
    help='The list of numbers of parameters.')
parser.add_argument('--L_0', type=float, default=1.,
                    help='Initial estimate of the local Lipschitz constant for \psi local model.')
parser.add_argument(
    '--time_stats_fname', type=str, default='time.txt', help='Name of the file with time measurements statistics')


args = parser.parse_args()


if __name__ == '__main__':
    start = time.time() # Start time measurement for experiments.
    Path(args.store_dir).mkdir(parents=True, exist_ok=True) # Create directory store_dir if it does not exist.
    
    np.random.seed(args.seed) # The random seed specification for reproducibility.
    x_0_dict = {n: np.random.randn(args.n_starts, n) - 7. for n in args.n_dims} # The dictionary of the initial values of parameters.
    
    for n in x_0_dict.keys():
        for i in range(x_0_dict[n].shape[0]):
            for x_i in x_0_dict[n][i]:
                assert x_i < -3.
    print('Initialization is consistent!')
    
    exp_res_dict = experiment_runner(args, x_0_dict) # Run experiments.
    pkl.dump(exp_res_dict, open(args.store_dir + '/flex_gnm_experiments_results.pkl', 'wb')) # Save infographics.
    
    plot_experiments_results(exp_res_dict, args) # Plot results.
    # plot_experiments_results(pkl.load(open(args.store_dir + '/flex_gnm_experiments_results.pkl', 'rb')), args) # Plot results.
    
    time_printer(exp_res_dict, args) # Save timer stats.
    
    start = time.time() - start # End time measurement for experiments.
    print(
        'Elapsed runtime: {} day(s), {} hour(s), {} minute(s), {} second(s)'.format(
            int(start // 86400), int(start // 3600 % 24), int(start // 60 % 60), int(start % 60)))

