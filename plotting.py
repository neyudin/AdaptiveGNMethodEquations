import numpy as np
import matplotlib.pyplot as plt
import warnings
from oracles import eps


warnings.filterwarnings('ignore', category=FutureWarning)


import seaborn as sns


"""
-----------------------------------------------------------------
Mapping between filename and figure from the supplement document 
-----------------------------------------------------------------
                    filename                      | figure name  
-----------------------------------------------------------------
comparison.eps                                    |   Figure 1   
-----------------------------------------------------------------
"""


def plot_experiments_results(exp_res_dict, args):
    """
    Plotting routine which draws results of the whole experiment set.
    Parameters
    ----------
    exp_res_dict : dict
        The whole infographics of the experiments.
    args : populated namespace object from ArgumentParser
        The system of equations evaluated at point x.
    Returns
    -------
    None
    """
    
    sns.set(font_scale=1.5)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), sharex=False, sharey=False)
    legend_flag = False
    for col, (name, oracle_name) in enumerate(zip(['Rosenbrock-Skokov', 'Hat'], [r'Equation $F_{R}(x) = \mathbf{0}_{m}$', r'Equation $F_{H}(x) = \mathbf{0}_{m}$'])):
        for row, stat_name in enumerate(['nabla_f_2_norm_vals', 'f_vals']):
            for mark_period, method_name, gnm_type, c, marker in zip([2, 4, 8], [r'three squares ($\tau_{k} = f_{1}(x_{k})$)', r'adaptive ($\tau_{k}$ optimal)', r'Gauss-Newton ($\tau_{k} = \phi(x_{k}, y)$)'],
                                    ['DetGNM', 'AdaptiveDetGNM', 'ClassicalDetGNM'], ['b', 'g', 'r'], ['o', '^', 'v']):
                data_sums = []
                data_sizes = []
                for iter_counter in range(args.N_iter):
                    for i in range(args.n_starts):
                        if iter_counter < len(exp_res_dict[gnm_type][name][args.n_dims[-1]][i][stat_name]):
                            if iter_counter >= len(data_sums):
                                data_sums.append(0.)
                                data_sizes.append(0)
                            data_sums[iter_counter] += exp_res_dict[gnm_type][name][args.n_dims[-1]][i][stat_name][iter_counter]
                            data_sizes[iter_counter] += 1
                data_sizes = np.array(data_sizes)
                data_means = np.array(data_sums) / data_sizes
                label = '{}'.format(method_name)
                axes[row, col].plot(np.arange(1, data_means.size + 1), data_means, color=c, marker=marker, markevery=mark_period * max(1, data_means.size // 10), markersize=15,
                                    linewidth=3, ls='--', label=label)
            axes[row, col].set_yscale('log')
            if col == 0:
                if stat_name == 'nabla_f_2_norm_vals':
                    axes[row, col].set_ylabel(r'$\|\nabla f_{2}(x_{k})\|$', fontsize=18)
                else:
                    axes[row, col].set_ylabel(r'$f_{1}(x_{k})$', fontsize=18)
            if row == 1:
                axes[row, col].set_xlabel(r'Number of outer iterations, $k$', fontsize=18)
            axes[row, col].set_title(oracle_name, fontsize=18)
            axes[row, col].axhline(y=eps, color='r', linestyle='-', linewidth=3)
            if not legend_flag:
                legend_flag = True
                handles, labels = axes[row, col].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1., .88), title="method:", fontsize=16)
    plt.savefig(fname=args.store_dir + '/comparison.eps')
    plt.close(fig)
    
    return None

