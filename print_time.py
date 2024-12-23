import numpy as np


def time_printer(exp_res_dict, args):
    """
    Routine to register time measurements of the whole experiment set.
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
    f = open(args.store_dir + '/' + args.time_stats_fname, 'w')
    
    print("DetGNM!", file=f)
    for name in ['Rosenbrock-Skokov', 'Hat']:
        print('Oracle:', name, file=f)
        for n in args.n_dims:
            print('    n:', n, file=f)
            avg_mean_time, avg_mean_sqr_time, mean_time, mean_sqr_time = 0., 0., 0., 0.
            for i in range(args.n_starts):
                avg_mean_time += exp_res_dict['DetGNM'][name][n][i]["avg_time_s"]
                avg_mean_sqr_time += exp_res_dict['DetGNM'][name][n][i]["avg_time_s"] ** 2
                mean_time += exp_res_dict['DetGNM'][name][n][i]["time_s"]
                mean_sqr_time += exp_res_dict['DetGNM'][name][n][i]["time_s"] ** 2
            avg_mean_time /= args.n_starts
            avg_mean_sqr_time /= args.n_starts
            mean_time /= args.n_starts
            mean_sqr_time /= args.n_starts
            avg_std = np.sqrt((avg_mean_sqr_time - avg_mean_time ** 2) * args.n_starts / (args.n_starts - 1))
            std = np.sqrt((mean_sqr_time - mean_time ** 2) * args.n_starts / (args.n_starts - 1))
            print('        avg mean time {:.4f} s. +- {:.4f} std s; mean time {:.4f} s. +- {:.4f} std s.'.format(avg_mean_time, avg_std, mean_time, std), file=f)
    
    print("AdaptiveDetGNM!", file=f)
    for name in ['Rosenbrock-Skokov', 'Hat']:
        print('Oracle:', name, file=f)
        for n in args.n_dims:
            print('    n:', n, file=f)
            avg_mean_time, avg_mean_sqr_time, mean_time, mean_sqr_time = 0., 0., 0., 0.
            for i in range(args.n_starts):
                avg_mean_time += exp_res_dict['AdaptiveDetGNM'][name][n][i]["avg_time_s"]
                avg_mean_sqr_time += exp_res_dict['AdaptiveDetGNM'][name][n][i]["avg_time_s"] ** 2
                mean_time += exp_res_dict['AdaptiveDetGNM'][name][n][i]["time_s"]
                mean_sqr_time += exp_res_dict['AdaptiveDetGNM'][name][n][i]["time_s"] ** 2
            avg_mean_time /= args.n_starts
            avg_mean_sqr_time /= args.n_starts
            mean_time /= args.n_starts
            mean_sqr_time /= args.n_starts
            avg_std = np.sqrt((avg_mean_sqr_time - avg_mean_time ** 2) * args.n_starts / (args.n_starts - 1))
            std = np.sqrt((mean_sqr_time - mean_time ** 2) * args.n_starts / (args.n_starts - 1))
            print('        avg mean time {:.4f} s. +- {:.4f} std s; mean time {:.4f} s. +- {:.4f} std s.'.format(avg_mean_time, avg_std, mean_time, std), file=f)
    
    print("ClassicalDetGNM!", file=f)
    for name in ['Rosenbrock-Skokov', 'Hat']:
        print('Oracle:', name, file=f)
        for n in args.n_dims:
            print('    n:', n, file=f)
            avg_mean_time, avg_mean_sqr_time, mean_time, mean_sqr_time = 0., 0., 0., 0.
            for i in range(args.n_starts):
                avg_mean_time += exp_res_dict['ClassicalDetGNM'][name][n][i]["avg_time_s"]
                avg_mean_sqr_time += exp_res_dict['ClassicalDetGNM'][name][n][i]["avg_time_s"] ** 2
                mean_time += exp_res_dict['ClassicalDetGNM'][name][n][i]["time_s"]
                mean_sqr_time += exp_res_dict['ClassicalDetGNM'][name][n][i]["time_s"] ** 2
            avg_mean_time /= args.n_starts
            avg_mean_sqr_time /= args.n_starts
            mean_time /= args.n_starts
            mean_sqr_time /= args.n_starts
            avg_std = np.sqrt((avg_mean_sqr_time - avg_mean_time ** 2) * args.n_starts / (args.n_starts - 1))
            std = np.sqrt((mean_sqr_time - mean_time ** 2) * args.n_starts / (args.n_starts - 1))
            print('        avg mean time {:.4f} s. +- {:.4f} std s; mean time {:.4f} s. +- {:.4f} std s.'.format(avg_mean_time, avg_std, mean_time, std), file=f)

    return None

