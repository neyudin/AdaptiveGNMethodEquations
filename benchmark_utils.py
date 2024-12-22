from optimizers import *
import gc
import time


def experiment_runner(args, x_0_dict):
    """
    Runner routine which performs the whole experiment set.
    Parameters
    ----------
    args : populated namespace object from ArgumentParser
        The system of equations evaluated at point x.
    x_0_dict : dict
        The dictionary of initial points x.
    Returns
    -------
    dict
        Aggregated experiment data.
    """
    gc.enable()
    gc.collect()
    
    exp_res_dict = dict()
    
    if args.verbose:
        print("Started DetGNM!")
    exp_res_dict['DetGNM'] = dict()
    for oracle_class, name in [(RosenbrockEvenSumOracle, 'Rosenbrock-Skokov'), (HatOracle, 'Hat')]:
        if args.verbose:
            print('Oracle:', name)
        exp_res_dict['DetGNM'][name] = dict()
        for n in args.n_dims:
            if args.verbose:
                print('    n:', n)
            exp_res_dict['DetGNM'][name][n] = dict()
            for i in range(args.n_starts):
                if args.verbose:
                    print('        start #:', i + 1)
                start = time.time()
                x, f_vals, nabla_f_2_norm_vals, _, _ = DetGNM(oracle_class(n), args.N_iter, x_0_dict[n][i], args.L_0, True, None)
                start = time.time() - start
                
                if name == 'Rosenbrock-Skokov':
                    rel_err = np.max(np.abs(x - np.ones_like(x)) / np.abs(x_0_dict[n][i] - np.ones_like(x_0_dict[n][i])))
                elif name == 'Hat':
                    rel_err = np.max(np.abs(x - x / np.linalg.norm(x)) / np.abs(x_0_dict[n][i] - x_0_dict[n][i] / np.linalg.norm(x_0_dict[n][i])))
                if args.verbose:
                    print('            relative solution error is {:.16f}'.format(rel_err))
                
                exp_res_dict['DetGNM'][name][n][i] = {'f_vals': f_vals, 'nabla_f_2_norm_vals': nabla_f_2_norm_vals, 'avg_time_s': start / len(f_vals), 'time_s': start, 'rel_err': rel_err}
                del _, f_vals, nabla_f_2_norm_vals, start
                gc.collect()
    
    if args.verbose:
        print("Started AdaptiveDetGNM!")
    exp_res_dict['AdaptiveDetGNM'] = dict()
    for oracle_class, name in [(RosenbrockEvenSumOracle, 'Rosenbrock-Skokov'), (HatOracle, 'Hat')]:
        if args.verbose:
            print('Oracle:', name)
        exp_res_dict['AdaptiveDetGNM'][name] = dict()
        for n in args.n_dims:
            if args.verbose:
                print('    n:', n)
            exp_res_dict['AdaptiveDetGNM'][name][n] = dict()
            for i in range(args.n_starts):
                if args.verbose:
                    print('        start #:', i + 1)
                start = time.time()
                x, f_vals, nabla_f_2_norm_vals, _, _ = DetGNM(oracle_class(n), args.N_iter, x_0_dict[n][i], args.L_0, True, None, "AdaptiveThreeSquares")
                
                if name == 'Rosenbrock-Skokov':
                    rel_err = np.max(np.abs(x - np.ones_like(x)) / np.abs(x_0_dict[n][i] - np.ones_like(x_0_dict[n][i])))
                elif name == 'Hat':
                    rel_err = np.max(np.abs(x - x / np.linalg.norm(x)) / np.abs(x_0_dict[n][i] - x_0_dict[n][i] / np.linalg.norm(x_0_dict[n][i])))
                if args.verbose:
                    print('            relative solution error is {:.16f}'.format(rel_err))
                
                start = time.time() - start
                exp_res_dict['AdaptiveDetGNM'][name][n][i] = {'f_vals': f_vals, 'nabla_f_2_norm_vals': nabla_f_2_norm_vals, 'avg_time_s': start / len(f_vals), 'time_s': start, 'rel_err': rel_err}
                del _, f_vals, nabla_f_2_norm_vals, start
                gc.collect()
    
    if args.verbose:
        print("Started ClassicalDetGNM!")
    exp_res_dict['ClassicalDetGNM'] = dict()
    for oracle_class, name in [(RosenbrockEvenSumOracle, 'Rosenbrock-Skokov'), (HatOracle, 'Hat')]:
        if args.verbose:
            print('Oracle:', name)
        exp_res_dict['ClassicalDetGNM'][name] = dict()
        for n in args.n_dims:
            if args.verbose:
                print('    n:', n)
            exp_res_dict['ClassicalDetGNM'][name][n] = dict()
            for i in range(args.n_starts):
                if args.verbose:
                    print('        start #:', i + 1)
                start = time.time()
                x, f_vals, nabla_f_2_norm_vals, _, _ = DetGNM(oracle_class(n), args.N_iter, x_0_dict[n][i], args.L_0, True, None, "GaussNewton")
                
                if name == 'Rosenbrock-Skokov':
                    rel_err = np.max(np.abs(x - np.ones_like(x)) / np.abs(x_0_dict[n][i] - np.ones_like(x_0_dict[n][i])))
                elif name == 'Hat':
                    rel_err = np.max(np.abs(x - x / np.linalg.norm(x)) / np.abs(x_0_dict[n][i] - x_0_dict[n][i] / np.linalg.norm(x_0_dict[n][i])))
                if args.verbose:
                    print('            relative solution error is {:.16f}'.format(rel_err))
                
                start = time.time() - start
                exp_res_dict['ClassicalDetGNM'][name][n][i] = {'f_vals': f_vals, 'nabla_f_2_norm_vals': nabla_f_2_norm_vals, 'avg_time_s': start / len(f_vals), 'time_s': start, 'rel_err': rel_err}
                del _, f_vals, nabla_f_2_norm_vals, start
                gc.collect()
    
    return exp_res_dict

