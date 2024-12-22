from opt_utils import *
from oracles import HatOracle, RosenbrockEvenSumOracle, lim_val, eps


def DetGNM(oracle, N, x_0, L_0, fast_update=True, tau_const=None, method_type="ThreeSquares"):
    """
    Find argminimum of f_1 using the deterministic Gauss-Newton method with exact proximal map and
    \tau_k = \hat{f}_1(x_k).
    Parameters
    ----------
    oracle : Oracle class instance
        Oracle of the optimization criterion.
    N : int
        The number of outer iterations.
    x_0 : array_like
        The initial parameter value.
    L_0 : float
        The initial value of local Lipschitz constant.
    fast_update : bool, default=True
        If true, every step is computed using the factor_step_probe and fast_probe_x functions,
        otherwise only probe_x is used.
    tau_const : float, default=None
        If not None, then the constant value is used for tau equal tau_const.
    method_type : str, default="ThreeSquares"
        A variant of Gauss-Newton method to execute. Possible values: "ThreeSquares", "AdaptiveThreeSquares", "GaussNewton".
    Returns
    -------
    x : array_like
        The approximated argminimum.
    f_vals : array_like
        The list of \hat{f}_1(x_k) values at each iteration.
    nabla_f_2_norm_vals : array_like
        The list of \|\nabla\hat{f}_2(x_k)\| values at each iteration.
    nabla_f_2_vals : array_like
        The list of \nabla\hat{f}_2(x_k) values at each iteration.
    n_inner_iters : array_like
        The list of numbers of inner iterations per each outer one.
    """
    f_vals, nabla_f_2_norm_vals, nabla_f_2_vals, n_inner_iters = [], [], [], []
    x = x_0.copy()
    L = L_0
    tau = oracle.f_1(x)
    tmp_tau = tau
    for i in range(N):
        tau = oracle.f_1(x) if tau_const is None else tau_const
        tmp_tau = tau
        
        if tau < eps:
            break
        F = oracle.F(x)
        dF = oracle.dF(x)
        
        if method_type == "ThreeSquares":
            if fast_update:
                Lambda, Q, *factored_QF = factor_step_probe(F, dF)
                tmp_x = fast_probe_x(x, 1., tau * L, F, dF, Lambda, Q, factored_QF)
            else:
                dFTdF = np.dot(dF.T, dF)
                v = np.dot(dF.T, F)
                try:
                    tmp_x = probe_x(x, 1., dFTdF + tau * L * np.eye(x.size), v)
                except np.linalg.LinAlgError as err:
                    print('Singular matrix encountered: {}!'.format(str(err)))
                    tmp_x = probe_x(x, 1., tau * L * np.eye(x.size), v)
        elif method_type == "AdaptiveThreeSquares":
            Lambda, Q, *factored_QF = factor_step_probe(F, dF)
            tmp_tau = sub_fgm_solver(lambda y: psi_grad(F, dF, L, y, Lambda, Q, factored_QF), tau, N=N)
            tmp_x = fast_probe_x(x, 1., tmp_tau * L, F, dF, Lambda, Q, factored_QF)
        else:
            tmp_x = sub_adpgpd_solver(dF, F, x, {"M_const": L, "max_sub_iter": N, "RelTolSoln": eps})
            tau = np.linalg.norm(F + np.dot(dF, tmp_x - x))
            tmp_tau = tau
        
        n = 1
        while oracle.f_1(tmp_x) > psi(F, dF, x, L, tmp_tau, tmp_x):
            L *= 2.
            
            if method_type == "ThreeSquares":
                if fast_update:
                    tmp_x = fast_probe_x(x, 1., tau * L, F, dF, Lambda, Q, factored_QF)
                else:
                    try:
                        tmp_x = probe_x(x, 1., dFTdF + tau * L * np.eye(x.size), v)
                    except np.linalg.LinAlgError as err:
                        print('Singular matrix encountered: {}!'.format(str(err)))
                        tmp_x = probe_x(x, 1., tau * L * np.eye(x.size), v)
            elif method_type == "AdaptiveThreeSquares":
                Lambda, Q, *factored_QF = factor_step_probe(F, dF)
                tmp_tau = sub_fgm_solver(lambda y: psi_grad(F, dF, L, y, Lambda, Q, factored_QF), tau, N=N)
                tmp_x = fast_probe_x(x, 1., tmp_tau * L, F, dF, Lambda, Q, factored_QF)
            else:
                tmp_x = sub_adpgpd_solver(dF, F, x, {"M_const": L, "max_sub_iter": N, "RelTolSoln": eps})
                tau = np.linalg.norm(F + np.dot(dF, tmp_x - x))
                tmp_tau = tau
            
            n += 1
        L = max(L / 2., L_0)
        x = tmp_x.copy()
        if method_type == "AdaptiveThreeSquares":
            tau = tmp_tau
        
        f_vals.append(oracle.f_1(x))
        nabla_f_2_vals.append(oracle.nabla_f_2(x))
        nabla_f_2_norm_vals.append(np.linalg.norm(nabla_f_2_vals[-1]))
        n_inner_iters.append(n)
        if nabla_f_2_norm_vals[-1] < eps:
            break
    
    return x, f_vals, nabla_f_2_norm_vals, nabla_f_2_vals, n_inner_iters

