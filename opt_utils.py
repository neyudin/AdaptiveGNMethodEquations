from oracles import eps, lim_val
import numpy as np
import time


phi_val = (1. + np.sqrt(5.)) / 2.


def psi(F, dF, x, L, tau, y):
    """
    Local model \psi_{x, L, \tau}(y) and \hat{\psi}_{x, L, \tau}(y, B) evaluated at point y.
    Parameters
    ----------
    F : array_like
        The system of equations evaluated at point x.
    dF : array_like
        The jacobian of system of equations evaluated at point x.
    x : array_like
        Anchor point for the local model.
    L : float
        The estimate of local Lipschitz constant.
    tau : float
        The hyperparameter of local model.
    y : array_like
        The evaluation point for local model.
    Returns
    -------
    float
        The value of local model evaluated at point y.
    """
    return tau / 2. + L * np.sum(np.square(y - x)) / 2. +\
        np.sum(np.square(F + np.dot(dF, y - x))) / (2. * tau)


def factor_step_probe(F, dF, dF2=None):
    """
    Factor computation of the next point in optimization procedure using spectral decomposition and
    Sherman-Morrison-Woodbury formula.
    Parameters
    ----------
    F : array_like
        The system of equations evaluated at point x.
    dF : array_like
        The jacobian of system of equations evaluated at point x.
    dF2 : array_like, default=None
        If not None, the doubly stochastic step is used and dF2 is tracted as independently
        sampled jacobian.
    Returns
    -------
    Tuple
        The tuple of factors for fast computation of the optimization step:
        Lambda, Q, ... and other factors.
        Lambda : array_like
            The diagonal matrix of eigenvalues of hessian-like matrix.
        Q : array_like
            The unitary matrix of eigenvectors for corresponding eigenvalues.
    """
    m, n = dF.shape
    if m > n:
        if dF2 is None:
            Lambda, Q = np.linalg.eigh(np.dot(dF.T, dF))
        else:
            Lambda, Q = np.linalg.eigh(np.dot(dF2.T, dF2))
        return Lambda, Q, np.dot(Q.T, np.dot(dF.T, F))
    if dF2 is None:
        Lambda, Q = np.linalg.eigh(np.dot(dF, dF.T))
        return Lambda, Q, Lambda * np.dot(Q.T, F)
    Lambda, Q = np.linalg.eigh(np.dot(dF2, dF2.T))
    return Lambda, Q, np.dot(dF.T, F), np.dot(dF2.T, Q), np.dot(Q.T, np.dot(dF2, np.dot(dF.T, F)))


def probe_x(x, eta, B, v):
    """
    Computation of the next point in optimization procedure: x - eta * B^{-1}v.
    Parameters
    ----------
    x : array_like
        Current optimizable point in the procedure.
    eta : float
        The step scale.
    B : array_like
        Hessian-like matrix evaluated at x.
    v : array_like
        Gradient of .5 * f_2(x) evaluated at x.
    Returns
    -------
    array_like
        The next optimizable point.
    """
    return x - eta * np.dot(np.linalg.inv(np.clip(B, a_min=-lim_val, a_max=lim_val)), v)


def fast_probe_x(x, eta, tauL, F, dF, Lambda, Q, factored_QF, dF2=None):
    """
    Computation of the next point in optimization procedure using spectral decomposition and
    Sherman-Morrison-Woodbury formula.
    Parameters
    ----------
    x : array_like
        Current optimizable point in the procedure.
    eta : float
        The step scale.
    tauL : float
        The value of \tau L.
    F : array_like
        The system of equations evaluated at point x.
    dF : array_like
        The jacobian of system of equations evaluated at point x.
    Lambda : array_like
        The diagonal matrix of eigenvalues of hessian-like matrix.
    Q : array_like
        The unitary matrix of eigenvectors for corresponding eigenvalues.
    factored_QF : tuple
        The tuple of matrices and vectors from factorization of computation of the next point.
    dF2 : array_like, default=None
        If not None, the doubly stochastic step is used and dF2 is tracted as independently
        sampled jacobian.
    Returns
    -------
    array_like
        The next optimizable point.
    """
    m, n = dF.shape
    if m > n:
        return x - eta * np.dot(Q, factored_QF[0] / np.clip(Lambda + tauL, a_min=eps, a_max=lim_val))
    if dF2 is None:
        return x - eta * np.dot(
            dF.T, F - np.dot(Q, factored_QF[0] / np.clip(Lambda + tauL, a_min=eps, a_max=lim_val))) /\
            np.clip(tauL, a_min=eps, a_max=lim_val)
    return x - eta * (
        factored_QF[0] - np.dot(
            factored_QF[1], factored_QF[2] / np.clip(Lambda + tauL, a_min=eps, a_max=lim_val))) /\
            np.clip(tauL, a_min=eps, a_max=lim_val)


def prox_l2_norm(w, lamb=1.):
    """!
    Compute the proximal operator of the \f$\ell_2\f$ - norm.
    Parameters
    ----------
    w : array_like
        Input vector.
    lamb : float, default=1.
        Penalty paramemeter.
    Returns
    -------
    array_like
       Output vector.
    """
    norm_w = np.linalg.norm(w, ord=2)
    return np.maximum(1. - lamb / norm_w, 0) * w


def prox_phi_conj(x, lbd=1.):
    """!
    Compute the proximal operator of the \f$\ell_2\f$ conjugate function.
    Parameters
    ----------
    x : array_like
        Input vector.
    lbd : float, default=1.
        Penalty paramemeter.
    Returns
    ----------
    array_like
       Output vector.
    """
    return x - lbd * prox_l2_norm(x / lbd, 1. / lbd)


def sub_adpgpd_solver(jac_est, func_est, x_til, prob_params, debug=False):
    """
    Accelerated Primal-Dual First-Order Solver to compute the next step in Gauss-Newton Method.
    Parameters
    ----------
    jac_est : array_like
        Jacobian (dF) estimate.
    func_est : array_like
        F-value estimate.
    x_til : array_like
        Current parameter value.
    prob_params : dict
        Method hyperparameters.
    debug : bool, default=False
        If equals True, the auxiliary debug output is printed.
    Returns
    ----------
    array_like
       Next optimizable parameter.
    """

    d_ = x_til.shape[0]
    p_ = func_est.shape[0]

    jac_est_t = jac_est.T

    M_const = prob_params.get('M_const', 1.)
    max_iter = prob_params.get('max_sub_iter', 100)

    L_const = np.linalg.norm(jac_est.dot(jac_est_t)) / M_const

    L_const_inv = 1. / L_const
    M_const_inv = 1. / M_const

    tau_cur = 1.

    u_cur = u_hat = np.zeros(p_)

    if debug:
        start_time = time.time()

    for k in range(max_iter):
        u_next = prox_phi_conj(u_hat - L_const_inv * (M_const_inv * jac_est.dot(jac_est_t.dot(u_hat)) - func_est), L_const_inv)

        tau_next = .5 * (1. + np.sqrt(1. + 4. * tau_cur * tau_cur))

        u_hat = u_next + ((tau_cur - 1.) / tau_next) * (u_next - u_cur)

        # Compute the solution change.
        abs_schg = np.linalg.norm(u_next - u_cur, ord=2);
        rel_schg = abs_schg / np.maximum(1., np.linalg.norm(u_cur, ord=2))

        if rel_schg <= prob_params['RelTolSoln']:
            if debug:
                print('Convergence achieved')
                print('The serarch direction norm and the feasibility gap is below the desired threshold')
            u_cur = u_next
            break

        if debug:
            print("Time: {:f}, SubProblem, Iter: {:5d}, Rel Sol Change: {:3.2e}".format(time.time() - start_time, k, rel_schg))

        # update
        u_cur = u_next
        tau_cur = tau_next

    return x_til - M_const_inv * jac_est.T.dot(u_cur)


def sub_fgm_solver(grad_func, tau, L=10000., qf=.01, N=1000):
    """!
    Accelerated Projected Gradient Method to compute optimal \f$\tau\f$.
    Parameters
    ----------
    grad_func : callable
        Gradient function of the optimized criterion.
    tau : float
        Initial \f$\tau\f$ value.
    L : float, default=10000.
        Lipschitz constant estimate for grad_func.
    qf : float, default=.01
        Inverse value of the condition number estimate for grad_func.
    N : int, default=1000
        Number of iterations to run the method.
    Returns
    ----------
    array_like
       Optimal \f$\tau\f$ value to perform the next step.
    """
    new_tau, tau_y = tau, tau
    for _ in range(N):
        grad = grad_func(tau_y)
        new_tau = max(tau_y - grad / L, eps)
        tau_y = max(new_tau + (1. - np.sqrt(qf)) / (1. + np.sqrt(qf)) * (new_tau - tau), eps)
        tau = new_tau
        if abs(grad) < eps:
            break
    return new_tau


def fast_diff_tau(tauL, F, dF, Lambda, Q, factored_QF, dF2=None):
    """!
    Auxiliary function to compute part of the derivative wrt. \f$\tau\f$.
    Parameters
    ----------
    tauL : float
        The value of \tau L.
    F : array_like
        The system of equations evaluated at point x.
    dF : array_like
        The jacobian of system of equations evaluated at point x.
    Lambda : array_like
        The diagonal matrix of eigenvalues of hessian-like matrix.
    Q : array_like
        The unitary matrix of eigenvectors for corresponding eigenvalues.
    factored_QF : array_like
        The tuple of matrices and vectors from factorization of computation of the next point.
    dF2 : array_like, default=None
        If not None, the doubly stochastic step is used and dF2 is tracted as independently
        sampled jacobian.
    Returns
    ----------
    array_like
       Output vector.
    """
    m, n = dF.shape
    if m > n:
        return np.dot(Q, factored_QF[0] / np.clip(Lambda + tauL, a_min=eps, a_max=lim_val))
    if dF2 is None:
        return np.dot(
            dF.T, F - np.dot(Q, factored_QF[0] / np.clip(Lambda + tauL, a_min=eps, a_max=lim_val))) /\
            np.clip(tauL, a_min=eps, a_max=lim_val)
    return (
        factored_QF[0] - np.dot(
            factored_QF[1], factored_QF[2] / np.clip(Lambda + tauL, a_min=eps, a_max=lim_val))) /\
            np.clip(tauL, a_min=eps, a_max=lim_val)


def psi_grad(F, dF, L, tau, Lambda, Q, factored_QF):
    """!
    Auxiliary function to compute the upper bound derivative wrt. \f$\tau\f$ with the next step applied.
    Parameters
    ----------
    F : array_like
        The system of equations evaluated at point x.
    dF : array_like
        The jacobian of system of equations evaluated at point x.
    L : float
        The estimate of local Lipschitz constant.
    tau : float
        The hyperparameter of local model.
    Lambda : array_like
        The diagonal matrix of eigenvalues of hessian-like matrix.
    Q : array_like
        The unitary matrix of eigenvectors for corresponding eigenvalues.
    factored_QF : array_like
        The tuple of matrices and vectors from factorization of computation of the next point.
    Returns
    ----------
    float
        Derivative wrt. \f$\tau\f$.
    """
    double_tau = 2. * tau
    double_tau_squared = double_tau * tau
    diff = fast_diff_tau(tau * L, F, dF, Lambda, Q, factored_QF)
    return .5 - np.sum(np.square(F)) / double_tau_squared + np.dot(diff, np.dot(dF.T, F)) / double_tau_squared + L * np.dot(diff, diff) / double_tau

