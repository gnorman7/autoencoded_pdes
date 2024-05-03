
from typing import Callable
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#%% Evaluation Metrics
def mmd2(X: torch.Tensor, Y: torch.Tensor, kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Computes the MMD^2 between two sets of samples X and Y.

    Args:
        X: Samples from distribution P (n_x, d)
        Y: Samples from distribution Q (n_y, d)
        kernel: Kernel function ( (n,d), (n,d) ) -> (n,n)

    Returns:
        mmd2: empirical estimate of MMD^2 between X and Y, (this can be negative)
    """
    n_x = X.shape[0]
    n_y = Y.shape[0]
    K_XX = kernel(X, X)
    K_YY = kernel(Y, Y)
    K_XY = kernel(X, Y)
    xx_term = (torch.sum(K_XX) - torch.trace(K_XX)) / (n_x * (n_x - 1))
    yy_term = (torch.sum(K_YY) - torch.trace(K_YY)) / (n_y * (n_y - 1))
    xy_term = torch.sum(K_XY) / (n_x * n_y)
    mmd2 = xx_term + yy_term - 2 * xy_term
    return mmd2

def mmd2_rq_mix(X: torch.Tensor, Y: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    """Computes the MMD^2 between two sets of samples X and Y using a mixture of Rational Quadratic kernels.

    Args:
        X: Samples from distribution P (n_x, d)
        Y: Samples from distribution Q (n_y, d)
        alphas: Collection of parameters for the mixture of kernels
    Returns:
        mmd2: empirical estimate of MMD^2 between X and Y, (this can be negative)
    """
    kernel = lambda X, Y: mixture_rq_kernel(X, Y, alphas)
    return mmd2(X, Y, kernel)

def rq_kernel(X: torch.Tensor, Y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Computes the Rational Quadratic kernel between two sets of samples X and Y.
    k(x, y) = (1 + ||x - y||^2 / (2 * alpha)) ^ (-alpha)

    Args:
        X: Samples from distribution P (n_x, d)
        Y: Samples from distribution Q (n_y, d)
        alpha: Alpha parameter for the kernel (default: 1.0)

    Returns:
        K: Kernel matrix (n_x, n_y)
    """
    n_x = X.shape[0]
    n_y = Y.shape[0]
    # do not use repeat, it is not memory efficient, expand and view do not copy
    big_X = X.unsqueeze(1).expand(n_x, n_y, X.shape[1])
    big_Y = Y.unsqueeze(0).expand(n_x, n_y, Y.shape[1])

    dist = torch.sum((big_X - big_Y) ** 2, dim=2)
    K = (1 + dist / (2 * alpha)) ** (-alpha)
    return K

def mixture_rq_kernel(X: torch.Tensor, Y: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    """Computes the mixture of Rational Quadratic kernels between two sets of samples X and Y.
    k(x, y) = sum_i alpha_i * k_i(x, y)

    Args:
        X: Samples from distribution P (n_x, d)
        Y: Samples from distribution Q (n_y, d)
        alphas: Collection of parameters for the kernels (n_k,)

    Returns:
        K: Kernel matrix (n_x, n_y)
    """
    n_k = len(alphas)
    K = torch.zeros(X.shape[0], Y.shape[0])
    for i in range(n_k):
        K += rq_kernel(X, Y, alpha=alphas[i])
    return K



#%% Solvers
def newton_solve(f: Callable[[torch.Tensor], torch.Tensor], x0: torch.Tensor, max_iter: int = 10,
                 res_tol: float = 1e-6, x_tol: float = 1e-6, jac_stb_term = 0.0,
                 damp=1.0, verbose: bool = False, debug: bool = False):
    """Solves a nonlinear system through Newton's method, with functorch autodiff.

    Args:
        f: function to find zero of, (n,) -> (n,)
        x0: initial guess, (n,)
        max_iter: maximum number of iterations (default: 10)
        res_tol: tolerance for residual (default: 1e-6)
        x_tol: tolerance for change in x (default: 1e-6)
        jac_stb_term: stability term added to jacobian as identity (default: 0.0)
        damp: damping factor for step (default: 1.0)
        verbose: print exit iterations and reason (default: False)
        debug: print intermediate values and store convergence (default: False)

    Returns:
        x: solution to f(x) = 0, (n,1)
    """
    x = x0
    f_grad = lambda x: torch.func.jacrev(f)(x)
    exit_reason = 'max_iter'
    if debug:
        res_hist = []
        x_hist = []
    for i in range(max_iter):
        f_val = f(x)
        res = torch.linalg.norm(f_val)
        if debug:
            res_hist.append(res)
        if res < res_tol:
            exit_reason = 'res_tol'
            break
        jac_f = f_grad(x)
        jac_f = jac_f + jac_stb_term * torch.eye(jac_f.shape[0])
        delta_x = -torch.linalg.solve(jac_f, f_val)
        x = x + damp * delta_x
        if debug:
            x_hist.append(x)
        if torch.linalg.norm(delta_x) < x_tol:
            exit_reason = 'x_tol'
            break

    if verbose:
        print(f'Exited after: {i+1} / {max_iter} iterations, reason: {exit_reason}')
    if debug:
        with torch.no_grad():
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.semilogy(res_hist)
            plt.title('Residual')
            plt.subplot(1, 2, 2)
            for i in range(len(x_hist)):
                # color based on iteration too, use magma
                cmap = plt.get_cmap('magma')
                color = cmap(i / len(x_hist))
                plt.plot(x_hist[i], color=color)
                plt.title('x')
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(x_hist)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ticks=[0, len(x_hist)], ax = plt.gca())
            cbar.set_label('More Iterations -->')



    return x

#%% Basic Finite Differences
def fd_centered(u: torch.Tensor, x: torch.Tensor):
    """
    Equispaced centered finite difference of `u` with respect to `x`.

    Args:
        u: Function values (n_x,)
        x: Grid points (n_x,)

    Returns:
        ux: Finite difference (n_x-2,)
    """
    dx = x[1] - x[0]
    ux = (u[2:] - u[:-2]) / (2.0 * dx)
    return ux

def fd_centered_2nd(u: torch.Tensor, x: torch.Tensor):
    """
    Equispaced centered 2nd order finite difference of `u` with respect to `x`.

    Args:
        u: Function values (n_x,)
        x: Grid points (n_x,)

    Returns:
        uxx: Finite difference (n_x-2,)
    """
    dx = x[1] - x[0]
    uxx = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx ** 2)
    return uxx
