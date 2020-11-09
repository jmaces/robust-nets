import itertools
import time

import numpy as np
import torch

from tqdm import tqdm

from operators import CGInverterLayer, l2_error, shrink


# ----- Iterative reconstruction algorithms -----


def admm_l1_rec_diag(
    y, OpA, OpW, x0, z0, lam, rho, iter=20, silent=False,
):
    """ ADMM for least squares solve with L1 regularization.

    Reconstruction algorithm for the inverse problem y = Ax + e under the
    analysis model z=Wx for sparse analysis coefficients z.

    min ||Ax - y||^2_2 + lambda ||Wx||_1

    Note: it assumes that A'*A is diagonalizable in k-space.

    Parameters
    ----------
    y : torch.Tensor
        The measurement vector.
    OpA : operators.LinearOperator
        The measurement operator, providing A and A'.
    OpW : operators.LinearOperator
        The analysis operator, providing W and W'.
    x0 : torchTensor
        Initial guess for the signal, typically torch.zeros(...) of
        appropriate size.
    z0 : torchTensor
        Initial guess for the coefficients, typically torch.zeros(...) of
        appropriate size.
    lam : float
        The regularization parameter lambda for the sparsity constraint.
    rho : float
        The Lagrangian augmentation parameter for the ADMM algorithm.
    iter : int, optional
        Number of ADMM iterations. (Default 20)
    silent : bool, optional
        Disable progress bar. (Default False)

    Returns
    -------
    tensor
       The recovered signal x.
    """

    # init iteration variables
    z = z0.clone()
    x = x0.clone()
    u = torch.zeros_like(z0)
    tv_kernel = OpW.get_fourier_kernel()

    # run main ADMM iterations
    t = tqdm(range(iter), desc="ADMM iterations", disable=silent)
    for it in t:
        # ADMM step 1) : signal update
        rhs = OpA.adj(y) + rho * OpW.adj(z - u)
        x = OpA.tikh(rhs, tv_kernel, rho)

        # ADMM step 2) : coefficient update
        zold, z = z, shrink(OpW(x) + u, lam / rho)

        # ADMM step 3 : dual variable update
        u = u + OpW(x) - z

        # evaluate
        with torch.no_grad():
            loss = (
                0.5 * (OpA(x) - y).pow(2).sum(dim=(-1, -2))
                + lam * OpW(x).abs().sum((-1, -2))
            ).mean()
            primal_residual = OpW(x) - z
            dual_residual = rho * OpW.adj(zold - z)

            t.set_postfix(
                loss=loss.item(),
                pres=torch.norm(primal_residual).item(),
                dres=torch.norm(dual_residual).item(),
            )

    return x, z


def admm_l1_rec(
    y, OpA, OpW, x0, z0, lam, rho, iter=20, silent=False, timeout=None,
):
    """ ADMM for least squares solve with L1 regularization.

    Reconstruction algorithm for the inverse problem y = Ax + e under the
    analysis model z=Wx for sparse analysis coefficients z.

    min ||Ax - y||^2_2 + lambda ||Wx||_1

    Parameters
    ----------
    y : torch.Tensor
        The measurement vector.
    OpA : operators.LinearOperator
        The measurement operator, providing A and A'.
    OpW : operators.LinearOperator
        The analysis operator, providing W and W'.
    x0 : torchTensor
        Initial guess for the signal, typically torch.zeros(...) of
        appropriate size.
    z0 : torchTensor
        Initial guess for the coefficients, typically torch.zeros(...) of
        appropriate size.
    lam : float
        The regularization parameter lambda for the sparsity constraint.
    rho : float
        The Lagrangian augmentation parameter for the ADMM algorithm.
    iter : int, optional
        Number of ADMM iterations. (Default 20)
    silent : bool, optional
        Disable progress bar. (Default False)
    timeout : int, optional
        Set runtime limit in seconds. (Default None)

    Returns
    -------
    tensor
       The recovered signal x.
    """

    # init iteration variables
    z = z0.clone()
    x = x0.clone()
    u = torch.zeros_like(z0)

    # prepare conjugate gradient inversion
    inverter = CGInverterLayer(
        x.shape[1:],
        lambda x: OpA.adj(OpA(x)) + rho * OpW.adj(OpW(x)),
        rtol=1e-6,
        atol=0.0,
        maxiter=200,
    )

    if timeout is not None:
        start_time = time.time()

    # run main ADMM iterations
    t = tqdm(range(iter), desc="ADMM iterations", disable=silent)
    for it in t:

        if timeout is not None:
            if time.time() > start_time + timeout:
                print("ADMM aborted due to timeout")
                return x, z

        # ADMM step 1) : signal update
        rhs = OpA.adj(y) + rho * OpW.adj(z - u)
        x = inverter(rhs, x)

        # ADMM step 2) : coefficient update
        zold, z = z, shrink(OpW(x) + u, lam / rho)

        # ADMM step 3 : dual variable update
        u = u + OpW(x) - z

        # evaluate
        with torch.no_grad():
            loss = (
                0.5 * (OpA(x) - y).pow(2).sum(dim=(-1, -2))
                + lam * OpW(x).abs().sum((-1, -2))
            ).mean()
            primal_residual = OpW(x) - z
            dual_residual = rho * OpW.adj(zold - z)

            t.set_postfix(
                loss=loss.item(),
                pres=torch.norm(primal_residual).item(),
                dres=torch.norm(dual_residual).item(),
            )

    return x, z


# ----- Utility -----


def grid_search(x, y, rec_func, grid):
    """ Grid search utility for tuning hyper-parameters. """
    err_min = np.inf
    grid_param = None

    grid_shape = [len(val) for val in grid.values()]
    err = torch.zeros(grid_shape)

    for grid_val, nidx in zip(
        itertools.product(*grid.values()), np.ndindex(*grid_shape)
    ):
        grid_param_cur = dict(zip(grid.keys(), grid_val))
        print(
            "Current grid parameters ("
            + str([cidx + 1 for cidx in nidx])
            + " / "
            + str(grid_shape)
            + "): "
            + str(grid_param_cur)
        )
        x_rec = rec_func(y, **grid_param_cur)
        err[nidx], _ = l2_error(x_rec, x, relative=True, squared=False)
        print("Rel. recovery error: {:1.2e}".format(err[nidx]), flush=True)
        if err[nidx] < err_min:
            grid_param = grid_param_cur
            err_min = err[nidx]

    return grid_param, err_min, err
