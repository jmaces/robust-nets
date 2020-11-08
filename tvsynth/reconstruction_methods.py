import torch

from tqdm import tqdm

from operators import prox_l2_constraint_conjugate, shrink


# ----- Iterative reconstruction algorithms -----


def primaldual(
    y,
    OpA,
    OpW,
    c0,
    eta,
    y0=None,
    iter=20,
    sigma=0.5,
    tau=0.5,
    theta=1.0,
    silent=False,
    report_pd_gap=False,
):
    """ Primal Dual algorithm.

    Reconstruction algorithm for the inverse problem y = Ax + e under the
    synthesis model x=Wc for some sparse coefficients c and ||e||_2 <= eta.

    min ||c||_1     s.t.    ||AWc - y||_2 <= eta

    Basic iteration steps are

    1) y_ = prox_l2_constraint_conjugate(y_+sig*AWc_, sigma*y, sigma*eta)
    2) c = shrink(c-tau*W'A'y_, tau)
    3) c_ = c + theta*(c - cold)


    Parameters
    ----------
    y : torch.Tensor
        The measurement vector.
    OpA : operators.LinearOperator
        The measurement operator, providing A and A'.
    OpW : operators.LinearOperator
        The synthesis operator, providing W and W'.
    c0 : torchTensor
        Initial guess for the coefficients, typically torch.zeros(...) of
        appropriate size.
    y0 : torchTensor
        Initial guess for the dual variable, typically zeros(...)
    eta : float
        The measurement noise level, specifying the constraint.
    iter : int, optional
        Number of primal dual iterations. (Default 20)
    sigma : float, optional
        Step size parameter, should satisfy sigma*tau*||AW||_2^2 < 1.
        (Default 0.5)
    tau : float, optional
        Step size parameter, should satisfy sigma*tau*||AW||_2^2 < 1.
        (Default 0.5)
    theta : float, optional
        DR parameter, arbitrary in [0,1]. (Default 0.5)
    silent : bool, optional
        Disable progress bar. (Default False)
    report_pd_gap : bool, optional
        Report pd-gap at the end.

    Returns
    -------
    tensor
       The recovered signal x=Wc and coefficients c.
    """

    # we do not explicitly check for sig*tau*||AW||_2^2 < 1 and trust that
    # the user read the documentation ;)

    if y0 is None:
        y0 = torch.zeros_like(y)

    # helper functions for primal-dual gap
    def F(_y):
        return ((_y - y).norm(p=2, dim=-1) > (eta + 1e-2)) * 1e4

    def Fstar(_y):
        return eta * _y.norm(p=2, dim=-1) + (y * _y).sum(dim=-1)

    def Gstar(_y):
        return ((torch.max(torch.abs(_y), dim=-1))[0] > (1.0 + 1e-2)) * 1e4

    # init iteration variables
    c = c0.clone()
    c_ = c.clone()
    y_ = y0.clone()

    # run main primal dual iterations
    for it in tqdm(range(iter), desc="Primal-Dual iterations", disable=silent):

        # primal dual step 1)
        y_ = prox_l2_constraint_conjugate(
            y_ + sigma * OpA(OpW(c_)), sigma * y, sigma * eta
        )
        # primal dual step 2)
        cold, c = c, shrink(c - tau * OpW.adj(OpA.adj(y_)), tau)

        # primal dual step 3
        c_ = c + theta * (c - cold)

    # compute primal dual gap
    if report_pd_gap:
        E = (
            F(OpA(OpW(c_)))
            + c_.abs().sum(dim=-1)
            + Fstar(y_)
            + Gstar(-OpW.adj(OpA.adj(y_)))
        )
        print("\n\n Primal Dual Gap: \t {:1.4e} \n\n".format(E.abs().max()))

    return OpW(c_), c_, y_
