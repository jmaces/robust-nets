from abc import ABC, abstractmethod

import numpy as np
import torch


# ----- Utilities -----


def get_operator_norm(*ops):
    """ Computes the l2-operator norm of a product of linear operators. """
    if all([hasattr(op, "get_matrix") for op in ops]):
        mat = ops[-1].get_matrix()
        for op in ops[:-1][::-1]:
            mat = torch.matmul(op.get_matrix(), mat)
        return np.linalg.norm(mat.cpu().numpy(), 2)
    else:
        raise ValueError(
            "Could not compute operator norm. At least one of "
            "the provided operators does not implement a matrix "
            "representation, which is required."
        )


def wrap_operator(A):
    """ Wraps linear operator as function handle. """
    if isinstance(A, torch.Tensor):

        def _Afunc(x):
            if x.dtype == torch.double:
                return torch.matmul(x, A.T.double())
            else:
                return torch.matmul(x, A.T)

    elif callable(A):
        _Afunc = A

    else:
        raise ValueError(
            "Unable to wrap as a function handle. "
            "Expected a Torch tensor or callable but got "
            "{} instead.".format(type(A).__name__)
        )
    return _Afunc


def l2_error(X, X_ref, relative=False, squared=False):
    """ Compute average l2-error.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor of shape [..., 1, N].
    X_ref : torch.Tensor
        The reference tensor of same shape.
    relative : bool, optional
        Use relative error. (Default False)
    squared : bool, optional
        Use squared error. (Default False)

    Returns
    -------
    err_av :
        The average error.
    err :
        Tensor with individual errors.

    """
    if squared:
        err = (X - X_ref).norm(p=2, dim=-1) ** 2
    else:
        err = (X - X_ref).norm(p=2, dim=-1)

    if relative:
        if squared:
            err = err / (X_ref.norm(p=2, dim=-1) ** 2)
        else:
            err = err / X_ref.norm(p=2, dim=-1)

    if X_ref.ndim > 1:
        err_av = err.sum() / X_ref.shape[0]
    else:
        err_av = err.sum()
    return err_av, err


def noise_gaussian(y, eta, n_seed=None, t_seed=None):
    """ Additive Gaussian noise. """
    if n_seed is not None:
        np.random.seed(n_seed)
    if t_seed is not None:
        torch.manual_seed(t_seed)

    noise = torch.randn_like(y)
    return y + eta / np.sqrt(y.shape[-1]) * noise


def noise_uniform(y, eta, n_seed=None, t_seed=None):
    """ Additive mean centered uniform noise. """
    if n_seed is not None:
        np.random.seed(n_seed)
    if t_seed is not None:
        torch.manual_seed(t_seed)

    noise = torch.rand_like(y) - 0.5
    return y + eta / np.sqrt(y.shape[-1] / 12) * noise


def noise_bernoulli(y, eta, n_seed=None, t_seed=None, p=1e-2):
    """ Additive Bernoulli noise with random sign (-> salt and pepper). """
    if n_seed is not None:
        np.random.seed(n_seed)
    if t_seed is not None:
        torch.manual_seed(t_seed)
    noise_pos = torch.bernoulli(p / 2 * torch.ones_like(y))
    noise_neg = torch.bernoulli(p / 2 * torch.ones_like(y))
    noise = noise_pos - noise_neg
    return y + eta / np.sqrt(y.shape[-1] * p * (1 - p / 2)) * noise


def get_tikhonov_matrix(OpA, OpW, reg_fac):
    """ Generalized Tikhonov regularized inversion. """
    A_mat = OpA.get_matrix().cpu().numpy()
    W_mat = OpW.get_matrix().cpu().numpy()
    W_mat = W_mat[0:-1, :]
    A_tikh = np.matmul(
        np.linalg.pinv(
            np.matmul(np.transpose(A_mat), A_mat)
            + reg_fac * np.matmul(np.transpose(W_mat), W_mat)
        ),
        np.transpose(A_mat),
    )
    return torch.tensor(A_tikh, device=OpA.device)


# ----- Thresholding, Projections, and Proximal Operators -----


def _shrink_single(x, thresh):
    """ Soft/Shrinkage thresholding for tensors. """
    return torch.nn.Softshrink(thresh)(x)


def _shrink_recursive(c, thresh):
    """ Soft/Shrinkage thresholding for nested tuples/lists of tensors. """
    if isinstance(c, (list, tuple)):
        return [_shrink_recursive(el, thresh) for el in c]
    else:
        return _shrink_single(c, thresh)


shrink = _shrink_single  # alias for convenience


def proj_l2_ball(x, centre, radius):
    """ Euclidean projection onto a closed l2-ball.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor to project.
    centre : torch.Tensor
        The centre of the ball.
    radius : float
        The radius of the ball. Must be non-negative.

    Returns
    -------
    torch.Tensor
        The projection of x onto the closed ball.
    """
    norm = torch.sqrt((x - centre).pow(2).sum(dim=(-2, -1), keepdim=True))
    radius, norm = torch.broadcast_tensors(radius, norm)
    fac = torch.ones_like(norm)
    fac[norm > radius] = radius[norm > radius] / norm[norm > radius]
    return fac * x + (1 - fac) * centre


def proj_linf_ball(x, centre, radius):
    """ Euclidean projection onto a closed l-inf-ball.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor to project.
    centre : torch.Tensor
        The centre of the ball.
    radius : float
        The radius of the ball. Must be non-negative.

    Returns
    -------
    torch.Tensor
        The projection of x onto the closed ball.
    """
    tmp = torch.min(x, centre + radius)
    return torch.max(tmp, centre - radius)


# proximals of convex constraints are just the projections onto them

prox_l2_constraint = proj_l2_ball  # alias for convenience
prox_linf_constraint = proj_linf_ball  # alias for convenience


def prox_of_constraint_conjugate(proj):
    """ Proximal operator of the conjugate of a convex constraint.

    Let C be a convex set and F(z) = Chi_C(z) be the characteristic function of
    C. Then the convex conjugate  of F is

        F*(x) = sup_{z in C} Re(<z, x>)

    and by the Moreau decompisition we know

        prox_F* = Id - prox_F = Id - proj_C

    thus the proximal of F* can be easily obtained from the projection onto C.

    This function wraps Id-proj_C for a given projection operator. The
    projection is assumed to take arguments proj(x, *args, **kwargs), where
    x is the input to project and further arguments can be used to specify C.

    Parameters
    ----------
    proj : callable
        The projection function to wrap.

    Returns
    -------
    callable
        A function providing the proximal of F*. It will take arguments
        x : torch.Tensor
            The input tensor.
        *args
            Additional variable length arguments passed on to the projection,
            e.g. to specify C.
        **kwargs
            Additional keyword arguments passed on to the projection, e.g. to
            specify C.

    """

    def _prox(x, *args, **kwargs):
        return x - proj(x, *args, **kwargs)

    return _prox


# build prox operators of constraint conjugates
prox_l2_constraint_conjugate = prox_of_constraint_conjugate(proj_l2_ball)
prox_linf_constraint_conjugate = prox_of_constraint_conjugate(proj_linf_ball)


# ----- Linear Operator Utilities -----


class LinearOperator(ABC):
    """ Abstract base class for linear (measurement) operators.

    Can be used for real operators

        A : R^n ->  R^m

    or complex operators

        A : C^n -> C^m.

    Can be applied to tensors of shape (n,) or (1, n) or batches thereof of
    shape (*, n) or (*, 1, n) in the real case, or analogously shapes (2, n)
    or (*, 2, n) in the complex case.

    Attributes
    ----------
    m : int
        Dimension of the co-domain of the operator.
    n : int
        Dimension of the domain of the operator.

    """

    def __init__(self, m, n):
        self.m = m
        self.n = n

    @abstractmethod
    def dot(self, x):
        """ Application of the operator to a vector.

        Computes Ax for a given vector x from the domain.

        Parameters
        ----------
        x : torch.Tensor
            Must be of shape to (*, n) or (*, 2, n).
        Returns
        -------
        torch.Tensor
            Will be of shape (*, m) or (*, 2, m).
        """
        pass

    @abstractmethod
    def adj(self, y):
        """ Application of the adjoint operator to a vector.

        Computes (A^*)y for a given vector y from the co-domain.

        Parameters
        ----------
        y : torch.Tensor
            Must be of shape (*, m) or (*, 2, m).

        Returns
        -------
        torch.Tensor
            Will be of shape (*, n) or (*, 2, n).
        """
        pass

    @abstractmethod
    def inv(self, y):
        """ Application of some inversion of the operator to a vector.

        Computes (A^dagger)y for a given vector y from the co-domain.
        A^dagger can for example be the pseudo-inverse.

        Parameters
        ----------
        y : torch.Tensor
            Must be of shape (*, m) or (*, 2, m).

        Returns
        -------
        torch.Tensor
            Will be of shape (*, n) or (*, 2, n).
        """
        pass

    def __call__(self, x):  # alias to make operator callable by using dot
        return self.dot(x)

    def get_matrix(self):
        """ Return the matrix representation of the operator.

        Returns
        -------
        torch.Tensor
            Will be of shape (m, n) if real or (2, m, n) if complex.
        """
        if hasattr(self, "t_A"):
            return self.t_A
        device = self.device if hasattr(self, "device") else None
        return self.dot(torch.eye(self.n, device=device)).T


# ----- Measurement Operators -----


class Gaussian(LinearOperator):
    """ Gaussian random matrix measurement operator.

    Implements the real operator R^n -> R^m

        A = 1/sqrt(m) W

    with Gaussian random matrix W=randn(m,n). The adjoint is the transpose.
    The inversion is given by the pseudo-inverse.

    Parameters
    ----------
    seed : int, optional
        Seed for the random number generator for drawing the random maxtrix W.
        Set to `None` for not setting any seed and keeping the current state
        of the random number generator. (Default `None`)
    device : torch.Device or int, optional
        The torch device or its ID to place the operator on. Set to `None` to
        use the global torch default device. (Default `None`)

    """

    def __init__(self, m, n, seed=None, device=None):
        super().__init__(m, n)
        self.seed = seed
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)

        self.t_A = (
            1
            / torch.sqrt(torch.tensor(m, dtype=torch.float))
            * torch.randn((m, n), dtype=torch.float)
        ).to(device)
        self.t_A_pinv = torch.pinverse(self.t_A)
        self._t_A_func = wrap_operator(self.t_A)
        self._t_A_adj_func = wrap_operator(self.t_A.T)
        self._t_A_pinv_func = wrap_operator(self.t_A_pinv)

    def dot(self, x):
        return self._t_A_func(x)

    def adj(self, y):
        return self._t_A_adj_func(y)

    def inv(self, y):
        return self._t_A_pinv_func(y)


class TVAnalysis(LinearOperator):
    """ 1D Total Variation analysis operator.

    Implements the real operator R^n -> R^n

           [[ -1 1  0  ...  0 ]
            [ 0  -1 1 0 ... 0 ]
        A = [ .     .  .    . ]
            [ .       -1  1 0 ]
            [ 0 ...    0 -1 1 ]
            [1/n 1/n ...   1/n]]

    which is the forward finite difference operator with von Neumann boundary
    conditions plus an extra row capturing the average.
    The adjoint is the transpose. The inversion is the pseudo-inverse.

    Can also be applied to complex tensors with shape [n, 2].
    It will then act upon the real part and imaginary part separately.

    Parameters
    ----------
    device : torch.Device or int, optional
        The torch device or its ID to place the operator on. Set to `None` to
        use the global torch default device. (Default `None`)

    See also
    --------
    TVSynthesis : The linear operator implementing the pseudo-inverse of A.

    """

    def __init__(self, n, device=None):
        super().__init__(n, n)
        self.device = device
        self.t_A = torch.cat(
            [
                torch.eye(n, device=device)[1:, :]
                - torch.eye(n, device=device)[:-1, :],
                1 / n * torch.ones(1, n, device=device),
            ],
            dim=0,
        )

        # construct pseudo inverse
        temp = np.flip((-1) * np.linspace(1 / n, (n - 1) / n, n - 1))
        Temp = np.tile(temp, (n, 1))
        Temp2 = np.tril(np.ones((n, n)), -1)
        pinvgrad = Temp + Temp2[0:n, 0 : n - 1]
        pinvgrad = np.concatenate((pinvgrad, np.ones((n, 1))), axis=1)
        self.t_A_pinv = torch.tensor(
            pinvgrad, device=self.device, dtype=torch.float
        )
        self._t_A_pinv_func = wrap_operator(self.t_A_pinv)

    def dot(self, x):
        return torch.cat(
            [
                x[..., 1:] - x[..., :-1],
                x.mean(axis=-1, keepdim=True)
                * torch.ones(x.shape[:-1] + (1,), device=self.device),
            ],
            dim=-1,
        )

    def adj(self, y):
        result = torch.zeros(y.shape[:-1] + (self.n,), device=self.device)
        result[..., 1:] += y[..., :-1]
        result[..., :-1] -= y[..., :-1]
        result += 1 / self.n * y[..., -1:]
        return result

    def inv(self, y):
        return self._t_A_pinv_func(y)


class TVSynthesis(LinearOperator):
    """ 1D Total Variation synthesis operator.

    Implements the real operator A : R^n -> R^n that is the pseudo-inverse of
    TVAnalysis. Thus, the inversion is the forward operator of TVAnalysis.
    The adjoint is the transpose.

    Can also be applied to complex tensors with shape [n, 2].
    It will then act upon the real part and imaginary part separately.

    Remark: Strictly speaking, TVAnalysis and TVSynthesis are not true
    analysis and synthesis operator pairs, as this would just simply mean
    taking the adjoint. Instead, by taking the pseudo-inverse leads we obtain
    a pair of operators that are the analysis/synthesis operator for the
    dual frame of the other and vice versa.

    Parameters
    ----------
    device : torch.Device or int, optional
        The torch device or its ID to place the operator on. Set to `None` to
        use the global torch default device. (Default `None`)

    See also
    --------
    TVAnalysis : The linear operator implementing the pseudo-inverse of A.

    """

    def __init__(self, n, device=None):
        super().__init__(n, n)
        self.device = device
        self.t_A_pinv = torch.cat(
            [
                torch.eye(n, device=device)[1:, :]
                - torch.eye(n, device=device)[:-1, :],
                1 / n * torch.ones(1, n, device=device),
            ],
            dim=0,
        )

        # construct pseudo inverse
        temp = np.flip((-1) * np.linspace(1 / n, (n - 1) / n, n - 1))
        Temp = np.tile(temp, (n, 1))
        Temp2 = np.tril(np.ones((n, n)), -1)
        pinvgrad = Temp + Temp2[0:n, 0 : n - 1]
        pinvgrad = np.concatenate((pinvgrad, np.ones((n, 1))), axis=1)
        self.t_A = torch.tensor(
            pinvgrad, device=self.device, dtype=torch.float
        )

        self._t_A_func = wrap_operator(self.t_A)
        self._t_A_adj_func = wrap_operator(self.t_A.T)

    def dot(self, x):
        return self._t_A_func(x)

    def adj(self, y):
        return self._t_A_adj_func(y)

    def inv(self, y):
        return torch.cat(
            [
                y[..., 1:] - y[..., :-1],
                y.mean(axis=-1)
                * torch.ones(y.shape[:-1] + (1,), device=self.device),
            ],
            dim=-1,
        )
