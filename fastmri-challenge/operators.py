import math

from abc import ABC, abstractmethod

import numpy as np
import skimage.transform
import torch

from fastmri_utils.data import transforms
from fastmri_utils.data.transforms import fftshift, ifftshift


# ----- Utilities -----


class RadialMaskFunc(object):
    """ Generates a golden angle radial spokes mask.

    Useful for subsampling a Fast-Fourier-Transform.
    Contains radial lines (spokes) through the center of the mask, with
    angles spaced according to the golden angle (~111.25°). The first line
    has angle 0° (horizontal). An offset parameter can be given to skip
    the first `offset*num_lines` lines.

    Parameters
    ----------
    shape : array_like
        A tuple specifying the size of the mask.
    num_lines : int
        Number of radial lines (spokes) in the mask.
    offset : int, optional
        Offset factor for the range of angles in the mask.
    """

    def __init__(self, shape, num_lines, offset=0):
        self.shape = shape
        self.num_lines = num_lines
        self.offset = offset
        self.mask = self._generate_radial_mask(shape, num_lines, offset)

    def __call__(self, shape, seed=None):

        if (self.mask.shape[0] != shape[-3]) or (
            self.mask.shape[1] != shape[-2]
        ):
            return torch.zeros(shape)

        return torch.reshape(
            self.mask, (len(shape) - 3) * (1,) + self.shape + (1,)
        )

    def _generate_radial_mask(self, shape, num_lines, offset=0):
        # generate line template and empty mask
        x, y = shape
        d = math.ceil(np.sqrt(2) * max(x, y))
        line = np.zeros((d, d))
        line[d // 2, :] = 1.0
        out = np.zeros((d, d))
        # compute golden angle sequence
        golden = (np.sqrt(5) - 1) / 2
        angles = (
            180.0
            * golden
            * np.arange(offset * num_lines, (offset + 1) * num_lines)
        )
        # draw lines
        for angle in angles:
            out += skimage.transform.rotate(line, angle, order=0)
        # crop mask to correct size
        out = out[
            d // 2 - math.floor(x / 2) : d // 2 + math.ceil(x / 2),
            d // 2 - math.floor(y / 2) : d // 2 + math.ceil(y / 2),
        ]
        # return binary mask
        return torch.tensor(out > 0)


def l2_error(X, X_ref, relative=False, squared=False, use_magnitude=True):
    """ Compute average l2-error of an image over last three dimensions.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor of shape [..., 1, W, H] for real images or
        [..., 2, W, H] for complex images.
    X_ref : torch.Tensor
        The reference tensor of same shape.
    relative : bool, optional
        Use relative error. (Default False)
    squared : bool, optional
        Use squared error. (Default False)
    use_magnitude : bool, optional
        Use complex magnitudes. (Default True)

    Returns
    -------
    err_av :
        The average error.
    err :
        Tensor with individual errors.

    """
    assert X_ref.ndim >= 3  # do not forget the channel dimension

    if X_ref.shape[-3] == 2 and use_magnitude:  # compare complex magnitudes
        X_flat = torch.flatten(torch.sqrt(X.pow(2).sum(-3)), -2, -1)
        X_ref_flat = torch.flatten(torch.sqrt(X_ref.pow(2).sum(-3)), -2, -1)
    else:
        X_flat = torch.flatten(X, -3, -1)
        X_ref_flat = torch.flatten(X_ref, -3, -1)

    if squared:
        err = (X_flat - X_ref_flat).norm(p=2, dim=-1) ** 2
    else:
        err = (X_flat - X_ref_flat).norm(p=2, dim=-1)

    if relative:
        if squared:
            err = err / (X_ref_flat.norm(p=2, dim=-1) ** 2)
        else:
            err = err / X_ref_flat.norm(p=2, dim=-1)

    if X_ref.ndim > 3:
        err_av = err.sum() / np.prod(X_ref.shape[:-3])
    else:
        err_av = err
    return err_av.squeeze(), err


def l2_error_meas(X, X_ref, relative=False, squared=False):
    """ Compute average l2-error of measurements over last two dimensions.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor of shape [..., 1, N] for real measurements or
        [..., 2, N] for complex measurements.
    X_ref : torch.Tensor
        The reference tensor of same shape.
    relative : bool, optional
        Use relative error. (Default False)
    squared : bool, optional
        Use squared error. (Default False)
    use_magnitude : bool, optional
        Use complex magnitudes. (Default True)

    Returns
    -------
    err_av :
        The average error.
    err :
        Tensor with errors.

    """
    assert X_ref.ndim >= 2  # do not forget the channel dimension

    X_flat = torch.flatten(X, -2, -1)
    X_ref_flat = torch.flatten(X_ref, -2, -1)

    if squared:
        err = (X_flat - X_ref_flat).norm(p=2, dim=-1) ** 2
    else:
        err = (X_flat - X_ref_flat).norm(p=2, dim=-1)

    if relative:
        if squared:
            err = err / (X_ref_flat.norm(p=2, dim=-1) ** 2)
        else:
            err = err / X_ref_flat.norm(p=2, dim=-1)

    if X_ref.ndim > 2:
        err_av = err.sum() / np.prod(X_ref.shape[:-2])
    else:
        err_av = err
    return err_av, err


def noise_gaussian(y, eta, n_seed=None, t_seed=None):
    """ Additive Gaussian noise. """
    if n_seed is not None:
        np.random.seed(n_seed)
    if t_seed is not None:
        torch.manual_seed(t_seed)

    noise = torch.randn_like(y)
    return y + eta / np.sqrt(np.prod(y.shape[1:])) * noise


def to_complex(x):
    """ Converts real images to complex by adding a channel dimension. """
    assert x.ndim >= 3 and (x.shape[-3] == 1 or x.shape[-3] == 2)
    # real tensor of shape (1, n1, n2) or batch of shape (*, 1, n1, n2)
    if x.shape[-3] == 1:
        imag = torch.zeros_like(x)
        out = torch.cat([x, imag], dim=-3)
    else:
        out = x
    return out


def rotate_real(x):
    """ Rotates the magnitude of a complex signal into the real channel. """
    assert x.ndim >= 3 and (x.shape[-3] == 2)
    x_rv = torch.zeros_like(x)
    x_rv[..., 0, :, :] = torch.sqrt(x.pow(2).sum(-3))
    return x_rv


def mult_complex(x, y):
    """ Multiply two complex tensors with real and imag in last dimension. """
    out_real = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    out_imag = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    return torch.stack([out_real, out_imag], dim=-1)


def div_complex(x, y):
    """ Divide two complex tensors with real and imag in last dimension. """
    out_real = x[..., 0] * y[..., 0] + x[..., 1] * y[..., 1]
    out_imag = x[..., 1] * y[..., 0] - x[..., 0] * y[..., 1]
    denom = y[..., 0].pow(2) + y[..., 1].pow(2)
    return torch.stack([out_real / denom, out_imag / denom], dim=-1)


def conj_complex(x):
    """ Complex conjugate of tensor with real and imag in last dimension. """
    out_real = x[..., 0]
    out_imag = -x[..., 1]
    return torch.stack([out_real, out_imag], dim=-1)


def im2vec(x, dims=(-2, -1)):
    """ Flattens last two dimensions of an image tensor to a vector. """
    return torch.flatten(x, *dims)


def vec2im(x, n):
    """ Unflattens the last dimension of a vector to two image dimensions. """
    return x.view(*x.shape[:-1], *n)


def prep_fft_channel(x):
    """ Rotates complex image dimension from channel to last position. """
    x = torch.reshape(x, x.shape[:-3] + (x.shape[-3] // 2, 2) + x.shape[-2:])
    return x.permute(*range(x.ndim - 3), -2, -1, -3)


def unprep_fft_channel(x):
    """ Rotates complex image dimension from last to channel position. """
    x = x.permute(*range(x.ndim - 3), -1, -3, -2)
    return torch.reshape(
        x, x.shape[:-4] + (x.shape[-4] * x.shape[-3],) + x.shape[-2:]
    )


def circshift(x, dim=-1, num=1):
    """ Circular shift by n along a dimension. """
    perm = list(range(num, x.shape[dim])) + list(range(0, num))
    if not dim == -1:
        return x.transpose(dim, -1)[..., perm].transpose(dim, -1)
    else:
        return x[..., perm]


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


# ----- Linear Operator Utilities -----


class LinearOperator(ABC):
    """ Abstract base class for linear (measurement) operators.

    Can be used for real operators

        A : R^(n1 x n2) ->  R^m

    or complex operators

        A : C^(n1 x n2) -> C^m.

    Can be applied to tensors of shape (n1, n2) or (1, n1, n2) or batches
    thereof of shape (*, n1, n2) or (*, 1, n1, n2) in the real case, or
    analogously shapes (2, n1, n2) or (*, 2, n1, n2) in the complex case.

    Attributes
    ----------
    m : int
        Dimension of the co-domain of the operator.
    n : tuple of int
        Dimensions of the domain of the operator.

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
            Must be of shape to (*, n1, n2) or (*, 2, n1, n2).
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
            Will be of shape (*, n1, n2) or (*, 2, n1, n2).
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
            Will be of shape (*, n1, n2) or (*, 2, n1, n2).
        """
        pass

    def __call__(self, x):  # alias to make operator callable by using dot
        return self.dot(x)


# ----- Measurement Operators -----


class Fourier(LinearOperator):
    """ 2D discrete Fourier transform.

    Implements the complex operator C^(n1, n2) -> C^m
    appling the (subsampled) Fourier transform.
    The adjoint is the conjugate transpose. The inverse is the same as adjoint.


    Parameters
    ----------
    mask : torch.Tensor
        The subsampling mask for the Fourier transform.

    """

    def __init__(self, mask):
        m = mask.nonzero().shape[0]
        n = mask.shape[-2:]
        super().__init__(m, n)
        self.mask = mask[0, 0, :, :].bool()

    def dot(self, x):
        """ Subsampled Fourier transform. """
        full_fft = unprep_fft_channel(transforms.fft2(prep_fft_channel(x)))
        return im2vec(full_fft)[..., im2vec(self.mask)]

    def adj(self, y):
        """ Adjoint is the zeor-filled inverse Fourier transform. """
        masked_fft = torch.zeros(
            *y.shape[:-1], self.n[0] * self.n[1], device=y.device
        )
        masked_fft[..., im2vec(self.mask)] = y
        return unprep_fft_channel(
            transforms.ifft2(prep_fft_channel(vec2im(masked_fft, self.n)))
        )

    def inv(self, y):
        """ Pseudo-inverse a.k.a. zero-filled IFFT. """
        return self.adj(y)

    def tikh(self, rhs, kernel, rho):
        """ Tikhonov regularized inversion.

        Solves the normal equation

            (F*F + rho W*W) x = F*y

        or more generally

            (F*F + rho W*W) x = z

        for a Tikhonov regularized least squares fit, assuming that the
        regularization W*W can be diagonalied by FFTs, i.e.

            W*W = F*D*F

        for some diagonal matrix D.

        Parameters
        ----------
        rhs : torch.Tensor
            The right hand side tensor z, often F*y for some y.
        kernel : torch.Tensor
            The Fourier kernel of W, containing the diagonal elements D.
        rho : float
            The regularization parameter.

        """
        assert rhs.ndim >= 3 and rhs.shape[-3] == 2  # assert complex images
        fft_rhs = transforms.fft2(prep_fft_channel(rhs))
        combined_kernel = prep_fft_channel(
            to_complex(self.mask.unsqueeze(0).to(rhs.device))
        ) + rho * kernel.to(rhs.device)
        fft_div = div_complex(fft_rhs, combined_kernel)
        return unprep_fft_channel(transforms.ifft2(fft_div))


class Fourier_matrix(LinearOperator, torch.nn.Module):
    """ 2D discrete Fourier transform based on Kroneckers of 1D Fourier.

    Implements the complex operator C^(n1, n2) -> C^m appling the (subsampled)
    Fourier transform. The adjoint is the conjugate transpose. The inverse is
    the same as adjoint.

    The Kronecker product implementation can be faster than the regular 2D
    implementation in certain situations.


    Parameters
    ----------
    mask : torch.Tensor
        The subsampling mask for the Fourier transform.

    """

    def __init__(self, mask):
        m = mask.nonzero().shape[0]
        n = mask.shape[-2:]
        LinearOperator.__init__(self, m, n)
        torch.nn.Module.__init__(self)
        self.mask = mask[0, 0, :, :].bool()
        self.fft2 = LearnableFourier2D(n, inverse=False, learnable=False)
        self.ifft2 = LearnableFourier2D(n, inverse=True, learnable=False)

    def dot(self, x):
        """ Subsampled Fourier transform. """
        full_fft = self.fft2(x)
        return im2vec(full_fft)[..., im2vec(self.mask)]

    def adj(self, y):
        """ Adjoint is the zeor-filled inverse Fourier transform. """
        masked_fft = torch.zeros(
            *y.shape[:-1], self.n[0] * self.n[1], device=y.device
        )
        masked_fft[..., im2vec(self.mask)] = y
        return self.ifft2(vec2im(masked_fft, self.n))

    def inv(self, y):
        """ Pseudo-inverse a.k.a. zero-filled IFFT. """
        return self.adj(y)

    def tikh(self, rhs, kernel, rho):
        """ Tikhonov regularized inversion.

        Solves the normal equation

            (F*F + rho W*W) x = F*y

        or more generally

            (F*F + rho W*W) x = z

        for a Tikhonov regularized least squares fit, assuming that the
        regularization W*W can be diagonalied by FFTs, i.e.

            W*W = F*D*F

        for some diagonal matrix D.

        Parameters
        ----------
        rhs : torch.Tensor
            The right hand side tensor z, often F*y for some y.
        kernel : torch.Tensor
            The Fourier kernel of W, containing the diagonal elements D.
        rho : float
            The regularization parameter.

        """
        assert rhs.ndim >= 3 and rhs.shape[-3] == 2  # assert complex images
        fft_rhs = prep_fft_channel(self.fft2(rhs))
        combined_kernel = prep_fft_channel(
            to_complex(self.mask.unsqueeze(0).to(rhs.device))
        ) + rho * kernel.to(rhs.device)
        fft_div = unprep_fft_channel(div_complex(fft_rhs, combined_kernel))
        return self.ifft2(fft_div)


class TVAnalysisPeriodic(LinearOperator):
    """ 2D Total Variation analysis operator.
    Implements the real operator R^(n1, n2) -> R^(2*n1*n2)
    appling the forward finite difference operator

           [[ -1 1  0  ...   0 ]
            [ 0  -1 1 0 ...  0 ]
            [ .     .  .     . ]
            [ .       -1  1  0 ]
            [ 0 ...    0 -1  1 ]
            [ 1 ...    0  0 -1 ]]

    with periodic boundary conditions along the rows and columns of an
    image. The adjoint is the transpose.
    Can also be applied to complex tensors with shape (2, n1, n2).
    It will then act upon the real part and imaginary part separately.
    Parameters
    ----------
    device : torch.Device or int, optional
        The torch device or its ID to place the operator on. Set to `None` to
        use the global torch default device. (Default `None`)
    """

    def __init__(self, n, device=None):
        super().__init__(n[0] * n[1] * 2, n)
        self.device = device

    def dot(self, x):
        row_diff = circshift(x, dim=-2) - x
        col_diff = circshift(x, dim=-1) - x
        return torch.cat([im2vec(row_diff), im2vec(col_diff)], dim=-1,)

    def adj(self, y):
        row_diff = vec2im(
            y[..., : self.n[0] * self.n[1]], (self.n[0], self.n[1])
        )
        col_diff = vec2im(
            y[..., self.n[0] * self.n[1] :], (self.n[0], self.n[1])
        )
        return (
            circshift(row_diff, dim=-2, num=self.n[0] - 1)
            - row_diff
            + circshift(col_diff, dim=-1, num=self.n[0] - 1)
            - col_diff
        )

    def get_fourier_kernel(self):
        """ The factors of the operator after diagonalization by 2D FFTs. """
        kernel = torch.zeros(self.n[0], self.n[1]).unsqueeze(-3)
        kernel[0, 0, 0] = 4
        kernel[0, 0, 1] = -1
        kernel[0, 1, 0] = -1
        kernel[0, 0, -1] = -1
        kernel[0, -1, 0] = -1

        return transforms.fftshift(
            torch.fft(prep_fft_channel(to_complex(kernel)), signal_ndim=2),
            dim=(-3, -2),
        )

    def inv(self, y):
        raise NotImplementedError(
            "This operator does not implement a direct " "inversion."
        )


class LearnableFourier1D(torch.nn.Module):
    """ Learnable 1D discrete Fourier transform.

    Implements a complex operator C^n -> C^n, which is learnable but
    initialized as the Fourier transform.


    Parameters
    ----------
    n : int
        Dimension of the domain and range of the operator.
    dim : int, optional
        Apply the 1D operator along specified axis for inputs with multiple
        axis. (Default is last axis)
    inverse : bool, optional
        Use the discrete inverse Fourier transform as initialization instead.
        (Default False)
    learnable : bool, optional
        Make operator learnable. Otherwise it will be kept fixed as the
        initialization. (Default True)

    """

    def __init__(self, n, dim=-1, inverse=False, learnable=True):
        super(LearnableFourier1D, self).__init__()
        self.n = n
        self.dim = dim
        eye_n = torch.stack([torch.eye(n), torch.zeros(n, n)], dim=-1)
        if inverse:
            fft_n = ifftshift(
                torch.ifft(
                    fftshift(eye_n, dim=1), signal_ndim=1, normalized=True
                ),
                dim=1,
            )
        else:
            fft_n = ifftshift(
                torch.fft(
                    fftshift(eye_n, dim=1), signal_ndim=1, normalized=True
                ),
                dim=1,
            )
        fft_real_n = fft_n[..., 0]
        fft_imag_n = fft_n[..., 1]
        fft_matrix = torch.cat(
            [
                torch.cat([fft_real_n, -fft_imag_n], dim=1),
                torch.cat([fft_imag_n, fft_real_n], dim=1),
            ],
            dim=0,
        )
        self.linear = torch.nn.Linear(2 * n, 2 * n, bias=False)
        self.linear.weight.data = (
            fft_matrix
            + 1 / (np.sqrt(self.n) * 32) * torch.randn_like(fft_matrix)
            if learnable
            else fft_matrix
        )
        self.linear.weight.requires_grad = learnable

    def forward(self, x):
        xt = torch.transpose(x, self.dim, -1)
        x_real = xt[..., 0, :, :]
        x_imag = xt[..., 1, :, :]
        x_vec = torch.cat([x_real, x_imag], dim=-1)
        fft_vec = self.linear(x_vec)
        fft_real = fft_vec[..., : self.n]
        fft_imag = fft_vec[..., self.n :]
        return torch.transpose(
            torch.stack([fft_real, fft_imag], dim=-3), -1, self.dim
        )


class LearnableFourier2D(torch.nn.Module):
    """ Learnable 2D discrete Fourier transform.

    Implements a complex operator C^(n1, n2) -> C^(n1, n2), which is learnable
    but initialized as the Fourier transform. Operates along the last two
    dimensions of inputs with more axis.


    Parameters
    ----------
    n : tuple of int
        Dimensions of the domain and range of the operator.
    inverse : bool, optional
        Use the discrete inverse Fourier transform as initialization instead.
        (Default False)
    learnable : bool, optional
        Make operator learnable. Otherwise it will be kept fixed as the
        initialization. (Default True)

    """

    def __init__(self, n, inverse=False, learnable=True):
        super(LearnableFourier2D, self).__init__()
        self.linear1 = LearnableFourier1D(
            n[0], dim=-2, inverse=inverse, learnable=learnable
        )
        self.linear2 = LearnableFourier1D(
            n[1], dim=-1, inverse=inverse, learnable=learnable
        )

    def forward(self, x):
        return self.linear1(self.linear2(x))


class LearnableInverter(torch.nn.Module):
    """ Learnable inversion of subsampled discrete Fourier transform.

    The zero-filling (transpose of the subsampling operator) is fixed.
    The inversion is learnable and initialized as a 2D inverse Fourier
    transform, realized as Kroneckers of 1D Fourier inversions.

    Implements a complex operator C^m -> C^(n1, n2).


    Parameters
    ----------
    n : tuple of int
        Dimensions of the range of the operator.
    mask : torch.Tensor
        The subsampling mask. Determines m.

    """

    def __init__(self, n, mask, learnable=True):
        super(LearnableInverter, self).__init__()
        self.n = n
        self.mask = mask[0, 0, :, :].bool()
        self.learnable_ifft = LearnableFourier2D(
            n, inverse=True, learnable=learnable
        )

    def forward(self, y):
        masked_fft = torch.zeros(
            *y.shape[:-1], self.n[0] * self.n[1], device=y.device
        )
        masked_fft[..., im2vec(self.mask)] = y
        return self.learnable_ifft(vec2im(masked_fft, self.n))
