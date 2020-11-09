import glob
import os
import random

import numpy as np
import odl
import torch

from fastmri_utils.data import transforms
from tqdm import tqdm

from operators import rotate_real, to_complex


# ----- Dataset creation, saving, and loading -----


def create_iterable_dataset(
    n, set_params, generator, gen_params,
):
    """ Creates training, validation, and test data sets.

    Samples data signals from a data generator and stores them.

    Parameters
    ----------
    n : int
        Dimension of signals x.
    set_params : dictionary
        Must contain values for the following keys:
        path : str
            Directory path for storing the data sets.
        num_train : int
            Number of samples in the training set.
        num_val : int
            Number of samples in the validation set.
        num_test : int
            Number of samples in the validation set.
    generator : callable
        Generator function to create signal samples x. Will be called with
        the signature generator(n, **gen_params).
    gen_params : dictionary
        Additional keyword arguments passed on to the signal generator.
    """
    N_train, N_val, N_test = [
        set_params[key] for key in ["num_train", "num_val", "num_test"]
    ]

    def _get_signal():
        x, _ = generator(n, **gen_params)
        return x

    os.makedirs(os.path.join(set_params["path"], "train"), exist_ok=True)
    os.makedirs(os.path.join(set_params["path"], "val"), exist_ok=True)
    os.makedirs(os.path.join(set_params["path"], "test"), exist_ok=True)

    for idx in tqdm(range(N_train), desc="generating training signals"):
        torch.save(
            _get_signal(),
            os.path.join(
                set_params["path"], "train", "sample_{}.pt".format(idx)
            ),
        )

    for idx in tqdm(range(N_val), desc="generating validation signals"):
        torch.save(
            _get_signal(),
            os.path.join(
                set_params["path"], "val", "sample_{}.pt".format(idx)
            ),
        )

    for idx in tqdm(range(N_test), desc="generating test signals"):
        torch.save(
            _get_signal(),
            os.path.join(
                set_params["path"], "test", "sample_{}.pt".format(idx)
            ),
        )


def sample_ellipses(
    n,
    c_min=10,
    c_max=20,
    max_axis=0.5,
    min_axis=0.05,
    margin_offset=0.3,
    margin_offset_axis=0.9,
    grad_fac=1.0,
    bias_fac=1.0,
    bias_fac_min=0.0,
    normalize=True,
    n_seed=None,
    t_seed=None,
):
    """ Creates an image of random ellipses.

    Creates a piecewise linear signal of shape n (two-dimensional) with a
    random number of ellipses with zero boundaries.
    The signal is created as a functions in the box [-1,1] x [-1,1].
    The image is generated such that it cannot have negative values.

    Parameters
    ----------
    n : tuple
        Height x width
    c_min : int, optional
        Minimum number of ellipses. (Default 10)
    c_max : int, optional
         Maximum number of ellipses. (Default 20)
    max_axis : double, optional
        Maximum radius of the ellipses. (Default .5, in [0, 1))
    min_axis : double, optional
        Minimal radius of the ellipses. (Default .05, in [0, 1))
    margin_offset : double, optional
        Minimum distance of the center coordinates to the image boundary.
        (Default .3, in (0, 1])
    margin_offset_axis : double, optional
        Offset parameter so that the ellipses to not touch the image boundary.
        (Default .9, in [0, 1))
    grad_fac : double, optional
        Specifies the slope of the random linear piece that is created for each
        ellipse. Set it to 0.0 for constant pieces. (Default 1.0)
    bias_fac : double, optional
        Scalar factor that upscales the bias of the linear/constant pieces of
        each ellipse. Essentially specifies the weights of the ellipsoid
        regions. (Default 1.0)
    bias_fac_min : double, optional
        Lower bound on the bias for the weights. (Default 0.0)
    normalize : bool, optional
        Normalizes the image to the interval [0, 1] (Default True)
    n_seed : int, optional
        Seed for the numpy random number generator for drawing the jump
        positions. Set to `None` for not setting any seed and keeping the
        current state of the random number generator. (Default `None`)
    t_seed : int, optional
        Seed for the troch random number generator for drawing the jump
        heights. Set to `None` for not setting any seed and keeping the
        current state of the random number generator. (Default `None`)

    Returns
    -------
    torch.Tensor
        Will be of shape n (two-dimensional).
    """

    if n_seed is not None:
        np.random.seed(n_seed)
    if t_seed is not None:
        torch.manual_seed(t_seed)

    c = np.random.randint(c_min, c_max)

    cen_x = (1 - margin_offset) * 2 * (np.random.rand(c) - 1 / 2)
    cen_y = (1 - margin_offset) * 2 * (np.random.rand(c) - 1 / 2)
    cen_max = np.maximum(np.abs(cen_x), np.abs(cen_y))

    ax_1 = np.minimum(
        min_axis + (max_axis - min_axis) * np.random.rand(c),
        (1 - cen_max) * margin_offset_axis,
    )
    ax_2 = np.minimum(
        min_axis + (max_axis - min_axis) * np.random.rand(c),
        (1 - cen_max) * margin_offset_axis,
    )

    weights = np.ones(c)
    rot = np.pi / 2 * np.random.rand(c)

    p = np.stack([weights, ax_1, ax_2, cen_x, cen_y, rot]).transpose()
    space = odl.discr.discr_sequence_space(n)

    coord_x = np.linspace(-1.0, 1.0, n[0])
    coord_y = np.linspace(-1.0, 1.0, n[1])
    m_x, m_y = np.meshgrid(coord_x, coord_y)

    X = np.zeros(n)
    for e in range(p.shape[0]):
        E = -np.ones(n)
        while E.min() < 0:
            E = odl.phantom.geometric.ellipsoid_phantom(
                space, p[e : (e + 1), :]
            ).asarray()
            E = E * (
                grad_fac * np.random.randn(1) * m_x
                + grad_fac * np.random.randn(1) * m_y
                + bias_fac_min
                + (bias_fac - bias_fac_min) * np.random.rand(1)
            )
        X = X + E

    X = torch.tensor(X, dtype=torch.float)

    if normalize:
        X = X / X.max()

    return X, torch.tensor(c, dtype=torch.float)


class IPDataset(torch.utils.data.Dataset):
    """ Datasets for imaging inverse problems.

    Loads image signals created by `create_iterable_dataset` from a directory.

    Implements the map-style dataset in `torch`.

    Attributed
    ----------
    subset : str
        One of "train", "val", "test".
    path : str
        The directory path. Should contain the subdirectories "train", "val",
        "test" containing the training, validation, and test data respectively.
    """

    def __init__(self, subset, path, transform=None, device=None):
        self.path = path
        self.files = glob.glob(os.path.join(path, subset, "*.pt"))
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # load image and add channel dimension
        if self.device is not None:
            out = (torch.load(self.files[idx]).unsqueeze(0).to(self.device),)
        else:
            out = (torch.load(self.files[idx]).unsqueeze(0),)
        return self.transform(out) if self.transform is not None else out


# ----- data transforms -----


class JointRandomCrop(object):
    """ Joint random cropping transform for (input, target) image pairs. """

    def __init__(self, size):
        self.size = size

    def __call__(self, imgs):
        iw, ih = imgs[0].shape[-2:]  # image width and height
        cw, ch = self.size  # crop width and height

        # sample random corner of cropping area
        w0 = random.randint(0, iw - cw) if iw > cw else 0
        h0 = random.randint(0, ih - ch) if ih > ch else 0
        return tuple(img[..., w0 : w0 + cw, h0 : h0 + ch] for img in imgs)


class Inversion(object):
    """ Inverse transform on (meas, target) tuples.

    Inverts meas to image domain and returns (inv, target) pair.

    Parameters
    ----------
    inverter : callable
        The inversion operation to use.

    """

    def __init__(self, inverter):
        self.inverter = inverter

    def __call__(self, inputs):
        meas, target = inputs
        inv = self.inverter(meas)
        return inv, target


class SimulateMeasurements(object):
    """ Forward operator on target samples.

    Computes measurements and returns (measurement, target) pair.

    Parameters
    ----------
    operator : callable
        The measurement operation to use.

    """

    def __init__(self, operator):
        self.operator = operator

    def __call__(self, target):
        (target,) = target
        meas = self.operator(target)
        return meas, target


class ComplexMagnitude(object):
    """ Removes a complex channel from (input, target) image pairs.

    Returns the magnitude of an image as a single channel. If the image has no
    complex channel, then it is passed on unaltered.

    """

    def __init__(self):
        pass

    def __call__(self, imgs):
        return tuple(
            [
                rotate_real(img)[..., 0:1, :, :]
                if img.shape[-3] == 2
                else torch.abs(img)
                for img in imgs
            ]
        )


class ToComplex(object):
    """ Adds a complex channel to images.

    Transforms images of shape [..., 1, W, H] to shape [..., 2, W, H]
    by concatenating an empty channel for the imaginary part.

    """

    def __init__(self):
        pass

    def __call__(self, imgs):
        return tuple([to_complex(img) for img in imgs])


class CenterCrop(object):
    """ Crops (input, target) image pairs to have matching size. """

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, imgs):
        return tuple([transforms.center_crop(img, self.shape) for img in imgs])


class Flatten(object):
    """ Flattens selected dimensions of tensors. """

    def __init__(self, start_dim, end_dim):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, inputs):
        return tuple(
            [torch.flatten(x, self.start_dim, self.end_dim) for x in inputs]
        )


class Normalize(object):
    """ Normalizes (input, target) pairs with respect to target or input. """

    def __init__(self, p=2, reduction="sum", use_target=True):
        self.p = p
        self.reduction = reduction
        self.use_target = use_target

    def __call__(self, inputs):
        inp, tar = inputs
        norm = torch.norm(tar if self.use_target else inp, p=self.p)
        if self.reduction == "mean" and not self.p == "inf":
            norm /= np.prod(tar.shape) ** (1 / self.p)
        return inputs[0] / norm, inputs[1] / norm


class Jitter(object):
    """ Adds random pertubations to the input of (input, target) pairs.
    """

    def __init__(self, eta, scale_lo, scale_hi, n_seed=None, t_seed=None):
        self.eta = eta
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi
        self.rng = np.random.RandomState(n_seed)
        self.trng = torch.Generator()
        if t_seed is not None:
            self.trng.manual_seed(t_seed)

    def __call__(self, inputs):
        meas, target = inputs
        m = meas.shape[-1]  # number of sampled measurements
        scale = (
            self.scale_lo + (self.scale_hi - self.scale_lo) * self.rng.rand()
        )
        noise = torch.randn(meas.shape, generator=self.trng).to(meas.device)
        meas_noisy = meas + self.eta / np.sqrt(m) * noise * scale
        return meas_noisy, target


# ---- run data generation -----
if __name__ == "__main__":
    import config

    np.random.seed(config.numpy_seed)
    torch.manual_seed(config.torch_seed)
    create_iterable_dataset(
        config.n, config.set_params, config.data_gen, config.data_params,
    )
