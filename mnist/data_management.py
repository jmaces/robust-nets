import os

import numpy as np
import torch

from torchvision.datasets import MNIST
from tqdm import tqdm


# ----- Dataset creation, saving, and loading -----


def create_dataset(m, n, measure, set_params, generator, gen_params):
    """ Creates training, validation, and test data sets.

    Produces data triples (x, c, y) given a measurement procedure

        y = Ax + noise

    and signal generator for x following a synthesis model

        x = Wc

    with typically sparse coefficients c. Three separate data sets for
    training, validation, and testing are created and stored.

    Parameters
    ----------
    m : int
        Dimension of measurements y.
    n : int
        Dimension of signals x.
    measure : callable
        The measurement procedure linking x to y.
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
        x, c = generator(n, **gen_params)
        return x, c

    X_train = torch.zeros(N_train, n)
    C_train = torch.zeros(N_train, n)
    for idx in tqdm(range(N_train), desc="generating training signals"):
        x, c = _get_signal()
        X_train[idx, :] = x
        C_train[idx, :] = c

    X_val = torch.zeros(N_val, n)
    C_val = torch.zeros(N_val, n)
    for idx in tqdm(range(N_val), desc="generating validation signals"):
        x, c = _get_signal()
        X_val[idx, :] = x
        C_val[idx, :] = c

    X_test = torch.zeros(N_test, n)
    C_test = torch.zeros(N_test, n)
    for idx in tqdm(range(N_test), desc="generating test signals"):
        x, c = _get_signal()
        X_test[idx, :] = x
        C_test[idx, :] = c

    Y_train = torch.zeros(N_train, m)
    for idx in tqdm(range(N_train), desc="computing training measurements"):
        Y_train[idx, :] = measure(X_train[idx, ...])

    Y_val = torch.zeros(N_val, m)
    for idx in tqdm(range(N_val), desc="computing validation measurements"):
        Y_val[idx, :] = measure(X_val[idx, ...])

    Y_test = torch.zeros(N_test, m)
    for idx in tqdm(range(N_test), desc="computing test measurements"):
        Y_test[idx, :] = measure(X_test[idx, ...])

    os.makedirs(set_params["path"], exist_ok=True)
    torch.save(X_train, os.path.join(set_params["path"], "train_x.pt"))
    torch.save(C_train, os.path.join(set_params["path"], "train_c.pt"))
    torch.save(Y_train, os.path.join(set_params["path"], "train_y.pt"))
    torch.save(X_val, os.path.join(set_params["path"], "val_x.pt"))
    torch.save(C_val, os.path.join(set_params["path"], "val_c.pt"))
    torch.save(Y_val, os.path.join(set_params["path"], "val_y.pt"))
    torch.save(X_test, os.path.join(set_params["path"], "test_x.pt"))
    torch.save(C_test, os.path.join(set_params["path"], "test_c.pt"))
    torch.save(Y_test, os.path.join(set_params["path"], "test_y.pt"))


def load_dataset(path, subset="train"):
    """ Loads a partial data set.

    Loads the training, validation, or testing subset of a data set generated
    by `create_dataset`.

    Parameters
    ----------
    path : str
        The path to the data set directory.
    subset : ["train", "val", "test"], optional
        The subset of the data set to load. (Default "train")

    Returns
    -------
    X : torch.Tensor
        The signal samples.
    Y : torch.Tensor
        The corresponding measurement samples.
    """
    X = torch.load(os.path.join(path, "{}_x.pt".format(subset)))
    C = torch.load(os.path.join(path, "{}_c.pt".format(subset)))
    Y = torch.load(os.path.join(path, "{}_y.pt".format(subset)))
    return X, C, Y


# ----- MNIST data -----


class MNISTData(object):
    def __init__(self, path):
        train = MNIST(path, train=True, download=True)
        test = MNIST(path, train=False, download=True)
        train_data = train.data
        train_labels = train.targets
        test_data = test.data
        test_labels = test.targets
        self.combined_data = torch.cat([train_data, test_data], dim=0)
        self.combined_labels = torch.cat([train_labels, test_labels], dim=0)
        self.cur_index = 0

    def __call__(self, n, **kwargs):
        cur_img_flat = self.combined_data[self.cur_index, ...].flatten()
        cur_label = self.combined_labels[self.cur_index, ...]
        self.cur_index += 1
        # normalize images from range [0, 255] to range [0,1]
        return cur_img_flat / 255.0, cur_label


# ----- noise utilities -----


class Jitter(object):
    """ Adds random pertubations to batches of inputs. """

    def __init__(self, eta, scale_lo, scale_hi, t_seed=None):
        self.eta = eta
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi
        self.trng = torch.Generator()
        if t_seed is not None:
            self.trng.manual_seed(t_seed)

    def __call__(self, inp):
        scale = self.scale_lo + (self.scale_hi - self.scale_lo) * torch.rand(
            inp.shape[:-1], generator=self.trng,
        ).to(inp.device)
        noise = torch.randn(inp.shape, generator=self.trng).to(inp.device)
        return inp + self.eta / np.sqrt(
            inp.shape[-1]
        ) * noise * scale.unsqueeze(-1)


# ---- run data generation -----
if __name__ == "__main__":
    import config

    OpA = config.meas_op(config.m, config.n, **config.meas_params)
    np.random.seed(config.numpy_seed)
    torch.manual_seed(config.torch_seed)
    create_dataset(
        config.m,
        config.n,
        OpA,
        config.set_params,
        config.data_gen,
        config.data_params,
    )
