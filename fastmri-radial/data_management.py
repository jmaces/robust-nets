import itertools
import os
import pathlib
import random

from abc import ABCMeta, abstractmethod

import h5py
import numpy as np
import pandas as pd
import torch

from fastmri_utils.data import transforms

from config import DATA_PATH, RESULTS_PATH
from operators import (
    prep_fft_channel,
    rotate_real,
    to_complex,
    unprep_fft_channel,
)


# ----- utilities -----


def filter_acquisition_fs(df):
    return df[
        (df["acquisition"] == "CORPDFS_FBK")
        | (df["acquisition"] == "CORPDFS_FBKREPEAT")
    ]


def filter_acquisition_no_fs(df):
    return df[df["acquisition"] == "CORPD_FBK"]


def explore_dataset(path, savepath):

    collected_data = []
    for fname in pathlib.Path(path).iterdir():
        f = h5py.File(fname, "r")
        kspace = f["kspace"]
        data_dict = {
            "fname": fname,
            "kspace_shape": kspace.shape,
        }
        data_dict.update(f.attrs)
        collected_data.append(data_dict)

    df = pd.DataFrame(collected_data)
    df.to_pickle(savepath)


# ----- sampling and shuffling -----


class RandomBlockSampler(torch.utils.data.Sampler):
    """ Samples elements randomly from consecutive blocks. """

    def __init__(self, data_source, block_size):
        self.data_source = data_source
        self.block_size = block_size

    @property
    def num_samples(self):
        # dataset size might change at runtime
        return len(self.data_source)

    def __iter__(self):
        blocks = itertools.groupby(
            range(self.num_samples), lambda k: k // self.block_size
        )
        block_list = [list(group) for key, group in blocks]
        random.shuffle(block_list)
        return itertools.chain(*block_list)

    def __len__(self):
        return self.num_samples


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


class MaskedFourierInversion(object):
    """ Inverse subsampled Fourier transform on (meas, target, mask) triples.

    Inverts zero filled meas to image domain and returns (inv, target) pair.

    """

    def __init__(self):
        pass

    def __call__(self, inputs):
        kspace, mask, target = inputs
        inv = unprep_fft_channel(transforms.ifft2(prep_fft_channel(kspace)))
        return inv, target


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


class CropOrPadAndResimulate(object):
    """ Generates simulated measurements from (input, mask, target) triples.

    Returns (new_input, new_mask, target) triples.

    Uses target, e.g. obtained from RSS reconstructions, crops or pads it,
    recalculates the measurements of the cropped data, applies the cropped
    mask.

    Requires mask to be a mask generating function, not a precomputed mask.
    (See AbstractMRIDataset proprety `keep_mask_as_func`.)

    """

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, inputs):
        kspace, mask, target = inputs

        # pad if necessary
        p1 = max(0, self.shape[0] - target.shape[-2])
        p2 = max(0, self.shape[1] - target.shape[-1])
        target_padded = torch.nn.functional.pad(
            target, (p2 // 2, -(-p2 // 2), p1 // 2, -(-p1 // 2)),
        )

        # crop if necessary
        target_cropped = transforms.center_crop(target_padded, self.shape)

        # resimulate
        kspace_cropped = transforms.fft2(prep_fft_channel(target_cropped))
        new_mask = mask(kspace_cropped.shape).expand_as(kspace_cropped)
        new_kspace = unprep_fft_channel(kspace_cropped * new_mask)
        new_mask = unprep_fft_channel(new_mask)

        tgs = target.shape[-3]
        if not tgs == 2:
            target_cropped = target_cropped[
                ..., ((tgs // 2) // 2) * 2 : ((tgs // 2) // 2) * 2 + 2, :, :
            ]

        return new_kspace, new_mask, target_cropped


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
        if self.use_target:
            tar = inputs[-1]
        else:
            tar = unprep_fft_channel(
                transforms.ifft2(prep_fft_channel(inputs[0]))
            )
        norm = torch.norm(tar, p=self.p)
        if self.reduction == "mean" and not self.p == "inf":
            norm /= np.prod(tar.shape) ** (1 / self.p)
        if len(inputs) == 2:
            return inputs[0] / norm, inputs[1] / norm
        else:
            return inputs[0] / norm, inputs[1], inputs[2] / norm


class Jitter(object):
    """ Adds random pertubations to the input of (input, mask, target) triples.
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
        kspace, mask, target = inputs
        m = mask.bool().sum()  # number of sampled measurements
        scale = (
            self.scale_lo + (self.scale_hi - self.scale_lo) * self.rng.rand()
        )
        noise = (
            torch.randn(kspace.shape, generator=self.trng).to(kspace.device)
            * mask
        )
        kspace_noisy = kspace + self.eta / np.sqrt(m) * noise * scale
        return kspace_noisy, mask, target


# ----- datasets -----
class AbstractMRIDataset(torch.utils.data.Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        subset,
        filter=None,
        num_sym_slices=0,
        multi_slice_gt=False,
        keep_mask_as_func=False,
        transform=None,
    ):
        # load data set meta data
        if subset == "train":
            metafile = "singlecoil_train_metadata.pkl"
        elif subset == "train_larger":
            metafile = "singlecoil_train_larger_metadata.pkl"
        elif subset == "val":
            metafile = "singlecoil_val_metadata.pkl"
        elif subset == "val_smaller":
            metafile = "singlecoil_val_smaller_metadata.pkl"
        elif subset == "test":
            metafile = "singlecoil_test_v2_metadata.pkl"
        df = pd.read_pickle(os.path.join(RESULTS_PATH, metafile))

        if isinstance(filter, (list, tuple)):
            for f in filter:
                df = f(df)
            self.df = df
        else:
            self.df = filter(df) if filter is not None else df
        self.num_sym_slices = num_sym_slices
        self.multi_slice_gt = multi_slice_gt
        self.keep_mask_as_func = keep_mask_as_func
        self.transform = transform

    def __len__(self):
        # get total number of slices in the data set volumes
        return self.df["kspace_shape"].apply(lambda s: s[0]).sum()

    @property
    def volumes(self):
        return len(self.df)

    def get_slices_in_volume(self, vol_idx):
        # cumulative number of slices in the data set volumes
        vols = self.df["kspace_shape"].apply(lambda s: s[0]).cumsum()

        # first slice of volume
        lo = vols.iloc[vol_idx - 1] if vol_idx > 0 else 0

        # last slice of volume
        hi = vols.iloc[vol_idx]
        return lo, hi

    def __getitem__(self, idx):

        # cumulative number of slices in the data set volumes
        vols = self.df["kspace_shape"].apply(lambda s: s[0]).cumsum() - 1

        # get index of volume and slice within volume
        vol_idx = vols.searchsorted(idx)
        sl_idx = idx if vol_idx == 0 else idx - (vols.iloc[vol_idx - 1] + 1)

        # select slices for multi-slice mode
        sl_from = sl_idx - self.num_sym_slices
        sl_to = sl_idx + self.num_sym_slices + 1

        # load data
        fname = self.df["fname"].iloc[vol_idx]
        data = h5py.File(fname, "r")

        # read out slices and pad if necessary
        sl_num = data["kspace"].shape[0]
        kspace_vol = transforms.to_tensor(
            np.asarray(
                data["kspace"][max(0, sl_from) : min(sl_to, sl_num), ...]
            )
        )
        kspace_vol_padded = torch.nn.functional.pad(
            kspace_vol,
            (0, 0, 0, 0, 0, 0, max(0, -sl_from), max(0, sl_to - sl_num)),
        )

        if self.multi_slice_gt:
            raise NotImplementedError("Multi slice not yet fully implemented.")
            # gt = transforms.ifft2(kspace_vol_padded)
        else:
            gt = (
                transforms.to_tensor(
                    np.asarray(
                        data["reconstruction_rss"][sl_idx : sl_idx + 1, ...]
                    )
                )
                if "reconstruction_rss" in data
                else None
            )

        out = self._process_data(kspace_vol_padded, gt)

        return self.transform(out) if self.transform is not None else out

    @abstractmethod
    def _process_data(self, kspace_data, gt_data):
        """ Processing of raw data, e.g. masking. """
        pass


class RandomMaskDataset(AbstractMRIDataset):
    def __init__(self, subset, mask_func, **kwargs):
        super(RandomMaskDataset, self).__init__(subset, **kwargs)
        self.mask_func = mask_func

    def _process_data(self, kspace_data, gt_data):
        mask = self.mask_func(kspace_data.shape).expand_as(kspace_data)
        kspace_data = unprep_fft_channel(kspace_data.unsqueeze(0))
        mask = unprep_fft_channel(mask.unsqueeze(0))
        kspace_masked = kspace_data * mask
        gt_data = to_complex(gt_data)
        if self.keep_mask_as_func:
            mask = self.mask_func
        return kspace_masked, mask, gt_data


class AlmostFixedMaskDataset(AbstractMRIDataset):
    def __init__(self, subset, mask_func, seed, **kwargs):
        super(AlmostFixedMaskDataset, self).__init__(subset, **kwargs)
        self.mask_func = mask_func
        self.seed = seed

    def _process_data(self, kspace_data, gt_data):
        mask = self.mask_func(kspace_data.shape, seed=self.seed).expand_as(
            kspace_data
        )
        kspace_data = unprep_fft_channel(kspace_data.unsqueeze(0))
        mask = unprep_fft_channel(mask.unsqueeze(0))
        kspace_masked = kspace_data * mask
        gt_data = to_complex(gt_data)
        if self.keep_mask_as_func:

            def mask(shape):
                return self.mask_func(shape, seed=self.seed)

        return kspace_masked, mask, gt_data


# ---- run data exploration -----

if __name__ == "__main__":
    os.makedirs(RESULTS_PATH, exist_ok=True)
    explore_dataset(
        os.path.join(DATA_PATH, "singlecoil_train"),
        os.path.join(RESULTS_PATH, "singlecoil_train_metadata.pkl"),
    )
    explore_dataset(
        os.path.join(DATA_PATH, "singlecoil_val"),
        os.path.join(RESULTS_PATH, "singlecoil_val_metadata.pkl"),
    )
    explore_dataset(
        os.path.join(DATA_PATH, "singlecoil_test_v2"),
        os.path.join(RESULTS_PATH, "singlecoil_test_v2_metadata.pkl"),
    )

    # make additional metafiles with a larger train and smaller val set
    train_meta = pd.read_pickle(
        os.path.join(RESULTS_PATH, "singlecoil_train_metadata.pkl")
    )
    val_meta = pd.read_pickle(
        os.path.join(RESULTS_PATH, "singlecoil_val_metadata.pkl")
    )
    train_larger_meta = train_meta.append(val_meta.iloc[:50])
    val_smaller_meta = val_meta.iloc[50:]
    train_larger_meta.to_pickle(
        os.path.join(RESULTS_PATH, "singlecoil_train_larger_metadata.pkl")
    )
    val_smaller_meta.to_pickle(
        os.path.join(RESULTS_PATH, "singlecoil_val_smaller_metadata.pkl")
    )
