import os

import numpy as np
import pandas as pd
import torch
import torchvision

from data_management import (
    CropOrPadAndResimulate,
    Flatten,
    Normalize,
    RandomMaskDataset,
    filter_acquisition_no_fs,
)
from operators import noise_gaussian, to_complex
from reconstruction_methods import admm_l1_rec_diag, grid_search


# ----- load configuration -----
import config  # isort:skip
import config_robustness as cfg_rob  # isort:skip

# ------ setup ----------
device = torch.device("cuda:0")

file_name = "grid_search_l1_fourier_"
save_path = os.path.join(config.RESULTS_PATH, "grid_search_l1")


# ----- load test data --------
test_data_params = {
    "mask_func": cfg_rob.mask_func,
    "filter": [filter_acquisition_no_fs],
    "num_sym_slices": 0,
    "multi_slice_gt": False,
    "simulate_gt": True,
    "keep_mask_as_func": True,
    "transform": torchvision.transforms.Compose(
        [
            CropOrPadAndResimulate((368, 368)),
            Flatten(0, -3),
            Normalize(reduction="mean", use_target=False),
        ],
    ),
}
test_data = RandomMaskDataset
test_data = test_data("val", **test_data_params)


# select samples
sample_vol = 0
sample_sl = 15

# load sample
lo, hi = test_data.get_slices_in_volume(sample_vol)

X_0 = test_data[lo + sample_sl][2].unsqueeze(0)
X_0 = to_complex(X_0.to(device))

# ----- noise setup --------
noise_rels = [0.0, 0.025]


def meas_noise(y, noise_level):
    return noise_gaussian(y, noise_level)


# ----- set up reconstruction method and grid params --------


def _reconstruct(y, lam, rho):
    x, _ = admm_l1_rec_diag(
        y,
        cfg_rob.OpA,
        cfg_rob.OpTV,
        cfg_rob.OpA.adj(y),
        cfg_rob.OpTV(cfg_rob.OpA.adj(y)),
        lam,
        rho,
        iter=400,
        silent=True,
    )
    return x


# parameter search grid
grid = {
    "lam": np.logspace(-6, -4, 30),
    "rho": np.logspace(-3, 0, 30),
}


def combine_results():
    results = pd.DataFrame(
        columns=[
            "noise_rel",
            "grid_param",
            "err_min",
            "grid",
            "err",
            "psnr",
            "ssim",
        ]
    )
    for idx in range(len(noise_rels)):
        results_cur = pd.read_pickle(
            os.path.join(
                save_path, "{}{:.2f}.pkl".format(file_name, noise_rels[idx])
            )
        )
        results.loc[idx] = results_cur.loc[0]

    os.makedirs(save_path, exist_ok=True)
    results.to_pickle(os.path.join(save_path, file_name + "all.pkl"))

    return results


# ------ perform grid search ---------

if __name__ == "__main__":

    os.makedirs(save_path, exist_ok=True)

    for noise_rel in noise_rels:

        noise_level = noise_rel * cfg_rob.OpA(X_0).norm(
            p=2, dim=(-2, -1), keepdim=True
        )
        Y_ref = meas_noise(cfg_rob.OpA(X_0), noise_level)
        grid_param, err_min, err, psnr, ssim = grid_search(
            X_0, Y_ref, _reconstruct, grid
        )

        results = pd.DataFrame(
            columns=[
                "noise_rel",
                "grid_param",
                "err_min",
                "grid",
                "err",
                "psnr",
                "ssim",
            ]
        )
        results.loc[0] = {
            "noise_rel": noise_rel,
            "grid_param": grid_param,
            "err_min": err_min,
            "grid": grid,
            "err": err,
            "psnr": psnr,
            "ssim": ssim,
        }

        results.to_pickle(
            os.path.join(
                save_path, "{}{:.2f}.pkl".format(file_name, noise_rel)
            )
        )
