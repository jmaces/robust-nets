import os

import numpy as np
import pandas as pd
import torch
import torchvision

from data_management import (
    AlmostFixedMaskDataset,
    CropOrPadAndResimulate,
    Flatten,
    Normalize,
    filter_acquisition_no_fs,
)
from operators import (
    Fourier,
    RadialMaskFunc,
    TVAnalysisPeriodic,
    im2vec,
    noise_gaussian,
    to_complex,
    unprep_fft_channel,
)
from reconstruction_methods import admm_l1_rec_diag, grid_search


# ----- load configuration -----
import config  # isort:skip

# ------ setup ----------
device = torch.device("cuda:0")

file_name = "grid_search_l1_fourier_"
save_path = os.path.join(config.RESULTS_PATH, "grid_search_l1")

# ----- operators --------
n = (320, 320)
mask_func = RadialMaskFunc(n, 50)
mask = unprep_fft_channel(mask_func((1, 1) + n + (1,)))
OpA = Fourier(mask)
OpTV = TVAnalysisPeriodic(n, device=device)

# ----- load test data --------
test_data_params = {
    "mask_func": mask_func,
    "seed": 1,
    "filter": [filter_acquisition_no_fs],
    "num_sym_slices": 0,
    "multi_slice_gt": False,
    "keep_mask_as_func": True,
    "transform": torchvision.transforms.Compose(
        [
            CropOrPadAndResimulate(n),
            Flatten(0, -3),
            Normalize(reduction="mean", use_target=True),
        ],
    ),
}
test_data = AlmostFixedMaskDataset
test_data = test_data("val", **test_data_params)

vols = range(30)
slices_in_vols = [test_data.get_slices_in_volume(vol_idx) for vol_idx in vols]
slices_selected = [
    range((lo + hi) // 2, (lo + hi) // 2 + 1) for lo, hi in slices_in_vols
]
samples = np.concatenate(slices_selected)

X_0 = torch.stack([test_data[s][2] for s in samples])
X_0 = to_complex(X_0.to(device))

Y_0 = torch.stack([test_data[s][0] for s in samples])
Y_0 = im2vec(to_complex(Y_0.to(device)))[..., im2vec(mask[0, 0, :, :].bool())]


# ----- noise setup --------
noise_min = 1e-3
noise_max = 0.03
noise_steps = 50
noise_rel = torch.tensor(
    np.logspace(np.log10(noise_min), np.log10(noise_max), num=noise_steps)
).float()
noise_rel = (
    torch.cat(
        [
            torch.zeros(1).float(),
            noise_rel,
            0.060 * torch.ones(1).float(),
            0.075 * torch.ones(1).float(),
            0.100 * torch.ones(1).float(),
        ]
    )
    .float()
    .to(device)
)


def meas_noise(y, noise_level):
    return noise_gaussian(y, noise_level)


# ----- set up reconstruction method and grid params --------


def _reconstruct(y, lam, rho):
    x, _ = admm_l1_rec_diag(
        y,
        OpA,
        OpTV,
        OpA.adj(y),
        OpTV(OpA.adj(y)),
        lam,
        rho,
        iter=400,
        silent=True,
    )
    return x


# parameter search grid
grid = {
    "lam": np.logspace(np.log10(0.005), np.log10(0.2), 30),
    "rho": np.logspace(np.log10(0.5), np.log10(30), 30),
}


def combine_results():
    results = pd.DataFrame(
        columns=["noise_rel", "grid_param", "err_min", "grid", "err"]
    )
    for idx in range(len(noise_rel)):
        results_cur = pd.read_pickle(
            os.path.join(save_path, file_name + str(idx) + ".pkl")
        )
        results.loc[idx] = results_cur.loc[idx]

    os.makedirs(save_path, exist_ok=True)
    results.to_pickle(os.path.join(save_path, file_name + "all.pkl"))

    return results


# ------ perform grid search ---------

if __name__ == "__main__":

    idx_noise = (int(os.environ.get("SGE_TASK_ID")) - 1,)

    for idx in idx_noise:
        noise_level = noise_rel[idx] * Y_0.norm(
            p=2, dim=(-2, -1), keepdim=True
        )
        Y_ref = meas_noise(Y_0, noise_level)
        grid_param, err_min, err = grid_search(X_0, Y_ref, _reconstruct, grid)

        results = pd.DataFrame(
            columns=["noise_rel", "grid_param", "err_min", "grid", "err"]
        )
        results.loc[idx] = {
            "noise_rel": noise_rel[idx],
            "grid_param": grid_param,
            "err_min": err_min,
            "grid": grid,
            "err": err,
        }

        os.makedirs(save_path, exist_ok=True)
        results.to_pickle(
            os.path.join(save_path, file_name + str(idx) + ".pkl")
        )
