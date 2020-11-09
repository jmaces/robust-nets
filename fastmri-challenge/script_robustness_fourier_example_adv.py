import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision

from matplotlib import rc
from piq import psnr, ssim

from data_management import (
    CropOrPadAndResimulate,
    Flatten,
    Normalize,
    RandomMaskDataset,
    filter_acquisition_no_fs,
)
from find_adversarial import err_measure_l2, grid_attack
from operators import rotate_real, to_complex


# ----- load configuration -----
import config  # isort:skip
import config_robustness as cfg_rob  # isort:skip
from config_robustness import methods  # isort:skip

# ------ general setup ----------

device = cfg_rob.device

save_path = os.path.join(config.RESULTS_PATH, "attacks")
save_results = os.path.join(save_path, "challenge_example_V0S15_adv.pkl")

do_plot = True
save_plot = True

# ----- attack setup -----

# select samples
sample_vol = 0
sample_sl = 15
it_init = 6
keep_init = 3

# select range relative noise
noise_rel = torch.tensor([0.00, 0.025]).float().unique(sorted=True)
print(noise_rel)

# select measure for reconstruction error
err_measure = err_measure_l2

# select reconstruction methods
methods_include = ["L1", "Tiramisu jit"]
methods = methods.loc[methods_include]

# select methods excluded from (re-)performing attacks
methods_no_calc = ["L1", "Tiramisu jit"]

# adjust constrast
v_min = 0.05
v_max = 4.5

# ----- perform attack -----

# load data and select sample
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


lo, hi = test_data.get_slices_in_volume(sample_vol)
print(
    "volume slices from {} to {}, selected {}".format(lo, hi, lo + sample_sl)
)
X_VOL = to_complex(
    torch.stack([test_data[sl_idx][2] for sl_idx in range(lo, hi)], dim=0)
).to(device)
X_MAX = rotate_real(X_VOL)[:, 0:1, ...].max().cpu()

X_0 = to_complex(test_data[lo + sample_sl][2].to(device)).unsqueeze(0)
X_0 = X_0.repeat(it_init, *((X_0.ndim - 1) * (1,)))
Y_0 = cfg_rob.OpA(X_0)


# create result table and load existing results from file
results = pd.DataFrame(
    columns=[
        "name",
        "X_adv_err",
        "X_ref_err",
        "X_adv_psnr",
        "X_ref_psnr",
        "X_adv_ssim",
        "X_ref_ssim",
        "X_adv",
        "X_ref",
        "Y_adv",
        "Y_ref",
    ]
)
results.name = methods.index
results = results.set_index("name")
# load existing results from file
if os.path.isfile(save_results):
    results_save = pd.read_pickle(save_results)
    for idx in results_save.index:
        if idx in results.index:
            results.loc[idx] = results_save.loc[idx]
else:
    results_save = results


# perform attacks
for (idx, method) in methods.iterrows():
    if idx not in methods_no_calc:
        (
            results.loc[idx].X_adv_err,
            results.loc[idx].X_ref_err,
            results.loc[idx].X_adv,
            results.loc[idx].X_ref,
            results.loc[idx].Y_adv,
            results.loc[idx].Y_ref,
        ) = grid_attack(
            method,
            noise_rel,
            X_0,
            Y_0,
            store_data=True,
            keep_init=keep_init,
            err_measure=err_measure,
        )

        results.loc[idx].X_adv_psnr = torch.zeros(len(noise_rel), X_0.shape[0])
        results.loc[idx].X_ref_psnr = torch.zeros(len(noise_rel), X_0.shape[0])
        results.loc[idx].X_adv_ssim = torch.zeros(len(noise_rel), X_0.shape[0])
        results.loc[idx].X_ref_ssim = torch.zeros(len(noise_rel), X_0.shape[0])

        for idx_noise in range(len(noise_rel)):
            results.loc[idx].X_adv_psnr[idx_noise, ...] = psnr(
                torch.clamp(
                    rotate_real(results.loc[idx].X_adv[idx_noise, ...])[
                        :, 0:1, ...
                    ],
                    v_min,
                    v_max,
                ),
                torch.clamp(rotate_real(X_0.cpu())[:, 0:1, ...], v_min, v_max),
                data_range=v_max - v_min,
                reduction="none",
            )
            results.loc[idx].X_ref_psnr[idx_noise, ...] = psnr(
                torch.clamp(
                    rotate_real(results.loc[idx].X_ref[idx_noise, ...])[
                        :, 0:1, ...
                    ],
                    v_min,
                    v_max,
                ),
                torch.clamp(rotate_real(X_0.cpu())[:, 0:1, ...], v_min, v_max),
                data_range=v_max - v_min,
                reduction="none",
            )
            results.loc[idx].X_adv_ssim[idx_noise, ...] = ssim(
                torch.clamp(
                    rotate_real(results.loc[idx].X_adv[idx_noise, ...])[
                        :, 0:1, ...
                    ],
                    v_min,
                    v_max,
                ),
                torch.clamp(rotate_real(X_0.cpu())[:, 0:1, ...], v_min, v_max),
                data_range=v_max - v_min,
                size_average=False,
            )
            results.loc[idx].X_ref_ssim[idx_noise, ...] = ssim(
                torch.clamp(
                    rotate_real(results.loc[idx].X_ref[idx_noise, ...])[
                        :, 0:1, ...
                    ],
                    v_min,
                    v_max,
                ),
                torch.clamp(rotate_real(X_0.cpu())[:, 0:1, ...], v_min, v_max),
                data_range=v_max - v_min,
                size_average=False,
            )

# save results
for idx in results.index:
    results_save.loc[idx] = results.loc[idx]
os.makedirs(save_path, exist_ok=True)
results_save.to_pickle(save_results)

# select the worst example for each noise level and method (rel err)
results_max = pd.DataFrame(
    columns=["name", "X_adv_err", "X_adv_psnr", "X_adv_ssim", "X_adv", "Y_adv"]
)
results_max.name = methods.index
results_max = results_max.set_index("name")
for (idx, method) in methods.iterrows():
    _, idx_adv = results.loc[idx].X_adv_err.max(dim=1)

    idx_noise = range(len(noise_rel))
    results_max.loc[idx].X_adv_err = results.loc[idx].X_adv_err[
        idx_noise, idx_adv, ...
    ]
    results_max.loc[idx].X_adv_psnr = results.loc[idx].X_adv_psnr[
        idx_noise, idx_adv, ...
    ]
    results_max.loc[idx].X_adv_ssim = results.loc[idx].X_adv_ssim[
        idx_noise, idx_adv, ...
    ]
    results_max.loc[idx].X_adv = results.loc[idx].X_adv[
        idx_noise, idx_adv, ...
    ]
    results_max.loc[idx].Y_adv = results.loc[idx].Y_adv[
        idx_noise, idx_adv, ...
    ]

# ----- plotting -----


def _implot(sub, im, vmin=v_min, vmax=v_max):
    if im.shape[-3] == 2:  # complex image
        image = sub.imshow(
            torch.sqrt(im.pow(2).sum(-3))[0, :, :].detach().cpu(),
            vmin=vmin,
            vmax=vmax,
        )
    else:  # real image
        image = sub.imshow(im[0, 0, :, :].detach().cpu(), vmin=vmin, vmax=vmax)

    image.set_cmap("gray")
    sub.set_xticks([])
    sub.set_yticks([])
    return image


if do_plot:

    # LaTeX typesetting
    rc("font", **{"family": "serif", "serif": ["Palatino"]})
    rc("text", usetex=True)

    X_0 = X_0.cpu()
    Y_0 = Y_0.cpu()

    # +++ ground truth +++
    fig, ax = plt.subplots(clear=True, figsize=(2.5, 2.5), dpi=200)
    im = _implot(ax, X_0)

    if save_plot:
        fig.savefig(
            os.path.join(
                save_path,
                "fig_example_challenge_V{}S{}_adv_gt.pdf".format(
                    sample_vol, sample_sl
                ),
            ),
            bbox_inches="tight",
            pad_inches=0,
        )

    # method-wise plots
    for (idx, method) in methods.iterrows():

        # +++ reconstructions per noise level +++
        for idx_noise in range(len(noise_rel)):

            results_ref = results_max
            X_cur = results_ref.loc[idx].X_adv[idx_noise, ...].unsqueeze(0)

            # adv
            fig, ax = plt.subplots(clear=True, figsize=(2.5, 2.5), dpi=200)
            im = _implot(ax, X_cur)

            ax.text(
                360,
                10,
                "rel.~$\\ell_2$-err: {:.2f}\\%".format(
                    results_ref.loc[idx].X_adv_err[idx_noise].item() * 100
                ),
                fontsize=10,
                color="white",
                horizontalalignment="right",
                verticalalignment="top",
            )
            ax.text(
                360,
                30,
                "PSNR: {:.2f}".format(
                    results_ref.loc[idx].X_adv_psnr[idx_noise].item()
                ),
                fontsize=10,
                color="white",
                horizontalalignment="right",
                verticalalignment="top",
            )
            ax.text(
                360,
                53,
                "SSIM: {:.2f}".format(
                    results_ref.loc[idx].X_adv_ssim[idx_noise].item()
                ),
                fontsize=10,
                color="white",
                horizontalalignment="right",
                verticalalignment="top",
            )

            if save_plot:
                fig.savefig(
                    os.path.join(
                        save_path,
                        "fig_example_challenge_V{}S{}_adv_".format(
                            sample_vol, sample_sl
                        )
                        + method.info["name_save"]
                        + "_{:.0e}".format(noise_rel[idx_noise].item())
                        + ".pdf",
                    ),
                    bbox_inches="tight",
                    pad_inches=0,
                )

            # not saved
            fig.suptitle(
                method.info["name_disp"]
                + " for rel. noise level = {:1.3f}".format(
                    noise_rel[idx_noise].item()
                )
            )

    plt.show()
