import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision

from matplotlib import rc
from piq import psnr, ssim

from data_management import (
    AlmostFixedMaskDataset,
    CropOrPadAndResimulate,
    Flatten,
    Normalize,
    filter_acquisition_no_fs,
)
from find_adversarial import err_measure_l2
from operators import noise_gaussian, rotate_real, to_complex


# ----- load configuration -----
import config  # isort:skip
import config_robustness as cfg_rob  # isort:skip
from config_robustness import methods  # isort:skip

# ------ general setup ----------

device = cfg_rob.device
torch.manual_seed(2)

save_path = os.path.join(config.RESULTS_PATH, "attacks")
save_results = os.path.join(save_path, "example_V8_S19_gauss.pkl")

do_plot = True
save_plot = True

# ----- attack setup -----

# select samples
sample_vol = 8
sample_sl = 19
it = 50

# noise
noise_type = noise_gaussian

# select range relative noise
noise_min = 1e-3
noise_max = 0.025
noise_steps = 50
noise_rel_grid = torch.tensor(
    np.logspace(np.log10(noise_min), np.log10(noise_max), num=noise_steps)
).float()
noise_rel_show = torch.tensor([0.00, 0.01, 0.015, 0.025, 0.10]).float()
noise_rel = (
    torch.cat([noise_rel_show, noise_rel_grid]).float().unique(sorted=True)
)
print(noise_rel)

# select measure for reconstruction error
err_measure = err_measure_l2

# select reconstruction methods
methods_include = ["L1", "UNet jit", "Tiramisu EE jit", "UNet It jit"]
methods = methods.loc[methods_include]

# select methods excluded from (re-)performing attacks
methods_no_calc = ["L1", "UNet jit", "Tiramisu EE jit", "UNet It jit"]

# ----- perform attack -----

# load data and select sample
test_data_params = {
    "mask_func": cfg_rob.mask_func,
    "seed": 1,
    "filter": [filter_acquisition_no_fs],
    "num_sym_slices": 0,
    "multi_slice_gt": False,
    "keep_mask_as_func": True,
    "transform": torchvision.transforms.Compose(
        [
            CropOrPadAndResimulate((320, 320)),
            Flatten(0, -3),
            Normalize(reduction="mean", use_target=True),
        ],
    ),
}
test_data = AlmostFixedMaskDataset
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
X_0 = X_0.repeat(it, *((X_0.ndim - 1) * (1,)))
Y_0 = cfg_rob.OpA(X_0)

# set range for plotting and similarity indices
v_min = 0.05
v_max = 4.50
print("Pixel values between {} and {}".format(v_min, v_max))

# create result table and load existing results from file
results = pd.DataFrame(columns=["name", "X_err", "X_psnr", "X_ssim", "X", "Y"])
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
        results.loc[idx].X_err = torch.zeros(len(noise_rel), X_0.shape[0])
        results.loc[idx].X_psnr = torch.zeros(len(noise_rel), X_0.shape[0])
        results.loc[idx].X_ssim = torch.zeros(len(noise_rel), X_0.shape[0])
        results.loc[idx].X = torch.zeros(
            len(noise_rel), 1, *X_0.shape[1:], device=torch.device("cpu")
        )
        results.loc[idx].Y = torch.zeros(
            len(noise_rel), 1, *Y_0.shape[1:], device=torch.device("cpu")
        )

        for idx_noise in range(len(noise_rel)):
            print(
                "Method: "
                + idx
                + "; Noise rel {}/{}".format(idx_noise + 1, len(noise_rel))
                + " (= {:1.3f})".format(noise_rel[idx_noise].item()),
                flush=True,
            )

            noise_level = noise_rel[idx_noise] * Y_0.norm(
                p=2, dim=(-2, -1), keepdim=True
            )
            Y = noise_type(Y_0, noise_level)
            X = method.reconstr(Y, noise_rel[idx_noise])

            print(
                (
                    (Y - Y_0).norm(p=2, dim=(-2, -1))
                    / (Y_0).norm(p=2, dim=(-2, -1))
                ).mean()
            )

            results.loc[idx].X_err[idx_noise, ...] = err_measure(X, X_0)
            results.loc[idx].X_psnr[idx_noise, ...] = psnr(
                torch.clamp(rotate_real(X.cpu())[:, 0:1, ...], v_min, v_max),
                torch.clamp(rotate_real(X_0.cpu())[:, 0:1, ...], v_min, v_max),
                data_range=v_max - v_min,
                reduction="none",
            )
            results.loc[idx].X_ssim[idx_noise, ...] = ssim(
                torch.clamp(rotate_real(X.cpu())[:, 0:1, ...], v_min, v_max),
                torch.clamp(rotate_real(X_0.cpu())[:, 0:1, ...], v_min, v_max),
                data_range=v_max - v_min,
                size_average=False,
            )
            results.loc[idx].X[idx_noise, ...] = X[0:1, ...].cpu()
            results.loc[idx].Y[idx_noise, ...] = Y[0:1, ...].cpu()

# save results
for idx in results.index:
    results_save.loc[idx] = results.loc[idx]
os.makedirs(save_path, exist_ok=True)
results_save.to_pickle(save_results)

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
    axins = ax.inset_axes([0.65, 0.65, 0.33, 0.33])
    _implot(axins, X_0)
    axins.set_xlim(75, 130)
    axins.set_ylim(170, 125)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.spines["bottom"].set_color("#a1c9f4")
    axins.spines["top"].set_color("#a1c9f4")
    axins.spines["left"].set_color("#a1c9f4")
    axins.spines["right"].set_color("#a1c9f4")
    ax.indicate_inset_zoom(axins, edgecolor="#a1c9f4")

    if save_plot:
        fig.savefig(
            os.path.join(
                save_path, "fig_example_S{}_gauss_gt.pdf".format(sample_vol)
            ),
            bbox_inches="tight",
            pad_inches=0,
        )

    # method-wise plots
    for (idx, method) in methods.iterrows():

        # +++ reconstructions per noise level +++
        for idx_noise in range(len(noise_rel_show)):

            idx_noise_cur = torch.where(
                noise_rel == noise_rel_show[idx_noise]
            )[0]
            X_cur = results.loc[idx].X[idx_noise_cur, ...].squeeze(0)

            fig, ax = plt.subplots(clear=True, figsize=(2.5, 2.5), dpi=200)

            im = _implot(ax, X_cur)

            ax.text(
                8,
                10,
                "rel.~$\\ell_2$-err: {:.2f}\\%".format(
                    results.loc[idx].X_err[idx_noise_cur, 0].item() * 100
                ),
                fontsize=10,
                color="white",
                horizontalalignment="left",
                verticalalignment="top",
            )
            ax.text(
                8,
                30,
                "PSNR: {:.2f}".format(
                    results.loc[idx].X_psnr[idx_noise_cur, 0].item()
                ),
                fontsize=10,
                color="white",
                horizontalalignment="left",
                verticalalignment="top",
            )
            ax.text(
                8,
                52,
                "SSIM: {:.2f}".format(
                    results.loc[idx].X_ssim[idx_noise_cur, 0].item()
                ),
                fontsize=10,
                color="white",
                horizontalalignment="left",
                verticalalignment="top",
            )

            axins = ax.inset_axes([0.65, 0.65, 0.33, 0.33])
            _implot(axins, X_cur)

            axins.set_xlim(75, 130)
            axins.set_ylim(170, 125)
            axins.set_xticks([])
            axins.set_yticks([])
            axins.spines["bottom"].set_color("#a1c9f4")
            axins.spines["top"].set_color("#a1c9f4")
            axins.spines["left"].set_color("#a1c9f4")
            axins.spines["right"].set_color("#a1c9f4")
            ax.indicate_inset_zoom(axins, edgecolor="#a1c9f4")

            if save_plot:
                fig.savefig(
                    os.path.join(
                        save_path,
                        "fig_example_S{}_gauss_".format(sample_vol)
                        + method.info["name_save"]
                        + "_{:1.2e}".format(noise_rel_show[idx_noise].item())
                        + ".pdf",
                    ),
                    bbox_inches="tight",
                    pad_inches=0,
                )

            # not saved
            plt.title(
                method.info["name_disp"]
                + " for rel. noise level = {:1.3f}".format(
                    noise_rel_show[idx_noise].item()
                )
            )

            # error plot
            fig, ax = plt.subplots(clear=True, figsize=(2.5, 2.5), dpi=200)
            im = _implot(
                ax,
                (rotate_real(X_cur) - rotate_real(X_0)).abs(),
                vmin=0.0,
                vmax=1.25,
            )

            if save_plot:
                fig.savefig(
                    os.path.join(
                        save_path,
                        "fig_example_S{}_gauss_err_".format(sample_vol)
                        + method.info["name_save"]
                        + "_{:1.2e}".format(noise_rel_show[idx_noise].item())
                        + ".pdf",
                    ),
                    bbox_inches="tight",
                    pad_inches=0,
                )

    # +++ error curves for all methods +++
    fig, ax = plt.subplots(clear=True, figsize=(5, 4), dpi=200)

    for (idx, method) in methods.iterrows():

        err_mean = results.loc[idx].X_err[:, :].mean(dim=1)
        err_std = results.loc[idx].X_err[:, :].std(dim=1)

        plt.plot(
            noise_rel[noise_rel <= noise_max],
            err_mean[noise_rel <= noise_max],
            linestyle=method.info["plt_linestyle"],
            linewidth=method.info["plt_linewidth"],
            marker=None,
            color=method.info["plt_color"],
            label=method.info["name_disp"],
        )

        if idx == "L1" or idx == "UNet It jit":
            plt.fill_between(
                noise_rel[noise_rel <= noise_max],
                err_mean[noise_rel <= noise_max]
                + err_std[noise_rel <= noise_max],
                err_mean[noise_rel <= noise_max]
                - err_std[noise_rel <= noise_max],
                alpha=0.10,
                color=method.info["plt_color"],
            )

    plt.yticks(np.arange(0, 1, step=0.02))
    plt.ylim((0.075, 0.145))
    ax.set_xticklabels(["{:,.1%}".format(x) for x in ax.get_xticks()])
    ax.set_yticklabels(["{:,.0%}".format(x) for x in ax.get_yticks()])
    plt.legend(loc="upper left", fontsize=12)

    if save_plot:
        fig.savefig(
            os.path.join(
                save_path, "fig_example_S{}_gauss_curve.pdf".format(sample_vol)
            ),
            bbox_inches="tight",
        )

    plt.show()
