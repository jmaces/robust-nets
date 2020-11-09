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
from operators import im2vec, noise_gaussian, rotate_real, to_complex


# ----- load configuration -----
import config  # isort:skip
import config_robustness as cfg_rob  # isort:skip
from config_robustness import methods  # isort:skip

# ------ general setup ----------

device = cfg_rob.device
torch.manual_seed(2)

save_path = os.path.join(config.RESULTS_PATH, "attacks")
save_results = os.path.join(save_path, "table_gauss.pkl")

do_plot = True
save_plot = True
save_table = True


# ----- attack setup -----

# ----- load test data --------
test_data_params = {
    "mask_func": cfg_rob.mask_func,
    "seed": 1,
    "filter": [filter_acquisition_no_fs],
    "num_sym_slices": 0,
    "multi_slice_gt": False,
    "keep_mask_as_func": True,
    "transform": torchvision.transforms.Compose(
        [
            CropOrPadAndResimulate(cfg_rob.n),
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
Y_0 = im2vec(to_complex(Y_0.to(device)))[
    ..., im2vec(cfg_rob.mask[0, 0, :, :].bool())
]

# no meas samples
it = 50

noise_type = noise_gaussian

# select range relative noise
noise_rel = torch.tensor([0.00, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.025])

# select measure for reconstruction error
err_measure = err_measure_l2

# select reconstruction methods
methods_include = [
    "L1",
    "UNet jit",
    "UNet EE jit",
    "Tiramisu jit",
    "Tiramisu EE jit",
    "UNet It jit",
]
methods_plot = ["L1", "UNet jit", "Tiramisu EE jit", "UNet It jit"]
methods = methods.loc[methods_include]

# select methods excluded from (re-)performing attacks
methods_no_calc = [
    "L1",
    "UNet jit",
    "UNet EE jit",
    "Tiramisu jit",
    "Tiramisu EE jit",
    "UNet It jit",
]

# ----- perform attack -----

# create result table
results = pd.DataFrame(columns=["name", "X_err", "X_psnr", "X_ssim"])
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

        s_len = X_0.shape[0]
        results.loc[idx].X_err = torch.zeros(len(noise_rel), s_len)
        results.loc[idx].X_psnr = torch.zeros(len(noise_rel), s_len)
        results.loc[idx].X_ssim = torch.zeros(len(noise_rel), s_len)

        for s in range(s_len):
            print("Sample: {}/{}".format(s + 1, s_len))
            X_0_s = X_0[s : s + 1, ...].repeat(it, *((X_0.ndim - 1) * (1,)))
            Y_0_s = Y_0[s : s + 1, ...].repeat(it, *((Y_0.ndim - 1) * (1,)))

            for idx_noise in range(len(noise_rel)):
                print(
                    "Method: "
                    + idx
                    + "; Noise rel {}/{}".format(idx_noise + 1, len(noise_rel))
                    + " (= {:1.3f})".format(noise_rel[idx_noise].item()),
                    flush=True,
                )

                noise_level = noise_rel[idx_noise] * Y_0_s.norm(
                    p=2, dim=(-2, -1), keepdim=True
                )
                Y = noise_type(Y_0_s, noise_level)
                X = method.reconstr(Y, noise_rel[idx_noise])

                print(
                    (
                        (Y - Y_0_s).norm(p=2, dim=(-2, -1))
                        / (Y_0_s).norm(p=2, dim=(-2, -1))
                    ).mean()
                )

                results.loc[idx].X_err[idx_noise, s] = err_measure(
                    X, X_0_s
                ).mean()
                results.loc[idx].X_psnr[idx_noise, s] = psnr(
                    rotate_real(X.cpu())[:, 0:1, ...],
                    rotate_real(X_0_s.cpu())[:, 0:1, ...],
                    data_range=4.5,
                    reduction="mean",
                )  # normalization as in ex-script
                results.loc[idx].X_ssim[idx_noise, s] = ssim(
                    rotate_real(X.cpu())[:, 0:1, ...],
                    rotate_real(X_0_s.cpu())[:, 0:1, ...],
                    data_range=4.5,
                    size_average=True,
                )  # normalization as in ex-script


# save results
for idx in results.index:
    results_save.loc[idx] = results.loc[idx]
os.makedirs(save_path, exist_ok=True)
results_save.to_pickle(save_results)

# ----- plotting -----

if do_plot:

    # LaTeX typesetting
    rc("font", **{"family": "serif", "serif": ["Palatino"]})
    rc("text", usetex=True)

    # +++ visualization of table +++
    fig, ax = plt.subplots(clear=True, figsize=(5, 4), dpi=200)

    for (idx, method) in methods.loc[methods_plot].iterrows():

        err_mean = results.loc[idx].X_err[:, :].mean(dim=-1)
        err_std = results.loc[idx].X_err[:, :].std(dim=-1)

        plt.plot(
            noise_rel,
            err_mean,
            linestyle=method.info["plt_linestyle"],
            linewidth=method.info["plt_linewidth"],
            marker=method.info["plt_marker"],
            color=method.info["plt_color"],
            label=method.info["name_disp"],
        )
        if idx == "L1" or idx == "UNet It jit":
            plt.fill_between(
                noise_rel,
                err_mean + err_std,
                err_mean - err_std,
                alpha=0.10,
                color=method.info["plt_color"],
            )

    plt.yticks(np.arange(0, 1, step=0.02))
    plt.ylim((0.055, 0.145))
    ax.set_xticklabels(["{:,.1%}".format(x) for x in ax.get_xticks()])
    ax.set_yticklabels(["{:,.0%}".format(x) for x in ax.get_yticks()])
    plt.legend(loc="upper left", fontsize=12)

    if save_plot:
        fig.savefig(
            os.path.join(save_path, "fig_table_gauss.pdf"), bbox_inches="tight"
        )

    plt.show()

if save_table:
    df = results.applymap(
        lambda res: {"mean": res.mean(dim=-1), "std": res.std(dim=-1)}
    )

    # extract mean and std
    df_ref_mean = (
        df.stack()
        .apply(pd.Series)["mean"]
        .apply(
            lambda res: pd.Series(
                res,
                index=[
                    "{{{:.1f}\\%}}".format(noise * 100) for noise in noise_rel
                ],
            )
        )
    )
    df_ref_std = (
        df.stack()
        .apply(pd.Series)["std"]
        .apply(
            lambda res: pd.Series(
                res,
                index=[
                    "{{{:.1f}\\%}}".format(noise * 100) for noise in noise_rel
                ],
            )
        )
    )

    # find best method per noise level and metric
    best_ref_l2 = df_ref_mean.xs("X_err", level=1).idxmin()
    best_ref_ssim = df_ref_mean.xs("X_ssim", level=1).idxmax()
    best_ref_psnr = df_ref_mean.xs("X_psnr", level=1).idxmax()

    # combine mean and std data into "mean\pmstd" strings
    for (idx, method) in methods.iterrows():
        df_ref_mean.loc[idx, "X_err"] = df_ref_mean.loc[idx, "X_err"].apply(
            lambda res: res * 100
        )
        df_ref_std.loc[idx, "X_err"] = df_ref_std.loc[idx, "X_err"].apply(
            lambda res: res * 100
        )

    df_ref_combined = df_ref_mean.combine(
        df_ref_std,
        lambda col1, col2: col1.combine(
            col2, lambda el1, el2: "{:.2f} \\pm {:.2f}".format(el1, el2)
        ),
    )

    # format best value per noise level and metric as bold
    for col, idx in best_ref_l2.iteritems():
        df_ref_combined.at[(idx, "X_err"), col] = (
            "\\bfseries " + df_ref_combined.at[(idx, "X_err"), col]
        )
    for col, idx in best_ref_ssim.iteritems():
        df_ref_combined.at[(idx, "X_ssim"), col] = (
            "\\bfseries " + df_ref_combined.at[(idx, "X_ssim"), col]
        )
    for col, idx in best_ref_psnr.iteritems():
        df_ref_combined.at[(idx, "X_psnr"), col] = (
            "\\bfseries " + df_ref_combined.at[(idx, "X_psnr"), col]
        )

    # rename rows and columns
    df_ref_combined = df_ref_combined.rename(
        index={
            "X_err": "rel.~$\\l{2}$-err. [\\%]",
            "X_ssim": "SSIM",
            "X_psnr": "PSNR",
        }
    )
    df_ref_combined = df_ref_combined.rename(
        index=methods["info"].apply(lambda res: res["name_disp"]).to_dict()
    )

    # save latex tabular
    df_ref_combined.to_latex(
        os.path.join(save_path, "table_ref.tex"),
        column_format=2 * "l" + len(noise_rel) * "S[separate-uncertainty]",
        multirow=True,
        escape=False,
    )
