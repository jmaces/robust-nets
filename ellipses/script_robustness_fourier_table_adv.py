import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from matplotlib import rc
from piq import psnr, ssim

from data_management import IPDataset
from find_adversarial import err_measure_l2, grid_attack
from operators import rotate_real, to_complex


# ----- load configuration -----
import config  # isort:skip
import config_robustness_fourier as cfg_rob  # isort:skip
from config_robustness_fourier import methods  # isort:skip

# ------ general setup ----------

device = cfg_rob.device

save_path = os.path.join(config.RESULTS_PATH, "attacks")
save_results = os.path.join(save_path, "table_adv.pkl")

do_plot = True
save_plot = True
save_table = True

# ----- attack setup -----

# select samples
samples = tuple(range(50, 100))

it_init = 6
keep_init = 3

# select range relative noise
noise_rel = torch.tensor([0.00, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08])

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

# select samples
test_data = IPDataset("test", config.DATA_PATH)
X_0 = torch.stack([test_data[s][0] for s in samples])
X_0 = to_complex(X_0.to(device))
Y_0 = cfg_rob.OpA(X_0)

# create result table
results = pd.DataFrame(
    columns=[
        "name",
        "X_adv_err",
        "X_ref_err",
        "X_adv_psnr",
        "X_ref_psnr",
        "X_adv_ssim",
        "X_ref_ssim",
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

        s_len = X_0.shape[0]
        results.loc[idx].X_adv_err = torch.zeros(len(noise_rel), s_len)
        results.loc[idx].X_ref_err = torch.zeros(len(noise_rel), s_len)
        results.loc[idx].X_adv_psnr = torch.zeros(len(noise_rel), s_len)
        results.loc[idx].X_ref_psnr = torch.zeros(len(noise_rel), s_len)
        results.loc[idx].X_adv_ssim = torch.zeros(len(noise_rel), s_len)
        results.loc[idx].X_ref_ssim = torch.zeros(len(noise_rel), s_len)

        for s in range(s_len):
            print("Sample: {}/{}".format(s + 1, s_len))
            X_0_s = X_0[s : s + 1, ...].repeat(
                it_init, *((X_0.ndim - 1) * (1,))
            )
            Y_0_s = Y_0[s : s + 1, ...].repeat(
                it_init, *((Y_0.ndim - 1) * (1,))
            )

            (
                X_adv_err_cur,
                X_ref_err_cur,
                X_adv_cur,
                X_ref_cur,
                _,
                _,
            ) = grid_attack(
                method,
                noise_rel,
                X_0_s,
                Y_0_s,
                store_data=True,
                keep_init=keep_init,
                err_measure=err_measure,
            )

            (
                results.loc[idx].X_adv_err[:, s],
                idx_max_adv_err,
            ) = X_adv_err_cur.max(dim=1)
            results.loc[idx].X_ref_err[:, s] = X_ref_err_cur.mean(dim=1)

            for idx_noise in range(len(noise_rel)):
                idx_max = idx_max_adv_err[idx_noise]
                results.loc[idx].X_adv_psnr[idx_noise, s] = psnr(
                    rotate_real(X_adv_cur[idx_noise, ...])[idx_max, 0:1, ...],
                    rotate_real(X_0_s.cpu())[0, 0:1, ...],
                    data_range=1.0,
                    reduction="none",
                )
                results.loc[idx].X_ref_psnr[idx_noise, s] = psnr(
                    rotate_real(X_ref_cur[idx_noise, ...])[:, 0:1, ...],
                    rotate_real(X_0_s.cpu())[:, 0:1, ...],
                    data_range=1.0,
                    reduction="mean",
                )
                results.loc[idx].X_adv_ssim[idx_noise, s] = ssim(
                    rotate_real(X_adv_cur[idx_noise, ...])[idx_max, 0:1, ...],
                    rotate_real(X_0_s.cpu())[0, 0:1, ...],
                    data_range=1.0,
                    size_average=False,
                )
                results.loc[idx].X_ref_ssim[idx_noise, s] = ssim(
                    rotate_real(X_ref_cur[idx_noise, ...])[:, 0:1, ...],
                    rotate_real(X_0_s.cpu())[:, 0:1, ...],
                    data_range=1.0,
                    size_average=True,
                )

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

        err_mean = results.loc[idx].X_adv_err[:, :].mean(dim=-1)
        err_std = results.loc[idx].X_adv_err[:, :].std(dim=-1)

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

    plt.yticks(np.arange(0, 1, step=0.05))
    plt.ylim((-0.01, 0.226))
    ax.set_xticklabels(["{:,.0%}".format(x) for x in ax.get_xticks()])
    ax.set_yticklabels(["{:,.0%}".format(x) for x in ax.get_yticks()])
    plt.legend(loc="upper left", fontsize=12)

    if save_plot:
        fig.savefig(
            os.path.join(save_path, "fig_table_adv.pdf"), bbox_inches="tight"
        )

    plt.show()

if save_table:
    df = results.applymap(
        lambda res: {"mean": res.mean(dim=-1), "std": res.std(dim=-1)}
    )

    # split adv and ref results
    df_adv = df[["X_adv_err", "X_adv_psnr", "X_adv_ssim"]]

    # extract mean and std
    df_adv_mean = (
        df_adv.stack()
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
    df_adv_std = (
        df_adv.stack()
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
    best_adv_l2 = df_adv_mean.xs("X_adv_err", level=1).idxmin()
    best_adv_ssim = df_adv_mean.xs("X_adv_ssim", level=1).idxmax()
    best_adv_psnr = df_adv_mean.xs("X_adv_psnr", level=1).idxmax()

    # combine mean and std data into "mean\pmstd" strings
    for (idx, method) in methods.iterrows():
        df_adv_mean.loc[idx, "X_adv_err"] = df_adv_mean.loc[
            idx, "X_adv_err"
        ].apply(lambda res: res * 100)
        df_adv_std.loc[idx, "X_adv_err"] = df_adv_std.loc[
            idx, "X_adv_err"
        ].apply(lambda res: res * 100)
    df_adv_combined = df_adv_mean.combine(
        df_adv_std,
        lambda col1, col2: col1.combine(
            col2, lambda el1, el2: "{:.2f} \\pm {:.2f}".format(el1, el2)
        ),
    )

    # format best value per noise level and metric as bold
    for col, idx in best_adv_l2.iteritems():
        df_adv_combined.at[(idx, "X_adv_err"), col] = (
            "\\bfseries " + df_adv_combined.at[(idx, "X_adv_err"), col]
        )
    for col, idx in best_adv_ssim.iteritems():
        df_adv_combined.at[(idx, "X_adv_ssim"), col] = (
            "\\bfseries " + df_adv_combined.at[(idx, "X_adv_ssim"), col]
        )
    for col, idx in best_adv_psnr.iteritems():
        df_adv_combined.at[(idx, "X_adv_psnr"), col] = (
            "\\bfseries " + df_adv_combined.at[(idx, "X_adv_psnr"), col]
        )

    # rename rows and columns
    df_adv_combined = df_adv_combined.rename(
        index={
            "X_adv_err": "rel.~$\\l{2}$-err. [\\%]",
            "X_adv_ssim": "SSIM",
            "X_adv_psnr": "PSNR",
        }
    )
    df_adv_combined = df_adv_combined.rename(
        index=methods["info"].apply(lambda res: res["name_disp"]).to_dict()
    )

    # save latex tabular
    df_adv_combined.to_latex(
        os.path.join(save_path, "table_adv.tex"),
        column_format=2 * "l" + len(noise_rel) * "S[separate-uncertainty]",
        multirow=True,
        escape=False,
    )
