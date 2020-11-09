import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from matplotlib import rc

from data_management import load_dataset
from find_adversarial import err_measure_l2, grid_attack


# ---- load configuration -----
import config  # isort:skip
import config_robustness as cfg_rob  # isort:skip
from config_robustness import methods  # isort:skip

# ------ general setup ----------

device = cfg_rob.device

save_path = os.path.join(config.RESULTS_PATH, "attacks")
save_results = os.path.join(save_path, "example_S18_adv.pkl")

do_plot = True
save_plot = True

# ----- data prep -----
X_test, C_test, Y_test = [
    tmp.unsqueeze(-2).to(device)
    for tmp in load_dataset(config.set_params["path"], subset="test")
]

# ----- attack setup -----

# select samples
sample = 18
it_init = 100
keep_init = 50

# select range relative noise
noise_min = 1e-3
noise_max = 0.10
noise_steps = 50
noise_rel_grid = torch.tensor(
    np.logspace(np.log10(noise_min), np.log10(noise_max), num=noise_steps)
).float()
noise_rel_show = torch.tensor([0.00, 0.01, 0.05, 0.10]).float()
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

# select samples
X_0 = X_test[sample : sample + 1, ...].repeat(
    it_init, *((X_test.ndim - 1) * (1,))
)
Y_0 = Y_test[sample : sample + 1, ...].repeat(
    it_init, *((Y_test.ndim - 1) * (1,))
)

# create result table and load existing results from file
results = pd.DataFrame(
    columns=[
        "name",
        "X_adv_err",
        "X_ref_err",
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

# save results
for idx in results.index:
    results_save.loc[idx] = results.loc[idx]
os.makedirs(save_path, exist_ok=True)
results_save.to_pickle(save_results)

# select the worst example for each noise level and method
results_max = pd.DataFrame(
    columns=[
        "name",
        "X_adv_err",
        "X_ref_err",
        "X_adv",
        "X_ref",
        "Y_adv",
        "Y_ref",
    ]
)
results_max.name = methods.index
results_max = results_max.set_index("name")
for (idx, method) in methods.iterrows():
    results_max.loc[idx].X_adv_err, idx_adv = results.loc[idx].X_adv_err.max(
        dim=1
    )
    results_max.loc[idx].X_ref_err, idx_ref = (
        results.loc[idx].X_ref_err[:, 0],
        0,
    )

    idx_noise = range(len(noise_rel))
    results_max.loc[idx].X_adv = results.loc[idx].X_adv[
        idx_noise, idx_adv, ...
    ]
    results_max.loc[idx].X_ref = results.loc[idx].X_ref[
        idx_noise, idx_ref, ...
    ]
    results_max.loc[idx].Y_adv = results.loc[idx].Y_adv[
        idx_noise, idx_adv, ...
    ]
    results_max.loc[idx].Y_ref = results.loc[idx].Y_ref[
        idx_noise, idx_ref, ...
    ]


# ----- plotting -----


def _implot(sub, im, vmin=0.0, vmax=1.0):
    im = im.reshape(-1, 1, 28, 28)  # vec to im
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
    plt.show()

    # method-wise plots
    for (idx, method) in methods.iterrows():

        # +++ reconstructions per noise level +++
        for idx_noise in range(len(noise_rel_show)):

            idx_noise_cur = torch.where(
                noise_rel == noise_rel_show[idx_noise]
            )[0]
            X_cur = results_max.loc[idx].X_adv[idx_noise_cur, ...]

            fig, ax = plt.subplots(
                1,
                2,
                clear=True,
                figsize=(5, 2.5),
                dpi=200,
                gridspec_kw={"wspace": 0.02},
            )

            im = _implot(ax[0], X_cur)

            im = _implot(ax[1], (X_cur - X_0[0:1, ...]).abs(), vmax=0.6)

            ax[1].text(
                0,
                28,
                "{:.2f}\\%".format(
                    results_max.loc[idx].X_adv_err[idx_noise_cur].item() * 100
                ),
                fontsize=30,
                color="#a1c9f4",
                horizontalalignment="left",
                verticalalignment="bottom",
            )

            if save_plot:
                fig.savefig(
                    os.path.join(
                        save_path,
                        "fig_example_S{}_adv_".format(sample)
                        + method.info["name_save"]
                        + "_{:.0e}".format(noise_rel_show[idx_noise].item())
                        + ".pdf",
                    ),
                    bbox_inches="tight",
                    pad_inches=0,
                )

            # not saved
            fig.suptitle(
                method.info["name_disp"]
                + " for rel. noise level = {:1.3f}".format(
                    noise_rel_show[idx_noise].item()
                )
            )

    # +++ error curves for all methods +++
    fig, ax = plt.subplots(clear=True, figsize=(5, 4), dpi=200)

    for (idx, method) in methods.iterrows():

        plt.plot(
            noise_rel[noise_rel <= noise_max],
            results_max.loc[idx].X_adv_err[noise_rel <= noise_max],
            linestyle=method.info["plt_linestyle"],
            linewidth=method.info["plt_linewidth"],
            marker=None,
            color=method.info["plt_color"],
            label=method.info["name_disp"],
        )

    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim((-0.008, 0.53))
    ax.set_xticklabels(["{:,.0%}".format(x) for x in ax.get_xticks()])
    ax.set_yticklabels(["{:,.0%}".format(x) for x in ax.get_yticks()])
    plt.legend(loc="upper left", fontsize=12)

    if save_plot:
        fig.savefig(
            os.path.join(
                save_path, "fig_example_S{}_adv_curve.pdf".format(sample)
            ),
            bbox_inches="tight",
        )

    plt.show()
