import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from matplotlib import rc

from data_management import load_dataset
from find_adversarial import err_measure_l2, grid_attack_classification


# ---- load configuration -----
import config  # isort:skip
import config_robustness as cfg_rob  # isort:skip
from config_robustness import methods  # isort:skip

# ------ general setup ----------

device = cfg_rob.device

save_path = os.path.join(config.RESULTS_PATH, "attacks")
save_results = os.path.join(save_path, "table_classification.pkl")

do_plot = True
save_plot = True

# ----- data prep -----
X_test, C_test, Y_test = [
    tmp.unsqueeze(-2).to(device)
    for tmp in load_dataset(config.set_params["path"], subset="test")
]
C_test = C_test[..., 0].squeeze().long()

# ----- attack setup -----

# select samples
samples = tuple(range(50))

it_init = 100
keep_init = 50

# select range relative noise
noise_rel = torch.tensor([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])

# select measure for reconstruction error
err_measure = err_measure_l2

# select reconstruction methods
methods_include = ["L1", "UNet jit", "Tiramisu EE jit", "UNet It jit"]
methods = methods.loc[methods_include]

# select methods excluded from (re-)performing attacks
methods_no_calc = ["L1", "UNet jit", "Tiramisu EE jit", "UNet It jit"]

# ----- perform attack -----

# select samples
X_0 = X_test[samples, ...]
C_0 = C_test[samples, ...]
Y_0 = Y_test[samples, ...]

# create result table
results = pd.DataFrame(
    columns=["name", "X_adv_err", "X_ref_err", "C_adv_acc", "C_ref_acc"]
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
        results.loc[idx].C_adv_acc = torch.zeros(len(noise_rel), s_len)
        results.loc[idx].C_ref_acc = torch.zeros(len(noise_rel), s_len)

        for s in range(s_len):
            print("Sample: {}/{}".format(s + 1, s_len))
            X_0_s = X_0[s : s + 1, ...].repeat(
                it_init, *((X_0.ndim - 1) * (1,))
            )
            C_0_s = C_0[s : s + 1, ...].repeat(
                it_init, *((C_0.ndim - 1) * (1,))
            )
            Y_0_s = Y_0[s : s + 1, ...].repeat(
                it_init, *((Y_0.ndim - 1) * (1,))
            )
            (
                X_adv_err_cur,
                X_ref_err_cur,
                C_adv_acc_cur,
                C_ref_acc_cur,
            ) = grid_attack_classification(
                method,
                noise_rel,
                X_0_s,
                C_0_s,
                Y_0_s,
                cfg_rob.convnet,
                store_data=False,
                keep_init=keep_init,
                err_measure=err_measure,
            )

            # among all successful attacks choose the best one
            X_adv_err_tmp = results.loc[idx].X_adv_err.clone()
            X_adv_err_tmp[results.loc[idx].C_adv_acc.bool()] = np.inf
            idx_adv = X_adv_err_tmp.argmin(dim=1)

            X_ref_err_tmp = results.loc[idx].X_ref_err.clone()
            X_ref_err_tmp[results.loc[idx].C_ref_acc.bool()] = np.inf
            idx_ref = 0  # take first sample

            idx_noise = range(len(noise_rel))
            results.loc[idx].C_adv_acc[:, s] = C_adv_acc_cur[
                idx_noise, idx_adv
            ]
            results.loc[idx].C_ref_acc[:, s] = C_ref_acc_cur[
                idx_noise, idx_ref
            ]
            results.loc[idx].X_adv_err[:, s] = X_adv_err_cur[
                idx_noise, idx_adv
            ]
            results.loc[idx].X_ref_err[:, s] = X_ref_err_cur[
                idx_noise, idx_ref
            ]

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
    for (idx, method) in methods.iterrows():

        err_adv_mean = results.loc[idx].X_adv_err[:, :].mean(dim=-1)
        err_adv_std = results.loc[idx].X_adv_err[:, :].std(dim=-1)
        err_ref_mean = results.loc[idx].X_ref_err[:, :].mean(dim=-1)
        err_ref_std = results.loc[idx].X_ref_err[:, :].std(dim=-1)

        plt.plot(
            noise_rel,
            err_adv_mean,
            linestyle=method.info["plt_linestyle"],
            linewidth=method.info["plt_linewidth"],
            marker=method.info["plt_marker"],
            color=method.info["plt_color"],
            label=method.info["name_disp"],
        )

    plt.yticks(np.arange(0, 1.01, step=0.1))
    plt.ylim((-0.01, 1.01))
    ax.set_xticklabels(["{:,.0%}".format(x) for x in ax.get_xticks()])
    ax.set_yticklabels(["{:,.0%}".format(x) for x in ax.get_yticks()])
    plt.legend(loc="lower right", fontsize=12)

    if save_plot:
        fig.savefig(
            os.path.join(save_path, "fig_table_classification_err.pdf"),
            bbox_inches="tight",
        )

    fig, ax = plt.subplots(clear=True, figsize=(5, 4), dpi=200)
    for (idx, method) in methods.iterrows():

        acc_adv_mean = results.loc[idx].C_adv_acc[:, :].mean(dim=-1)
        acc_adv_std = results.loc[idx].C_adv_acc[:, :].std(dim=-1)
        acc_ref_mean = results.loc[idx].C_ref_acc[:, :].mean(dim=-1)
        acc_ref_std = results.loc[idx].C_ref_acc[:, :].std(dim=-1)

        plt.plot(
            noise_rel,
            acc_adv_mean,
            linestyle=method.info["plt_linestyle"],
            linewidth=method.info["plt_linewidth"],
            marker=method.info["plt_marker"],
            color=method.info["plt_color"],
            label=method.info["name_disp"],
        )

    plt.yticks(np.arange(0, 1.01, step=0.25))
    plt.ylim((-0.01, 1.01))
    ax.set_xticklabels(["{:,.0%}".format(x) for x in ax.get_xticks()])
    ax.set_yticklabels(["{:,.0%}".format(x) for x in ax.get_yticks()])
    plt.legend(loc="upper right", fontsize=12)

    if save_plot:
        fig.savefig(
            os.path.join(save_path, "fig_table_classification_acc.pdf"),
            bbox_inches="tight",
        )

    plt.show()
