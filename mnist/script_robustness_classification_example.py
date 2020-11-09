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
save_results = os.path.join(save_path, "example_classification_s2.pkl")

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
sample = 2
it_init = 100
keep_init = 50

# select range relative noise
noise_rel = torch.tensor([0.05, 0.10, 0.15, 0.20])

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
C_0 = C_test[sample : sample + 1, ...].repeat(
    it_init, *((C_test.ndim - 1) * (1,))
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
        "C_adv_acc",
        "C_ref_acc",
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
            results.loc[idx].C_adv_acc,
            results.loc[idx].C_ref_acc,
            results.loc[idx].X_adv,
            results.loc[idx].X_ref,
            results.loc[idx].Y_adv,
            results.loc[idx].Y_ref,
        ) = grid_attack_classification(
            method,
            noise_rel,
            X_0,
            C_0,
            Y_0,
            cfg_rob.convnet,
            store_data=True,
            keep_init=keep_init,
            err_measure=err_measure,
        )

# save results
for idx in results.index:
    results_save.loc[idx] = results.loc[idx]
os.makedirs(save_path, exist_ok=True)
results_save.to_pickle(save_results)

# among successful attacks select the best example per noise level and method
results_best = pd.DataFrame(
    columns=[
        "name",
        "X_adv_err",
        "X_ref_err",
        "C_adv_acc",
        "C_ref_acc",
        "X_adv",
        "X_ref",
        "Y_adv",
        "Y_ref",
    ]
)
results_best.name = methods.index
results_best = results_best.set_index("name")
for (idx, method) in methods.iterrows():
    X_adv_err_cur = results.loc[idx].X_adv_err.clone()
    X_adv_err_cur[results.loc[idx].C_adv_acc.bool()] = np.inf
    idx_adv = X_adv_err_cur.argmin(dim=1)

    X_ref_err_cur = results.loc[idx].X_ref_err.clone()
    X_ref_err_cur[results.loc[idx].C_ref_acc.bool()] = np.inf
    idx_ref = X_ref_err_cur.argmin(dim=1)

    idx_noise = range(len(noise_rel))
    results_best.loc[idx].X_adv_err = results.loc[idx].X_adv_err[
        idx_noise, idx_adv
    ]
    results_best.loc[idx].X_ref_err = results.loc[idx].X_ref_err[
        idx_noise, idx_ref
    ]
    results_best.loc[idx].C_adv_acc = results.loc[idx].C_adv_acc[
        idx_noise, idx_adv
    ]
    results_best.loc[idx].C_ref_acc = results.loc[idx].C_ref_acc[
        idx_noise, idx_ref
    ]
    results_best.loc[idx].Y_adv = results.loc[idx].Y_adv[idx_noise, idx_adv]
    results_best.loc[idx].X_adv = results.loc[idx].X_adv[idx_noise, idx_adv]
    results_best.loc[idx].X_ref = results.loc[idx].X_ref[idx_noise, idx_ref]
    results_best.loc[idx].Y_adv = results.loc[idx].Y_adv[idx_noise, idx_adv]
    results_best.loc[idx].Y_ref = results.loc[idx].Y_ref[idx_noise, idx_ref]


# ----- plotting -----


def _implot(sub, im, vmin=0.0, vmax=1.0):
    im = im.reshape(-1, 1, 28, 28)  # vec to im
    sub.imshow(
        im[0, 0, :, :].detach().cpu(), vmin=vmin, vmax=vmax, cmap="gray"
    )
    sub.get_xaxis().set_ticks([])
    sub.get_yaxis().set_ticks([])


if do_plot:

    # LaTeX typesetting
    rc("font", **{"family": "serif", "serif": ["Palatino"]})
    rc("text", usetex=True)

    X_0 = X_0.cpu()
    Y_0 = Y_0.cpu()
    C_0 = C_0.cpu()

    for (idx, method) in methods.iterrows():
        C_pred = cfg_rob.convnet(X_0.to(device).view(-1, 1, 28, 28)).cpu()
        C_adv_pred = cfg_rob.convnet(
            results_best.loc[idx].X_adv.to(device).view(-1, 1, 28, 28)
        ).cpu()
        C_ref_pred = cfg_rob.convnet(
            results_best.loc[idx].X_ref.to(device).view(-1, 1, 28, 28)
        ).cpu()

        for idx_noise in range(len(noise_rel)):

            # plot
            fig, sub = plt.subplots(
                1,
                1,
                clear=True,
                figsize=(5, 5),
                dpi=200,
                gridspec_kw={"wspace": 0.02},
            )
            _implot(
                sub,
                results_best.loc[idx].X_adv[idx_noise : idx_noise + 1, ...],
            )
            sub.text(
                27,
                28,
                "$\\texttt{{{}}}$".format(
                    C_adv_pred.argmax(dim=-1)[idx_noise]
                ),
                fontsize=80,
                color="#a1c9f4",
                horizontalalignment="right",
                verticalalignment="bottom",
            )

            if save_plot:
                fig.savefig(
                    os.path.join(
                        save_path,
                        "fig_example_S{}_classification_{}_{:1.2e}.pdf".format(
                            sample,
                            method.info["name_save"],
                            noise_rel[idx_noise],
                        ),
                    ),
                    bbox_inches="tight",
                    pad_inches=0,
                )

    plt.show()
