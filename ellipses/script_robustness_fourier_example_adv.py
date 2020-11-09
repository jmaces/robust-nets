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
torch.manual_seed(1)

save_path = os.path.join(config.RESULTS_PATH, "attacks")
save_results = os.path.join(save_path, "example_S66_adv.pkl")

do_plot = True
save_plot = True
transfer_to_TV = True
transfer_to_Net = True

# ----- attack setup -----

# select samples
sample = 66
it_init = 6
keep_init = 3

# dynamic range for plotting & similarity indices
v_min = 0.0
v_max = 0.9

# select range relative noise
noise_min = 1e-3
noise_max = 0.08
noise_steps = 50
noise_rel_grid = torch.tensor(
    np.logspace(np.log10(noise_min), np.log10(noise_max), num=noise_steps)
).float()
noise_rel_show = torch.tensor([0.00, 0.01, 0.03, 0.08]).float()
noise_rel = (
    torch.cat([noise_rel_show, noise_rel_grid]).float().unique(sorted=True)
)
noise_rel = torch.cat([noise_rel, torch.tensor([0.081, 0.031])]).float()
print(noise_rel)

# select measure for reconstruction error
err_measure = err_measure_l2

# select reconstruction methods
methods_include = ["L1", "UNet jit", "Tiramisu EE jit", "UNet It jit"]
methods = methods.loc[methods_include]

# select methods excluded from (re-)performing attacks
methods_no_calc = ["L1", "UNet jit", "Tiramisu EE jit", "UNet It jit"]

# ----- perform attack -----

# select sample
test_data = IPDataset("test", config.DATA_PATH)
X_0 = test_data[sample][0]
X_0 = to_complex(X_0.to(device)).unsqueeze(0)
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
    axins = ax.inset_axes([0.65, 0.65, 0.33, 0.33])
    _implot(axins, X_0)
    axins.set_xlim(47, 92)
    axins.set_ylim(152, 107)
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
                save_path, "fig_example_S{}_adv_gt.pdf".format(sample)
            ),
            bbox_inches="tight",
            pad_inches=0,
        )

    # method-wise plots
    for (idx, method) in methods.iterrows():

        # +++ reconstructions per noise level +++
        for idx_noise in range(len(noise_rel_show)):

            results_ref = results_max
            idx_noise_cur = torch.where(
                noise_rel == noise_rel_show[idx_noise]
            )[0]
            X_cur = results_ref.loc[idx].X_adv[idx_noise_cur, ...]

            fig, ax = plt.subplots(clear=True, figsize=(2.5, 2.5), dpi=200)
            im = _implot(ax, X_cur)

            ax.text(
                252,
                256,
                "rel.~$\\ell_2$-err: {:.2f}\\%".format(
                    results_ref.loc[idx].X_adv_err[idx_noise_cur].item() * 100
                ),
                fontsize=10,
                color="white",
                horizontalalignment="right",
                verticalalignment="bottom",
            )
            ax.text(
                252,
                213,
                "PSNR: {:.2f}".format(
                    results_ref.loc[idx].X_adv_psnr[idx_noise_cur].item()
                ),
                fontsize=10,
                color="white",
                horizontalalignment="right",
                verticalalignment="bottom",
            )
            ax.text(
                252,
                233,
                "SSIM: {:.2f}".format(
                    results_ref.loc[idx].X_adv_ssim[idx_noise_cur].item()
                ),
                fontsize=10,
                color="white",
                horizontalalignment="right",
                verticalalignment="bottom",
            )

            axins = ax.inset_axes([0.65, 0.65, 0.33, 0.33])
            _implot(axins, X_cur)

            axins.set_xlim(47, 92)
            axins.set_ylim(152, 107)
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
                        "fig_example_S{}_adv_".format(sample)
                        + method.info["name_save"]
                        + "_{:1.2e}".format(noise_rel_show[idx_noise].item())
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

            # error plot
            fig, ax = plt.subplots(clear=True, figsize=(2.5, 2.5), dpi=200)
            im = _implot(
                ax,
                (rotate_real(X_cur) - rotate_real(X_0)).abs(),
                vmin=0.0,
                vmax=0.6,
            )

            if save_plot:
                fig.savefig(
                    os.path.join(
                        save_path,
                        "fig_example_S{}_adv_err_".format(sample)
                        + method.info["name_save"]
                        + "_{:1.2e}".format(noise_rel_show[idx_noise].item())
                        + ".pdf",
                    ),
                    bbox_inches="tight",
                    pad_inches=0,
                )

            # +++ transferablility of examples +++
            if transfer_to_TV and (method.net is None or idx == "UNet It jit"):

                X_rec = method.reconstr(
                    results_ref.loc["UNet It jit"]
                    .Y_adv[len(noise_rel) - 3, ...]
                    .unsqueeze(0)
                    .to(device),
                    noise_rel_show[idx_noise].float(),
                )

                fig, ax = plt.subplots(clear=True, figsize=(2.5, 2.5), dpi=200)
                im = _implot(ax, X_rec)

                ax.text(
                    252,
                    256,
                    "rel.~$\\ell_2$-err: {:.2f}\\%".format(
                        err_measure(
                            rotate_real(X_rec)[:, 0:1, ...].cpu(),
                            rotate_real(X_0)[0:1, 0:1, ...].cpu(),
                        ).item()
                        * 100
                    ),
                    fontsize=10,
                    color="white",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                )
                ax.text(
                    252,
                    213,
                    "PSNR: {:.2f}".format(
                        psnr(
                            torch.clamp(
                                rotate_real(X_rec)[:, 0:1, ...].cpu(),
                                v_min,
                                v_max,
                            ),
                            torch.clamp(
                                rotate_real(X_0)[0:1, 0:1, ...].cpu(),
                                v_min,
                                v_max,
                            ),
                            data_range=v_max - v_min,
                        ).item()
                    ),
                    fontsize=10,
                    color="white",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                )
                ax.text(
                    252,
                    233,
                    "SSIM: {:.2f}".format(
                        ssim(
                            torch.clamp(
                                rotate_real(X_rec)[:, 0:1, ...].cpu(),
                                v_min,
                                v_max,
                            ),
                            torch.clamp(
                                rotate_real(X_0)[0:1, 0:1, ...].cpu(),
                                v_min,
                                v_max,
                            ),
                            data_range=v_max - v_min,
                        ).item()
                    ),
                    fontsize=10,
                    color="white",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                )

                axins = ax.inset_axes([0.65, 0.65, 0.33, 0.33])
                _implot(axins, X_rec)
                axins.set_xlim(47, 92)
                axins.set_ylim(152, 107)
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
                            "fig_example_S{}_adv_transfer_".format(sample)
                            + method.info["name_save"]
                            + "_{:1.2e}".format(
                                noise_rel_show[idx_noise].item()
                            )
                            + ".pdf",
                        ),
                        bbox_inches="tight",
                        pad_inches=0,
                    )

                # not saved
                fig.suptitle(
                    "Transfer: "
                    + method.info["name_disp"]
                    + " for rel. noise level = {:1.3f}".format(
                        noise_rel_show[idx_noise].item()
                    )
                )

                # other zoom
                fig, ax = plt.subplots(clear=True, figsize=(2.5, 2.5), dpi=200)
                im = _implot(ax, X_rec)

                ax.text(
                    252,
                    256,
                    "rel.~$\\ell_2$-err: {:.2f}\\%".format(
                        err_measure(
                            rotate_real(X_rec)[:, 0:1, ...].cpu(),
                            rotate_real(X_0)[0:1, 0:1, ...].cpu(),
                        ).item()
                        * 100
                    ),
                    fontsize=10,
                    color="white",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                )
                ax.text(
                    252,
                    213,
                    "PSNR: {:.2f}".format(
                        psnr(
                            torch.clamp(
                                rotate_real(X_rec)[:, 0:1, ...].cpu(),
                                v_min,
                                v_max,
                            ),
                            torch.clamp(
                                rotate_real(X_0)[0:1, 0:1, ...].cpu(),
                                v_min,
                                v_max,
                            ),
                            data_range=v_max - v_min,
                        ).item()
                    ),
                    fontsize=10,
                    color="white",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                )
                ax.text(
                    252,
                    233,
                    "SSIM: {:.2f}".format(
                        ssim(
                            torch.clamp(
                                rotate_real(X_rec)[:, 0:1, ...].cpu(),
                                v_min,
                                v_max,
                            ),
                            torch.clamp(
                                rotate_real(X_0)[0:1, 0:1, ...].cpu(),
                                v_min,
                                v_max,
                            ),
                            data_range=v_max - v_min,
                        ).item()
                    ),
                    fontsize=10,
                    color="white",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                )

                axins = ax.inset_axes([0.65, 0.65, 0.33, 0.33])
                _implot(axins, X_rec)
                axins.set_xlim(47, 83)
                axins.set_ylim(55, 19)
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
                            "fig_example_S{}_adv_transfer_III_".format(sample)
                            + method.info["name_save"]
                            + "_{:1.2e}".format(
                                noise_rel_show[idx_noise].item()
                            )
                            + ".pdf",
                        ),
                        bbox_inches="tight",
                        pad_inches=0,
                    )

                # not saved
                fig.suptitle(
                    "Transfer: "
                    + method.info["name_disp"]
                    + " for rel. noise level = {:1.3f}".format(
                        noise_rel_show[idx_noise].item()
                    )
                )

            # +++ transferablility of examples II +++
            if transfer_to_Net and method.net is not None:

                X_rec = method.reconstr(
                    results_ref.loc["L1"]
                    .Y_adv[len(noise_rel) - 3, ...]
                    .unsqueeze(0)
                    .to(device),
                    noise_rel_show[idx_noise].float(),
                )

                fig, ax = plt.subplots(clear=True, figsize=(2.5, 2.5), dpi=200)
                im = _implot(ax, X_rec)

                ax.text(
                    252,
                    256,
                    "rel.~$\\ell_2$-err: {:.2f}\\%".format(
                        err_measure(
                            rotate_real(X_rec)[:, 0:1, ...].cpu(),
                            rotate_real(X_0)[0:1, 0:1, ...].cpu(),
                        ).item()
                        * 100
                    ),
                    fontsize=10,
                    color="white",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                )
                ax.text(
                    252,
                    213,
                    "PSNR: {:.2f}".format(
                        psnr(
                            torch.clamp(
                                rotate_real(X_rec)[:, 0:1, ...].cpu(),
                                v_min,
                                v_max,
                            ),
                            torch.clamp(
                                rotate_real(X_0)[0:1, 0:1, ...].cpu(),
                                v_min,
                                v_max,
                            ),
                            data_range=v_max - v_min,
                        ).item()
                    ),
                    fontsize=10,
                    color="white",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                )
                ax.text(
                    252,
                    233,
                    "SSIM: {:.2f}".format(
                        ssim(
                            torch.clamp(
                                rotate_real(X_rec)[:, 0:1, ...].cpu(),
                                v_min,
                                v_max,
                            ),
                            torch.clamp(
                                rotate_real(X_0)[0:1, 0:1, ...].cpu(),
                                v_min,
                                v_max,
                            ),
                            data_range=v_max - v_min,
                        ).item()
                    ),
                    fontsize=10,
                    color="white",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                )

                axins = ax.inset_axes([0.65, 0.65, 0.33, 0.33])
                _implot(axins, X_rec)
                axins.set_xlim(47, 92)
                axins.set_ylim(152, 107)
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
                            "fig_example_S{}_adv_transfer_II_".format(sample)
                            + method.info["name_save"]
                            + "_{:1.2e}".format(
                                noise_rel_show[idx_noise].item()
                            )
                            + ".pdf",
                        ),
                        bbox_inches="tight",
                        pad_inches=0,
                    )

                # not saved
                fig.suptitle(
                    "Transfer: "
                    + method.info["name_disp"]
                    + " for rel. noise level = {:1.3f}".format(
                        noise_rel_show[idx_noise].item()
                    )
                )

    # +++ error curves for all methods +++

    fig, ax = plt.subplots(clear=True, figsize=(5, 4), dpi=200)

    for (idx, method) in methods.iterrows():

        plt.plot(
            noise_rel[noise_rel <= noise_max][:-1],
            results_max.loc[idx].X_adv_err[noise_rel <= noise_max][:-1],
            linestyle=method.info["plt_linestyle"],
            linewidth=method.info["plt_linewidth"],
            marker=None,
            color=method.info["plt_color"],
            label=method.info["name_disp"],
        )

    plt.yticks(np.arange(0, 1, step=0.05))
    plt.ylim((-0.006, 0.236))
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
