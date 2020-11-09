import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from matplotlib import rc
from piq import psnr, ssim

from data_management import IPDataset
from find_adversarial import err_measure_l2, grid_attack


# ----- load configuration -----
import config  # isort:skip
import config_robustness_radon as cfg_rob  # isort:skip
from config_robustness_radon import methods  # isort:skip

# ------ general setup ----------

device = cfg_rob.device
torch.manual_seed(10)


save_path = os.path.join(config.RESULTS_PATH, "attacks")

do_plot = True
save_plot = True

# ----- attack setup -----

# select samples
sample = 48
it_init = 6
keep_init = 3

save_results = os.path.join(
    save_path, "example_S{}_radon_inverse_crime.pkl".format(sample)
)

# select range relative noise
noise_rel = torch.tensor([1e-2]).float()

# select measure for reconstruction error
err_measure = err_measure_l2

# select reconstruction methods
methods_include = ["UNet", "UNet jit"]
methods = methods.loc[methods_include]

# select methods excluded from (re-)performing attacks
methods_no_calc = ["UNet", "UNet jit"]

# ----- perform attack -----

# select sample
test_data = IPDataset("test", config.DATA_PATH)
X_0 = test_data[sample][0]
X_0 = X_0.to(device).unsqueeze(0)
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
                torch.clamp(results.loc[idx].X_adv[idx_noise, ...], 0.0, 1.0),
                X_0.cpu(),
                data_range=1.0,
                reduction="none",
            )
            results.loc[idx].X_ref_psnr[idx_noise, ...] = psnr(
                torch.clamp(results.loc[idx].X_ref[idx_noise, ...], 0.0, 1.0),
                X_0.cpu(),
                data_range=1.0,
                reduction="none",
            )
            results.loc[idx].X_adv_ssim[idx_noise, ...] = ssim(
                torch.clamp(results.loc[idx].X_adv[idx_noise, ...], 0.0, 1.0),
                X_0.cpu(),
                data_range=1.0,
                size_average=False,
            )
            results.loc[idx].X_ref_ssim[idx_noise, ...] = ssim(
                torch.clamp(results.loc[idx].X_ref[idx_noise, ...], 0.0, 1.0),
                X_0.cpu(),
                data_range=1.0,
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


def _implot(sub, im, vmin=0.0, vmax=1.0):
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
    axins = ax.inset_axes([0.02, 0.68, 0.37, 0.37])
    _implot(axins, X_0)
    axins.set_xlim(120, 162)
    axins.set_ylim(82, 55)
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
                save_path, "fig_example_S{}_radon_crime_gt.pdf".format(sample)
            ),
            bbox_inches="tight",
            pad_inches=0,
        )

    # method-wise plots
    for (idx, method) in methods.iterrows():

        # +++ reconstructions per noise level +++
        for idx_noise in range(len(noise_rel)):

            results_ref = results_max
            X_cur = results_ref.loc[idx].X_adv[idx_noise : idx_noise + 1, ...]

            fig, ax = plt.subplots(clear=True, figsize=(2.5, 2.5), dpi=200)

            im = _implot(ax, X_cur)

            ax.text(
                4,
                258,
                "rel.~$\\ell_2$-err: {:.2f}\\%".format(
                    results_ref.loc[idx].X_adv_err[idx_noise].item() * 100
                ),
                fontsize=10,
                color="white",
                horizontalalignment="left",
                verticalalignment="bottom",
            )
            ax.text(
                252,
                233,
                "PSNR: {:.2f}".format(
                    results_ref.loc[idx].X_adv_psnr[idx_noise].item()
                ),
                fontsize=10,
                color="white",
                horizontalalignment="right",
                verticalalignment="bottom",
            )
            ax.text(
                252,
                254,
                "SSIM: {:.2f}".format(
                    results_ref.loc[idx].X_adv_ssim[idx_noise].item()
                ),
                fontsize=10,
                color="white",
                horizontalalignment="right",
                verticalalignment="bottom",
            )

            axins = ax.inset_axes([0.02, 0.68, 0.37, 0.37])
            _implot(axins, X_cur)
            axins.set_xlim(120, 162)
            axins.set_ylim(82, 55)
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
                        "fig_example_S{}_radon_crime_".format(sample)
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

            if method.net is not None:
                inverter = method.net.inverter

                fig, ax = plt.subplots(clear=True, figsize=(2.5, 2.5), dpi=200)

                inv = inverter(
                    results_ref.loc[idx]
                    .Y_adv[idx_noise, ...]
                    .unsqueeze(0)
                    .to(device)
                ).cpu()
                _implot(ax, inv, vmin=inv.min(), vmax=inv.max())

                axins = ax.inset_axes([0.02, 0.68, 0.37, 0.37])
                _implot(axins, inv, vmin=inv.min(), vmax=inv.max())
                axins.set_xlim(120, 162)
                axins.set_ylim(82, 55)
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
                            "fig_example_S{}_radon_crime_inverter_".format(
                                sample
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
                    + " for rel. noise level = {:1.3f} (inverter)".format(
                        noise_rel[idx_noise].item()
                    )
                )


plt.show()
