import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision

from matplotlib import font_manager, rc
from PIL import Image, ImageDraw, ImageFont
from piq import psnr, ssim

from data_management import (
    AlmostFixedMaskDataset,
    CropOrPadAndResimulate,
    Flatten,
    Normalize,
    filter_acquisition_no_fs,
)
from find_adversarial import err_measure_l2
from operators import rotate_real, to_complex


# ----- load configuration -----
import config  # isort:skip
import config_robustness as cfg_rob  # isort:skip
from config_robustness import methods  # isort:skip

# ------ general setup ----------
device = cfg_rob.device

save_path = os.path.join(config.RESULTS_PATH, "attacks")
save_results = os.path.join(save_path, "example_V8_S19_ood.pkl")

do_plot = True
save_plot = True
clean_image = False  # switch perturbation on or off

# ----- attack setup -----

# select samples
sample_vol = 8
sample_sl = 19
it = 50

# set range for plotting and similarity indices
v_min = 0.05
v_max = 4.50
print("Pixel values between {} and {}".format(v_min, v_max))


# structured perturbation
def text_perturbation(shape):
    # create empty image
    img = Image.new("P", shape)

    # adapt font choice depending on font availability on your system
    system_fonts = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
    filtered_fonts = [
        font for font in system_fonts if "DejaVuSansMono.ttf" in font
    ]
    # draw text on image
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(filtered_fonts[0], size=16)
    draw.text(
        (0.27 * shape[0], 0.350 * shape[1]),
        "I",
        255,
        font=font,
        stroke_width=0,
    )
    font = ImageFont.truetype(filtered_fonts[0], size=13)
    draw.text(
        (0.30 * shape[0], 0.360 * shape[1]),
        "E",
        255,
        font=font,
        stroke_width=0,
    )
    font = ImageFont.truetype(filtered_fonts[0], size=10)
    draw.text(
        (0.33 * shape[0], 0.3725 * shape[1]),
        "E",
        255,
        font=font,
        stroke_width=0,
    )
    font = ImageFont.truetype(filtered_fonts[0], size=7)
    draw.text(
        (0.355 * shape[0], 0.385 * shape[1]),
        "E",
        255,
        font=font,
        stroke_width=0,
    )

    t_img = (
        torch.tensor(np.array(img.getdata()).astype(np.float32).reshape(shape))
        / 255
    )
    return to_complex(t_img.unsqueeze(0)).unsqueeze(0).to(device)


def square_perturbation(shape):
    t_img = torch.zeros(shape)
    x, y = int(0.4675 * shape[0]), int(0.39 * shape[1])
    t_img[(x - 1) : (x + 2), (y - 1) : (y + 2)] = 1

    return to_complex(t_img.unsqueeze(0)).unsqueeze(0).to(device)


# select range of relative perturbations
pert1_scaling = -1.25 if not clean_image else 0.0
pert2_scaling = 2.75 if not clean_image else 0.0


def _perturbation(shape):
    p1 = pert1_scaling * text_perturbation(shape)
    p2 = pert2_scaling * square_perturbation(shape)
    return p1 + p2


# select measure for reconstruction error
err_measure = err_measure_l2

# select reconstruction methods
methods_include = ["Tiramisu EE jit", "Tiramisu EE"]
methods = methods.loc[methods_include]

# select methods excluded from (re-)performing attacks
methods_no_calc = ["Tiramisu EE jit", "Tiramisu EE"]

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
print(X_0.min(), X_0.max())
P_0 = _perturbation(X_0.shape[-2:])
X_0 = X_0 + P_0
print(X_0.min(), X_0.max())
Y_0 = cfg_rob.OpA(X_0)

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

# perform structured perturbations
for (idx, method) in methods.iterrows():
    if idx not in methods_no_calc:
        results.loc[idx].X_err = torch.zeros(1)
        results.loc[idx].X_psnr = torch.zeros(1)
        results.loc[idx].X_ssim = torch.zeros(1)
        results.loc[idx].X = torch.zeros(
            1, *X_0.shape[1:], device=torch.device("cpu")
        )
        results.loc[idx].Y = torch.zeros(
            1, *Y_0.shape[1:], device=torch.device("cpu")
        )

        print("Method: " + idx, flush=True)

        Y = Y_0
        X = method.reconstr(Y, torch.zeros(1))

        results.loc[idx].X_err = err_measure(X, X_0)
        results.loc[idx].X_psnr = psnr(
            torch.clamp(rotate_real(X.cpu())[:, 0:1, ...], v_min, v_max),
            torch.clamp(rotate_real(X_0.cpu())[:, 0:1, ...], v_min, v_max),
            data_range=v_max - v_min,
            reduction="none",
        )
        results.loc[idx].X_ssim = ssim(
            torch.clamp(rotate_real(X.cpu())[:, 0:1, ...], v_min, v_max),
            torch.clamp(rotate_real(X_0.cpu())[:, 0:1, ...], v_min, v_max),
            data_range=v_max - v_min,
            size_average=False,
        )
        results.loc[idx].X = X[0:1, ...].cpu()
        results.loc[idx].Y = Y[0:1, ...].cpu()

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


x_inset = (83, 132) if not clean_image else (75, 130)
y_inset = (160, 111) if not clean_image else (170, 125)
inset_pos = (0.65, 0.65, 0.33, 0.33)

if do_plot:

    # LaTeX typesetting
    rc("font", **{"family": "serif", "serif": ["Palatino"]})
    rc("text", usetex=True)

    X_0 = X_0.cpu()
    Y_0 = Y_0.cpu()

    # +++ ground truth +++
    fig, ax = plt.subplots(clear=True, figsize=(2.5, 2.5), dpi=200)
    im = _implot(ax, X_0)

    axins = ax.inset_axes(inset_pos)
    _implot(axins, X_0)
    axins.set_xlim(*x_inset)
    axins.set_ylim(*y_inset)
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
                "fig_example_S{}_ood_gt{}.pdf".format(
                    sample_vol, "" if not clean_image else "_clean"
                ),
            ),
            bbox_inches="tight",
            pad_inches=0,
        )

    # method-wise plots
    for (idx, method) in methods.iterrows():

        X_cur = results.loc[idx].X

        fig, ax = plt.subplots(clear=True, figsize=(2.5, 2.5), dpi=200)

        im = _implot(ax, X_cur)

        ax.text(
            8,
            10,
            "rel.~$\\ell_2$-err: {:.2f}\\%".format(
                results.loc[idx].X_err.item() * 100
            ),
            fontsize=10,
            color="white",
            horizontalalignment="left",
            verticalalignment="top",
        )
        ax.text(
            8,
            30,
            "PSNR: {:.2f}".format(results.loc[idx].X_psnr.item()),
            fontsize=10,
            color="white",
            horizontalalignment="left",
            verticalalignment="top",
        )
        ax.text(
            8,
            52,
            "SSIM: {:.2f}".format(results.loc[idx].X_ssim.item()),
            fontsize=10,
            color="white",
            horizontalalignment="left",
            verticalalignment="top",
        )

        axins = ax.inset_axes(inset_pos)
        _implot(axins, X_cur)
        axins.set_xlim(*x_inset)
        axins.set_ylim(*y_inset)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.spines["bottom"].set_color("#a1c9f4")
        axins.spines["top"].set_color("#a1c9f4")
        axins.spines["left"].set_color("#a1c9f4")
        axins.spines["right"].set_color("#a1c9f4")
        ax.indicate_inset_zoom(axins, edgecolor="#a1c9f4")

        if not clean_image and idx == "Tiramisu EE":
            axins2 = ax.inset_axes([0.02, 0.02, 0.33, 0.33])
            _implot(axins2, X_0)
            axins2.set_xlim(*x_inset)
            axins2.set_ylim(*y_inset)
            axins2.set_xticks([])
            axins2.set_yticks([])
            axins2.spines["bottom"].set_color("#a1c9f4")
            axins2.spines["top"].set_color("#a1c9f4")
            axins2.spines["left"].set_color("#a1c9f4")
            axins2.spines["right"].set_color("#a1c9f4")
            ax.text(
                8,
                205,
                "ground truth",
                fontsize=7,
                color="#a1c9f4",
                horizontalalignment="left",
                verticalalignment="bottom",
            )

        if save_plot:
            fig.savefig(
                os.path.join(
                    save_path,
                    "fig_example_S{}_ood_".format(sample_vol)
                    + method.info["name_save"]
                    + "{}.pdf".format("" if not clean_image else "_clean"),
                ),
                bbox_inches="tight",
                pad_inches=0,
            )

        # not saved
        plt.title(method.info["name_disp"])

    plt.show()
