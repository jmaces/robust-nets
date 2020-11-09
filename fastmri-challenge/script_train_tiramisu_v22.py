import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
import torchvision

from fastmri_utils.common import subsample
from piq import SSIMLoss

from data_management import (
    CropOrPadAndResimulate,
    Flatten,
    Jitter,
    Normalize,
    RandomMaskDataset,
    filter_acquisition_no_fs,
)
from networks import IterativeNet, Tiramisu
from operators import rotate_real


# ----- load configuration -----
from config import RESULTS_PATH  # isort:skip

# ----- global configuration -----
mpl.use("agg")
device = torch.device("cuda:0")

train_phases = 1

# ----- network configuration -----
subnet_params = {
    "in_channels": 2,
    "drop_factor": 0.0,
    "down_blocks": (6, 8, 10, 12, 14),
    "up_blocks": (14, 12, 10, 8, 6),
    "pool_factors": (2, 2, 2, 2, 2),
    "bottleneck_layers": 20,
    "growth_rate": 12,
    "out_chans_first_conv": 16,
    "out_channels": 2,
}
subnet = Tiramisu

it_net_params = {
    "num_iter": 1,
    "lam": 1.0,
    "lam_learnable": False,
    "final_dc": False,
    "resnet_factor": 1.0,
    "concat_mask": False,
    "multi_slice": False,
}

# ----- training configuration -----
ssimloss = SSIMLoss(data_range=1e1)
maeloss = torch.nn.L1Loss(reduction="mean")


def ssim_l1_loss(pred, tar):
    return 0.7 * ssimloss(
        rotate_real(pred)[:, 0:1, ...], rotate_real(tar)[:, 0:1, ...],
    ) + 0.3 * maeloss(
        rotate_real(pred)[:, 0:1, ...], rotate_real(tar)[:, 0:1, ...],
    )


train_params = {
    "num_epochs": [40],
    "batch_size": [4],
    "loss_func": ssim_l1_loss,
    "save_path": [
        os.path.join(
            RESULTS_PATH,
            "challenge_4_no_fs_tiramisu_v22_"
            "train_phase_{}".format((i + 1) % (train_phases + 1)),
        )
        for i in range(train_phases + 1)
    ],
    "save_epochs": 1,
    "optimizer": torch.optim.Adam,
    "optimizer_params": [{"lr": 1e-4, "eps": 1e-4, "weight_decay": 1e-5}],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 1, "gamma": 1.0},
    "acc_steps": [1],
    "train_transform": torchvision.transforms.Compose(
        [
            CropOrPadAndResimulate((368, 368)),
            Flatten(0, -3),
            Normalize(reduction="mean", use_target=False),
            Jitter(1e1, 0.0, 1.0),
        ]
    ),
    "val_transform": torchvision.transforms.Compose(
        [
            CropOrPadAndResimulate((368, 368)),
            Flatten(0, -3),
            Normalize(reduction="mean", use_target=False),
        ],
    ),
    "train_loader_params": {"shuffle": True, "num_workers": 8},
    "val_loader_params": {"shuffle": False, "num_workers": 8},
}

# ----- data configuration -----
mask_func = subsample.RandomMaskFunc(
    center_fractions=[0.08], accelerations=[4]
)


train_data_params = {
    "mask_func": mask_func,
    "filter": [filter_acquisition_no_fs],
    "num_sym_slices": 0,
    "multi_slice_gt": False,
    "simulate_gt": True,
    "keep_mask_as_func": True,
}
train_data = RandomMaskDataset

val_data_params = {
    "mask_func": mask_func,
    "filter": [filter_acquisition_no_fs],
    "num_sym_slices": 0,
    "multi_slice_gt": False,
    "simulate_gt": True,
    "keep_mask_as_func": True,
}
val_data = RandomMaskDataset


# ------ save hyperparameters -------
os.makedirs(train_params["save_path"][-1], exist_ok=True)
with open(
    os.path.join(train_params["save_path"][-1], "hyperparameters.txt"), "w"
) as file:
    for key, value in subnet_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in it_net_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in val_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    file.write("train_phases" + ": " + str(train_phases) + "\n")

# ------ construct network and train -----
subnet = subnet(**subnet_params).to(device)
it_net = IterativeNet(subnet, **it_net_params).to(device)

train_data = train_data("train_larger", **train_data_params)
val_data = val_data("val", **val_data_params)

for i in range(train_phases):
    train_params_cur = {}
    for key, value in train_params.items():
        train_params_cur[key] = (
            value[i] if isinstance(value, (tuple, list)) else value
        )

    print("Phase {}:".format(i + 1))
    for key, value in train_params_cur.items():
        print(key + ": " + str(value))

    it_net.train_on(train_data, val_data, **train_params_cur)


# ----- pick best weights and save them ----
ssim = []
psnr = []
nmse = []
for i in range(1, train_params["num_epochs"][-1] + 1):
    t = pandas.read_pickle(
        os.path.join(
            train_params["save_path"][-2], "val_metrics_epoch{}.pkl".format(i)
        )
    )
    ssim.append(t["ssim"].mean())
    psnr.append(t["psnr"].mean())
    nmse.append(t["nmse"].mean())

ssim = np.array(ssim)
ssim_max = np.argmax(ssim) + 1

psnr = np.array(psnr)
psnr_max = np.argmax(psnr) + 1

nmse = np.array(nmse)
nmse_min = np.argmin(nmse) + 1

ssim_w = torch.load(
    os.path.join(
        train_params["save_path"][-2],
        "model_weights_epoch{}.pt".format(ssim_max),
    )
)
torch.save(
    ssim_w,
    os.path.join(
        train_params["save_path"][-2],
        "model_weights_maxSSIM_{}.pt".format(ssim_max),
    ),
)

psnr_w = torch.load(
    os.path.join(
        train_params["save_path"][-2],
        "model_weights_epoch{}.pt".format(psnr_max),
    )
)
torch.save(
    psnr_w,
    os.path.join(
        train_params["save_path"][-2],
        "model_weights_maxPSNR_{}.pt".format(psnr_max),
    ),
)

nmse_w = torch.load(
    os.path.join(
        train_params["save_path"][-2],
        "model_weights_epoch{}.pt".format(nmse_min),
    )
)
torch.save(
    nmse_w,
    os.path.join(
        train_params["save_path"][-2],
        "model_weights_minNMSE_{}.pt".format(nmse_min),
    ),
)

# plot overview
fig = plt.figure()
plt.plot(ssim, label="ssim")
plt.legend()
fig.savefig(
    os.path.join(train_params["save_path"][-2], "plot_ssim.png"),
    bbox_inches="tight",
)

fig = plt.figure()
plt.plot(psnr, label="psnr")
plt.legend()
fig.savefig(
    os.path.join(train_params["save_path"][-2], "plot_psnr.png"),
    bbox_inches="tight",
)

fig = plt.figure()
plt.plot(nmse, label="nmse")
plt.legend()
fig.savefig(
    os.path.join(train_params["save_path"][-2], "plot_nmse.png"),
    bbox_inches="tight",
)
