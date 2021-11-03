import os

import matplotlib as mpl
import torch
import torchvision

from data_management import IPDataset, Jitter, SimulateMeasurements, ToComplex
from networks import IterativeNet, UNet
from operators import Fourier as Fourier
from operators import Fourier_matrix as Fourier_m
from operators import (
    LearnableInverter,
    RadialMaskFunc,
    rotate_real,
    unprep_fft_channel,
)


# ----- load configuration -----
import config  # isort:skip

# ----- global configuration -----
mpl.use("agg")
device = torch.device("cuda:0")
torch.cuda.set_device(0)

# ----- measurement configuration -----
mask_func = RadialMaskFunc(config.n, 40)
mask = unprep_fft_channel(mask_func((1, 1) + config.n + (1,)))
OpA_m = Fourier_m(mask)
OpA = Fourier(mask)
inverter = LearnableInverter(config.n, mask, learnable=False)


# ----- network configuration -----
subnet_params = {
    "in_channels": 2,
    "drop_factor": 0.0,
    "base_features": 32,
    "out_channels": 2,
}
subnet = UNet

it_net_params = {
    "num_iter": 8,
    "lam": 8 * [0.1],
    "lam_learnable": True,
    "final_dc": True,
    "resnet_factor": 1.0,
    "operator": OpA_m,
    "inverter": inverter,
}

# ----- training configuration -----
mseloss = torch.nn.MSELoss(reduction="sum")


def loss_func(pred, tar):
    return (
        mseloss(rotate_real(pred)[:, 0:1, ...], rotate_real(tar)[:, 0:1, ...],)
        / pred.shape[0]
    )


train_phases = 1
train_params = {
    "num_epochs": [15],
    "batch_size": [10],
    "loss_func": loss_func,
    "save_path": [
        os.path.join(
            config.RESULTS_PATH,
            "Fourier_UNet_it_jit-nojit_"
            "train_phase_{}".format((i + 1) % (train_phases + 1)),
        )
        for i in range(train_phases + 1)
    ],
    "save_epochs": 1,
    "optimizer": torch.optim.Adam,
    "optimizer_params": [{"lr": 4e-5, "eps": 2e-4, "weight_decay": 1e-3}],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 1, "gamma": 1.0},
    "acc_steps": [1],
    "train_transform": torchvision.transforms.Compose(
        [ToComplex(), SimulateMeasurements(OpA), Jitter(1e-1, 0.0, 1.0)]
    ),
    "val_transform": torchvision.transforms.Compose(
        [ToComplex(), SimulateMeasurements(OpA)],
    ),
    "train_loader_params": {"shuffle": True, "num_workers": 8},
    "val_loader_params": {"shuffle": False, "num_workers": 8},
}

# ----- data configuration -----

train_data_params = {
    "path": config.DATA_PATH,
}
train_data = IPDataset

val_data_params = {
    "path": config.DATA_PATH,
}
val_data = IPDataset

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
it_net.load_state_dict(
    torch.load(
        "results/Fourier_UNet_it_jit-nojit_pre_train_phase_1/model_weights.pt",
        map_location=torch.device(device),
    )
)

train_data = train_data("train", **train_data_params)
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
