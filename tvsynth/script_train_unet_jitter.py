import os

import matplotlib as mpl
import torch

from data_management import Jitter, load_dataset
from networks import IterativeNet, UNet
from operators import TVAnalysis, get_tikhonov_matrix


# --- load configuration -----
import config  # isort:skip

# ----- general setup -----
mpl.use("agg")
device = torch.device("cuda:0")


# ----- operators -----
OpA = config.meas_op(config.m, config.n, device=device, **config.meas_params)
OpTV = TVAnalysis(config.n, device=device)


# ----- build linear inverter  ------
reg_fac = 2e-2

inverter = torch.nn.Linear(OpA.m, OpA.n, bias=False)
inverter.weight.requires_grad = False
inverter.weight.data = get_tikhonov_matrix(OpA, OpTV, reg_fac)

# ----- network configuration -----
subnet_params = {
    "in_channels": 1,
    "out_channels": 1,
    "drop_factor": 0.0,
    "base_features": 64,
}
subnet = UNet

it_net_params = {
    "operator": OpA,
    "inverter": inverter,
    "num_iter": 1,
    "lam": 0.0,
    "lam_learnable": False,
    "final_dc": False,
}

# ----- training setup ------
mse_loss = torch.nn.MSELoss(reduction="sum")


def loss_func(pred, tar):
    return mse_loss(pred, tar) / pred.shape[0]


train_phases = 2
train_params = {
    "num_epochs": [200, 75],
    "batch_size": [40, 40],
    "loss_func": loss_func,
    "save_path": [
        os.path.join(
            config.RESULTS_PATH,
            "unet_jitter_"
            "train_phase_{}".format((i + 1) % (train_phases + 1)),
        )
        for i in range(train_phases + 1)
    ],
    "save_epochs": 1,
    "optimizer": torch.optim.Adam,
    "optimizer_params": [
        {"lr": 8e-5, "eps": 1e-5, "weight_decay": 5e-3},
        {"lr": 5e-5, "eps": 1e-5, "weight_decay": 5e-3},
    ],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 1, "gamma": 1.0},
    "acc_steps": [1, 200],
    "train_transform": Jitter(2e0, 0.0, 1.0),
}


# -----data prep -----
X_train, C_train, Y_train = [
    tmp.unsqueeze(-2).to(device)
    for tmp in load_dataset(config.set_params["path"], subset="train")
]

X_val, C_val, Y_val = [
    tmp.unsqueeze(-2).to(device)
    for tmp in load_dataset(config.set_params["path"], subset="val")
]


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
    file.write("train_phases" + ": " + str(train_phases) + "\n")

# ------ construct network and train -----
subnet = subnet(**subnet_params).to(device)
it_net = IterativeNet(subnet, **it_net_params).to(device)
for i in range(train_phases):
    train_params_cur = {}
    for key, value in train_params.items():
        train_params_cur[key] = (
            value[i] if isinstance(value, (tuple, list)) else value
        )

    print("Phase {}:".format(i + 1))
    for key, value in train_params_cur.items():
        print(key + ": " + str(value))

    it_net.train_on((Y_train, X_train), (Y_val, X_val), **train_params_cur)
