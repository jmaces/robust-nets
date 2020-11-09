import os

import matplotlib as mpl
import torch

from data_management import load_dataset
from networks import Simple2DConvNet


# ----- load configuration -----
import config  # isort:skip

# ----- setup -----
mpl.use("agg")
device = torch.device("cpu")

# ----- network configuration -----
net_params = {
    "drop_factor": 0.5,
}
net = Simple2DConvNet

# ----- training setup ------
ce_loss = torch.nn.CrossEntropyLoss(reduction="mean")
train_phases = 2
train_params = {
    "num_epochs": [20, 10],
    "batch_size": [40, 40],
    "loss_func": ce_loss,
    "save_path": [
        os.path.join(
            config.RESULTS_PATH,
            "convnet_" "train_phase_{}".format((i + 1) % (train_phases + 1)),
        )
        for i in range(train_phases + 1)
    ],
    "save_epochs": 1,
    "optimizer": torch.optim.Adam,
    "optimizer_params": [
        {"lr": 2e-4, "eps": 1e-5, "weight_decay": 1e-5},
        {"lr": 5e-5, "eps": 1e-5, "weight_decay": 1e-5},
    ],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 1, "gamma": 1.0},
    "acc_steps": [1, 1],
}

# -----data prep -----
X_train, C_train, Y_train = [
    tmp.unsqueeze(-2).to(device)
    for tmp in load_dataset(config.set_params["path"], subset="train")
]
X_train = X_train.reshape(-1, 1, 28, 28)
C_train = C_train[..., 0].squeeze().long()

X_val, C_val, Y_val = [
    tmp.unsqueeze(-2).to(device)
    for tmp in load_dataset(config.set_params["path"], subset="val")
]
X_val = X_val.reshape(-1, 1, 28, 28)
C_val = C_val[..., 0].squeeze().long()


# ------ save hyperparameters -------
os.makedirs(train_params["save_path"][-1], exist_ok=True)
with open(
    os.path.join(train_params["save_path"][-1], "hyperparameters.txt"), "w"
) as file:
    for key, value in net_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_params.items():
        file.write(key + ": " + str(value) + "\n")
    file.write("train_phases" + ": " + str(train_phases) + "\n")

# ------ construct network and train -----
net = net(**net_params).to(device)
for i in range(train_phases):
    train_params_cur = {}
    for key, value in train_params.items():
        train_params_cur[key] = (
            value[i] if isinstance(value, (tuple, list)) else value
        )

    print("Phase {}:".format(i + 1))
    for key, value in train_params_cur.items():
        print(key + ": " + str(value))

    net.train_on((X_train, C_train), (X_val, C_val), **train_params_cur)
