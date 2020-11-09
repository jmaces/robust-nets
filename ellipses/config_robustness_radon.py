import os

import numpy as np
import pandas as pd
import torch

import config

from find_adversarial import PAdam, untargeted_attack
from networks import IterativeNet, Tiramisu, UNet
from operators import Radon, TVAnalysisPeriodic, noise_poisson, proj_l2_ball
from reconstruction_methods import admm_l1_rec


# ------ setup ----------
device = torch.device("cuda:0")
torch.cuda.set_device(0)

# ----- operators -----
theta = torch.linspace(0, 180, 61)[:-1]  # 60 lines, exclude endpoint
OpA = Radon(config.n, theta)
OpAIt = Radon(config.n, theta)
OpAIt.adj = OpAIt.inv
OpTV = TVAnalysisPeriodic(config.n, device=device)

# ----- methods --------
methods = pd.DataFrame(columns=["name", "info", "reconstr", "attacker", "net"])
methods = methods.set_index("name")

noise_ref = noise_poisson

# ----- set up L1 --------
# grid search parameters for L1 via admm
grid_search_file = os.path.join(
    config.RESULTS_PATH, "grid_search_l1", "grid_search_l1_radon_all.pkl"
)
gs_params = pd.read_pickle(grid_search_file)


def _get_gs_param(noise_rel):
    idx = (gs_params.noise_rel - noise_rel).abs().to_numpy().argmin()
    return gs_params.grid_param[idx]["lam"], gs_params.grid_param[idx]["rho"]


# the actual reconstruction method
def _reconstructL1(y, noise_rel):
    lam, rho = _get_gs_param(noise_rel.numpy())
    x, _ = admm_l1_rec(
        y,
        OpA,
        OpTV,
        0.0 * OpA.adj(y),
        0.0 * OpTV(OpA.adj(y)),
        lam,
        rho,
        iter=200,
        silent=False,
    )
    return x


# the reconstruction method used for the L1 attack
# (less iterations due to high computational costs)
def _reconstructL1_adv(y, lam, rho, x0, z0):
    x, _ = admm_l1_rec(y, OpA, OpTV, x0, z0, lam, rho, iter=20, silent=True)
    return x


# loss
mseloss = torch.nn.MSELoss(reduction="sum")


# attack function for L1
def _attackerL1(x0, noise_rel, yadv_init=None, batch_size=6):

    # compute noiseless measurements
    y0 = OpA(x0)

    if noise_rel == 0.0:
        return y0, y0, y0

    # compute absolute noise levels
    noise_level = noise_rel * y0.norm(p=2, dim=(-2, -1), keepdim=True)
    # compute noisy measurements for reference
    yref = noise_ref(OpA(x0), noise_level)

    # attack parameters
    adv_init_fac = 3.0 * noise_level
    adv_param = {
        "codomain_dist": mseloss,
        "domain_dist": None,
        "mixed_dist": None,
        "weights": (1.0, 1.0, 1.0),
        "optimizer": PAdam,
        "projs": None,
        "iter": 15,
        "stepsize": 5e0,
    }
    # get ADMM tuning parameters for noise_rel
    lam, rho = _get_gs_param(noise_rel.numpy())

    # compute good start values for _reconstructL1_adv
    x0_adv, z0_adv = admm_l1_rec(
        y0,
        OpA,
        OpTV,
        0.0 * OpA.adj(y0),
        0.0 * OpTV(OpA.adj(y0)),
        lam,
        rho,
        iter=200,
        silent=False,
    )
    # compute initialization
    yadv = y0.clone().detach() + (
        adv_init_fac / np.sqrt(np.prod(y0.shape[-2:]))
    ) * torch.randn_like(y0)

    if yadv_init is not None:
        yadv[0 : yadv_init.shape[0], ...] = yadv_init.clone().detach()

    for idx_batch in range(0, yadv.shape[0], batch_size):
        print(
            "Attack for samples "
            + str(list(range(idx_batch, idx_batch + batch_size)))
        )

        adv_param["projs"] = [
            lambda y: proj_l2_ball(
                y,
                y0[idx_batch : idx_batch + batch_size, ...],
                noise_level[idx_batch : idx_batch + batch_size, ...],
            )
        ]
        # perform attack
        yadv[idx_batch : idx_batch + batch_size, ...] = untargeted_attack(
            lambda y: _reconstructL1_adv(
                y,
                lam,
                rho,
                x0_adv[idx_batch : idx_batch + batch_size, ...],
                z0_adv[idx_batch : idx_batch + batch_size, ...],
            ),
            yadv[idx_batch : idx_batch + batch_size, ...]
            .clone()
            .requires_grad_(True),
            y0[idx_batch : idx_batch + batch_size, ...],
            t_out_ref=x0[idx_batch : idx_batch + batch_size, ...],
            **adv_param
        ).detach()

    return yadv, yref, y0


methods.loc["L1"] = {
    "info": {
        "name_disp": "TV$[\\eta]$",
        "name_save": "tv",
        "plt_color": "#e8000b",
        "plt_marker": "o",
        "plt_linestyle": "-",
        "plt_linewidth": 2.75,
    },
    "reconstr": _reconstructL1,
    "attacker": lambda x0, noise_rel, yadv_init=None: _attackerL1(
        x0, noise_rel, yadv_init=yadv_init
    ),
    "net": None,
}
methods.loc["L1", "net"] = None


# ----- set up net attacks --------

# the actual reconstruction method for any net
def _reconstructNet(y, noise_rel, net):
    return net.forward(y)


# attack function for any net
def _attackerNet(x0, noise_rel, net, yadv_init=None, batch_size=3):

    # compute noiseless measurements
    y0 = OpA(x0)

    if noise_rel == 0.0:
        return y0, y0, y0

    # compute absolute noise levels
    noise_level = noise_rel * y0.norm(p=2, dim=(-2, -1), keepdim=True)
    # compute noisy measurements for reference
    yref = noise_ref(OpA(x0), noise_level)  # noisy measurements

    # attack parameters
    adv_init_fac = 3.0 * noise_level
    adv_param = {
        "codomain_dist": mseloss,
        "domain_dist": None,
        "mixed_dist": None,
        "weights": (1.0, 1.0, 1.0),
        "optimizer": PAdam,
        "projs": None,
        "iter": 500,
        "stepsize": 5e0,
    }
    # compute initialization
    yadv = y0.clone().detach() + (
        adv_init_fac / np.sqrt(np.prod(y0.shape[-2:]))
    ) * torch.randn_like(y0)

    if yadv_init is not None:
        yadv[0 : yadv_init.shape[0], ...] = yadv_init.clone().detach()

    for idx_batch in range(0, yadv.shape[0], batch_size):
        print(
            "Attack for samples "
            + str(list(range(idx_batch, idx_batch + batch_size)))
        )

        adv_param["projs"] = [
            lambda y: proj_l2_ball(
                y,
                y0[idx_batch : idx_batch + batch_size, ...],
                noise_level[idx_batch : idx_batch + batch_size, ...],
            )
        ]
        # perform attack
        yadv[idx_batch : idx_batch + batch_size, ...] = untargeted_attack(
            lambda y: _reconstructNet(y, 0.0, net),
            yadv[idx_batch : idx_batch + batch_size, ...]
            .clone()
            .requires_grad_(True),
            y0[idx_batch : idx_batch + batch_size, ...],
            t_out_ref=x0[idx_batch : idx_batch + batch_size, ...],
            **adv_param
        ).detach()

    return yadv, yref, y0


# ----- load nets -----

# create a net and load weights from file
def _load_net(path, subnet, subnet_params, it_net_params):
    subnet = subnet(**subnet_params).to(device)
    it_net = IterativeNet(subnet, **it_net_params).to(device)
    it_net.load_state_dict(torch.load(path, map_location=torch.device(device)))
    it_net.freeze()
    it_net.eval()
    return it_net


def _append_net(name, info, net):
    methods.loc[name] = {
        "info": info,
        "reconstr": lambda y, noise_rel: _reconstructNet(y, noise_rel, net),
        "attacker": lambda x0, noise_rel, yadv_init=None: _attackerNet(
            x0, noise_rel, net, yadv_init=yadv_init
        ),
        "net": net,
    }
    pass


# ----- UNets -----

unet_params = {
    "in_channels": 1,
    "drop_factor": 0.0,
    "base_features": 36,
    "out_channels": 1,
}


_append_net(
    "UNet jit",
    {
        "name_disp": "UNet",
        "name_save": "unet_jit",
        "plt_color": "ff7c00",
        "plt_marker": "o",
        "plt_linestyle": ":",
        "plt_linewidth": 2.75,
    },
    _load_net(
        "results/Radon_UNet_jitter_v3_train_phase_2/model_weights.pt",
        UNet,
        unet_params,
        {
            "num_iter": 1,
            "lam": 0.0,
            "lam_learnable": False,
            "final_dc": False,
            "resnet_factor": 1.0,
            "operator": OpA,
            "inverter": OpA.inv,
        },
    ),
)

_append_net(
    "UNet",
    {
        "name_disp": "UNet no jit",
        "name_save": "unet",
        "plt_color": "darkorange",
        "plt_marker": "o",
        "plt_linestyle": ":",
        "plt_linewidth": None,
    },
    _load_net(
        "results/Radon_UNet_Hann_v2_train_phase_2/model_weights.pt",
        UNet,
        unet_params,
        {
            "num_iter": 1,
            "lam": 0.0,
            "lam_learnable": False,
            "final_dc": False,
            "resnet_factor": 1.0,
            "operator": OpA,
            "inverter": OpA.inv,
        },
    ),
)


# ----- Tiramisu -----
tiramisu_params = {
    "in_channels": 1,
    "out_channels": 1,
    "drop_factor": 0.0,
    "down_blocks": (5, 7, 9, 12, 15),
    "up_blocks": (15, 12, 9, 7, 5),
    "pool_factors": (2, 2, 2, 2, 2),
    "bottleneck_layers": 20,
    "growth_rate": 16,
    "out_chans_first_conv": 16,
}


_append_net(
    "Tiramisu jit",
    {
        "name_disp": "Tira",
        "name_save": "tiramisu_jit",
        "plt_color": "turquoise",
        "plt_marker": "o",
        "plt_linestyle": "-",
        "plt_linewidth": None,
    },
    _load_net(
        "results/Radon_Tiramisu_jitter_v6_train_phase_1/"
        + "model_weights_epoch19.pt",
        Tiramisu,
        tiramisu_params,
        {
            "num_iter": 1,
            "lam": 0.0,
            "lam_learnable": False,
            "final_dc": False,
            "resnet_factor": 1.0,
            "operator": OpA,
            "inverter": OpA.inv,
        },
    ),
)

_append_net(
    "Tiramisu",
    {
        "name_disp": "Tira no jit",
        "name_save": "tiramisu",
        "plt_color": "turquoise",
        "plt_marker": "o",
        "plt_linestyle": "-",
        "plt_linewidth": None,
    },
    _load_net(
        "results/Radon_Tiramisu_Hann_v5_train_phase_1/model_weights.pt",
        Tiramisu,
        tiramisu_params,
        {
            "num_iter": 1,
            "lam": 0.0,
            "lam_learnable": False,
            "final_dc": False,
            "resnet_factor": 1.0,
            "operator": OpA,
            "inverter": OpA.inv,
        },
    ),
)
