import numpy as np
import pandas as pd
import torch

import config

from find_adversarial import PAdam, carlini_wagner, untargeted_attack
from networks import IterativeNet, Simple2DConvNet, Tiramisu, UNet
from operators import (
    TVAnalysis,
    TVSynthesis,
    get_operator_norm,
    get_tikhonov_matrix,
    noise_gaussian,
    proj_l2_ball,
)
from reconstruction_methods import primaldual


# ----- setup -----
device = torch.device("cuda:0")
torch.cuda.set_device(0)

# ----- operators -----
OpA = config.meas_op(config.m, config.n, device=device, **config.meas_params)
OpTVSynth = TVSynthesis(config.n, device=device)
OpTV = TVAnalysis(config.n, device=device)

noise_ref = noise_gaussian

# ----- set up L1 --------

# recovery parameters for L1 via primal dual
OpAW_norm = get_operator_norm(OpA, OpTVSynth)
rec_params = {
    "sigma": 1.0 / (OpAW_norm + 1e0),
    "tau": 1.0 / (OpAW_norm + 1e0),
    "theta": 1e0,
}


# the actual reconstruction method
def _reconstructL1(y, noise_level):
    x, _, _ = primaldual(
        y,
        OpA,
        OpTVSynth,
        iter=50000,
        c0=torch.zeros(config.n,).to(device),
        y0=torch.zeros(config.m,).to(device),
        eta=noise_level,
        silent=True,
        **rec_params
    )
    return x


# the reconstruction method used for the L1 attack
# (less iterations due to high computational costs)
def _reconstructL1_adv(y, noise_level, c0, y0):
    x, _, _ = primaldual(
        y,
        OpA,
        OpTVSynth,
        iter=2000,
        c0=c0,
        y0=y0,
        eta=noise_level,
        silent=True,
        **rec_params
    )
    return x


# attack function for L1
def _attackerL1(x0, noise_rel, yadv_init=None):

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
        "codomain_dist": torch.nn.MSELoss(reduction="sum"),
        "domain_dist": None,
        "weights": (1.0, 1.0),
        "optimizer": PAdam,
        "projs": [lambda y: proj_l2_ball(y, y0, noise_level)],
        "iter": 30,
        "stepsize": 5e0,
    }
    # compute good start values for _reconstructL1_adv
    _, c0_adv, y0_adv = primaldual(
        y0,
        OpA,
        OpTVSynth,
        iter=50000,
        c0=torch.zeros(config.n,).to(device),
        y0=torch.zeros(config.m,).to(device),
        eta=noise_level,
        silent=True,
        **rec_params
    )
    # compute initialization
    yadv = y0.clone().detach() + (
        adv_init_fac / np.sqrt(np.prod(y0.shape[-2:]))
    ) * torch.randn_like(y0)

    if yadv_init is not None:
        yadv[0 : yadv_init.shape[0], ...] = yadv_init.clone().detach()

    yadv = yadv.requires_grad_(True)

    # perform attack
    yadv = untargeted_attack(
        lambda y: _reconstructL1_adv(y, noise_level, c0_adv, y0_adv),
        yadv,
        y0,
        t_out_ref=x0,
        **adv_param
    ).detach()

    return yadv, yref, y0


# ----- set up networks --------

# the actual reconstruction method for any net
def _reconstructNet(y, noise_level, net):
    return net.forward(y)


# attack function for any net
def _attackerNet(x0, noise_rel, net, yadv_init=None):

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
        "codomain_dist": torch.nn.MSELoss(reduction="sum"),
        "domain_dist": None,
        "weights": (1.0, 1.0),
        "optimizer": PAdam,
        "projs": [lambda y: proj_l2_ball(y, y0, noise_level)],
        "iter": 100,
        "stepsize": 5e0,
    }
    # compute initialization
    yadv = y0.clone().detach() + (
        adv_init_fac / np.sqrt(np.prod(y0.shape[-2:]))
    ) * torch.randn_like(y0)

    if yadv_init is not None:
        yadv[0 : yadv_init.shape[0], ...] = yadv_init.clone().detach()

    yadv = yadv.requires_grad_(True)

    # perform attack
    yadv = untargeted_attack(
        lambda y: _reconstructNet(y, 0.0, net),
        yadv,
        y0,
        t_out_ref=x0,
        **adv_param
    ).detach()

    return yadv, yref, y0


# ----- setup classification concatenation -----

convnet_params = {
    "drop_factor": 0.5,
}

convnet = Simple2DConvNet(**convnet_params).to(device)
convnet.load_state_dict(
    torch.load(
        "results/convnet_train_phase_2/model_weights.pt",
        map_location=torch.device(device),
    )
)
convnet.freeze()
convnet.eval()


# classification attacks for L1 reconstructions
def _attackerL1Classification(x0, c0, noise_rel, yadv_init=None):

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
        "codomain_dist": carlini_wagner(threshold=0.0),
        "domain_dist": None,
        "weights": (1.0, 1.0),
        "optimizer": PAdam,
        "projs": [lambda y: proj_l2_ball(y, y0, noise_level)],
        "iter": 100,
        "stepsize": 5e-1,
    }

    # compute good start values for _reconstructL1_adv
    _, c0_adv, y0_adv = primaldual(
        y0,
        OpA,
        OpTVSynth,
        iter=50000,
        c0=torch.zeros(config.n,).to(device),
        y0=torch.zeros(config.m,).to(device),
        eta=noise_level,
        silent=True,
        **rec_params
    )
    # compute initialization
    yadv = y0.clone().detach() + (
        adv_init_fac / np.sqrt(np.prod(y0.shape[-2:]))
    ) * torch.randn_like(y0)

    if yadv_init is not None:
        yadv[0 : yadv_init.shape[0], ...] = yadv_init.clone()

    yadv = yadv.requires_grad_(True)

    # perform attack
    yadv = untargeted_attack(
        lambda y: convnet(
            _reconstructL1_adv(y, noise_level, c0_adv, y0_adv).view(
                -1, 1, 28, 28
            )
        ),
        yadv,
        y0,
        t_out_ref=c0,
        **adv_param
    ).detach()

    return yadv, yref, y0


# classification attacks for network reconstructions
def _attackerNetClassification(x0, c0, noise_rel, net, yadv_init=None):

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
        "codomain_dist": carlini_wagner(threshold=0.0),
        "domain_dist": None,
        "weights": (1.0, 1.0),
        "optimizer": PAdam,
        "projs": [lambda y: proj_l2_ball(y, y0, noise_level)],
        "iter": 100,
        "stepsize": 5e-1,
    }

    # compute initialization
    yadv = y0.clone().detach() + (
        adv_init_fac / np.sqrt(np.prod(y0.shape[-2:]))
    ) * torch.randn_like(y0)

    if yadv_init is not None:
        yadv[0 : yadv_init.shape[0], ...] = yadv_init.clone()

    yadv = yadv.requires_grad_(True)

    # perform attack
    yadv = untargeted_attack(
        lambda y: convnet(_reconstructNet(y, 0.0, net).view(-1, 1, 28, 28)),
        yadv,
        y0,
        t_out_ref=c0,
        **adv_param
    ).detach()

    return yadv, yref, y0


# ----- network helper functions -----

# create a fresh tikhonov inverter layer
def _get_inverter_tikh(reg_fac=2e-2):
    inverter_tikh = torch.nn.Linear(OpA.m, OpA.n, bias=False)
    inverter_tikh.weight.requires_grad = False
    inverter_tikh.weight.data = get_tikhonov_matrix(OpA, OpTV, reg_fac)
    return inverter_tikh


# create a net and load weights from file
def _load_net(path, subnet, subnet_params, it_net_params):
    subnet = subnet(**subnet_params).to(device)
    it_net = IterativeNet(subnet, **it_net_params).to(device)
    it_net.load_state_dict(torch.load(path, map_location=torch.device(device)))
    it_net.freeze()
    it_net.eval()
    return it_net


# add network to method collection
def _append_net(name, info, net):
    methods.loc[name] = {
        "info": info,
        "reconstr": lambda y, noise_level: _reconstructNet(
            y, noise_level, net
        ),
        "attacker": lambda x0, noise_rel, yadv_init=None: _attackerNet(
            x0, noise_rel, net, yadv_init=yadv_init
        ),
        "attacker_classification": lambda x0, c0, noise_rel, yadv_init=None: (
            _attackerNetClassification(
                x0, c0, noise_rel, net, yadv_init=yadv_init
            )
        ),
        "net": net,
    }
    pass


# ----- ----- Build Method Collection ----- -----
methods = pd.DataFrame(
    columns=[
        "name",
        "info",
        "reconstr",
        "attacker",
        "attacker_classification",
        "net",
    ]
)
methods = methods.set_index("name")

# add L1
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
    "attacker_classification": lambda x0, c0, noise_rel, yadv_init=None: (
        _attackerL1Classification(x0, c0, noise_rel, yadv_init=yadv_init)
    ),
    "net": None,
}
methods.loc["L1", "net"] = None


# add UNet variants

unet_params = {
    "in_channels": 1,
    "out_channels": 1,
    "drop_factor": 0.0,
    "base_features": 64,
}


_append_net(
    "UNet jit",
    {
        "name_disp": "UNet",
        "name_save": "unet_jit",
        "plt_color": "#ff7c00",
        "plt_marker": "o",
        "plt_linestyle": ":",
        "plt_linewidth": 2.75,
    },
    _load_net(
        "results/unet_jitter_train_phase_2/model_weights.pt",
        UNet,
        unet_params,
        {
            "operator": OpA,
            "inverter": _get_inverter_tikh(reg_fac=2e-2),
            "num_iter": 1,
            "lam": 0.0,
            "lam_learnable": False,
            "final_dc": False,
        },
    ),
)


_append_net(
    "UNet EE jit",
    {
        "name_disp": "UNetFL",
        "name_save": "unet_ee_jit",
        "plt_color": "maroon",
        "plt_marker": "o",
        "plt_linestyle": "-",
        "plt_linewidth": None,
    },
    _load_net(
        "results/unet_ee_jitter_train_phase_2/model_weights.pt",
        UNet,
        unet_params,
        {
            "operator": OpA,
            "inverter": _get_inverter_tikh(),  # placeholder, learned
            "num_iter": 1,
            "lam": 0.0,
            "lam_learnable": False,
            "final_dc": False,
        },
    ),
)


_append_net(
    "UNet It jit",
    {
        "name_disp": "ItNet",
        "name_save": "unet_it_jit",
        "plt_color": "#023eff",
        "plt_marker": "o",
        "plt_linestyle": "--",
        "plt_linewidth": 2.75,
    },
    _load_net(
        "results/unet_it_tikh_jitter_train_phase_2/model_weights.pt",
        UNet,
        unet_params,
        {
            "operator": OpA,
            "inverter": _get_inverter_tikh(reg_fac=2e-2),
            "num_iter": 8,
            "lam": 8 * [0.1],
            "lam_learnable": False,
            "final_dc": True,
        },
    ),
)

_append_net(
    "UNet It",
    {
        "name_disp": "ItNet w\\slash o jittering",
        "name_save": "unet_it_nojit",
        "plt_color": "midnightblue",
        "plt_marker": "o",
        "plt_linestyle": (0, (3, 5, 1, 5, 1, 5)),
        "plt_linewidth": 2.75,
    },
    _load_net(
        "results/unet_it_tikh_train_phase_2/model_weights.pt",
        UNet,
        unet_params,
        {
            "operator": OpA,
            "inverter": _get_inverter_tikh(reg_fac=2e-2),
            "num_iter": 8,
            "lam": 8 * [0.1],
            "lam_learnable": False,
            "final_dc": True,
        },
    ),
)

# add Tiramisu variants

tiramisu_params = {
    "in_channels": 1,
    "out_channels": 1,
    "drop_factor": 0.0,
    "down_blocks": (5, 7, 9, 12, 15),
    "up_blocks": (15, 12, 9, 7, 5),
    "pool_factors": (2, 2, 2, 2, 2),
    "bottleneck_layers": 25,
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
        "results/tiramisu_jitter_train_phase_2/model_weights.pt",
        Tiramisu,
        tiramisu_params,
        {
            "operator": OpA,
            "inverter": _get_inverter_tikh(reg_fac=2e-2),
            "num_iter": 1,
            "lam": 0.0,
            "lam_learnable": False,
            "final_dc": False,
        },
    ),
)


_append_net(
    "Tiramisu EE jit",
    {
        "name_disp": "TiraFL",
        "name_save": "tiramisu_ee_jit",
        "plt_color": "#1ac938",
        "plt_marker": "o",
        "plt_linestyle": "-.",
        "plt_linewidth": 2.75,
    },
    _load_net(
        "results/tiramisu_ee_jitter_train_phase_2/model_weights.pt",
        Tiramisu,
        tiramisu_params,
        {
            "operator": OpA,
            "inverter": _get_inverter_tikh(),  # placeholder, learned
            "num_iter": 1,
            "lam": 0.0,
            "lam_learnable": False,
            "final_dc": False,
        },
    ),
)
