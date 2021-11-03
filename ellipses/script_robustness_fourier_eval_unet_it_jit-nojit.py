import os

import matplotlib.pyplot as plt
import torch

from matplotlib import rc
from tqdm import tqdm

from data_management import IPDataset
from networks import IterativeNet, UNet
from operators import Fourier as Fourier
from operators import Fourier_matrix as Fourier_m
from operators import (
    LearnableInverter,
    RadialMaskFunc,
    to_complex,
    unprep_fft_channel,
)


# ----- load configuration -----
import config  # isort:skip

# ----- global configuration -----
device = torch.device("cuda:0")
torch.cuda.set_device(0)
save_path = os.path.join(config.RESULTS_PATH, "eval", "unet_it_jit-nojit")
save_plot = True

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
    "lam_learnable": False,
    "final_dc": True,
    "resnet_factor": 1.0,
    "operator": OpA_m,
    "inverter": inverter,
}

# ------ construct network and load weights -----
subnet = subnet(**subnet_params).to(device)
it_net = IterativeNet(subnet, **it_net_params).to(device)
it_net.load_state_dict(
    torch.load(
        "results/Fourier_UNet_it_jit-nojit_train_phase_1/model_weights.pt",
        map_location=torch.device(device),
    )
)
it_net.freeze()
it_net.eval()

# ----- evaluation setup -----

# select samples
samples = range(150)
test_data = IPDataset("test", config.DATA_PATH)

# dynamic range for plotting
v_min = 0.0
v_max = 0.9


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


# LaTeX typesetting
rc("font", **{"family": "serif", "serif": ["Palatino"]})
rc("text", usetex=True)


# ----- run evaluation -----
if save_plot:
    os.makedirs(save_path, exist_ok=True)

for sample in tqdm(samples):
    X_0 = test_data[sample][0]
    X_0 = to_complex(X_0.to(device)).unsqueeze(0)
    Y_0 = OpA(X_0)
    X_rec = it_net(Y_0)

    fig, ax = plt.subplots(clear=True, figsize=(2.5, 2.5), dpi=200)
    im = _implot(ax, X_rec)
    plt.title("ItNet jit. (mod.) / sample {}".format(sample))

    if save_plot:
        fig.savefig(
            os.path.join(
                save_path,
                "fig_unet_it_jit-nojit_example_S{}.png".format(sample),
            ),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()
    else:
        plt.show()
