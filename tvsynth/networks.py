import os

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
import torch

from tqdm import tqdm

from operators import l2_error


# ----- ----- Abstract Base Network ----- -----


class InvNet(torch.nn.Module, metaclass=ABCMeta):
    """ Abstract base class for networks solving linear inverse problems.

    The network is intended for the denoising of a direct inversion of a 1D
    signal from (noisy) linear measurements. The measurement model

        y = Ax + noise

    can be used to obtain an approximate reconstruction x_ from y using, e.g.,
    the pseudo-inverse of A. The task of the network is either to directly
    obtain x from y or denoise and improve this first inversion x_ towards x.

    """

    def __init__(self):
        super(InvNet, self).__init__()

    @abstractmethod
    def forward(self, z):
        """ Applies the network to a batch of inputs z. """
        pass

    def freeze(self):
        """ Freeze all model weights, i.e. prohibit further updates. """
        for param in self.parameters():
            param.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    def _train_step(
        self, batch_idx, batch, loss_func, optimizer, batch_size, acc_steps
    ):
        inp, tar = batch
        inp = inp.to(self.device)
        tar = tar.to(self.device)
        pred = self.forward(inp)

        loss = loss_func(pred, tar) / acc_steps
        loss.backward()
        if (batch_idx // batch_size + 1) % acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss * acc_steps, inp, tar, pred

    def _val_step(self, batch_idx, batch, loss_func):
        inp, tar = batch
        inp = inp.to(self.device)
        tar = tar.to(self.device)
        pred = self.forward(inp)
        loss = loss_func(pred, tar)
        return loss, inp, tar, pred

    def _on_epoch_end(
        self,
        epoch,
        save_epochs,
        save_path,
        logging,
        loss,
        inp,
        tar,
        pred,
        v_loss,
        v_inp,
        v_tar,
        v_pred,
        val_data,
    ):

        self._print_info()

        logging = logging.append(
            {
                "loss": loss.item(),
                "val_loss": v_loss.item(),
                "rel_l2_error": l2_error(
                    pred, tar, relative=True, squared=False
                )[0].item(),
                "val_rel_l2_error": l2_error(
                    v_pred, v_tar, relative=True, squared=False
                )[0].item(),
            },
            ignore_index=True,
            sort=False,
        )

        print(logging.tail(1))

        if (epoch + 1) % save_epochs == 0:
            fig = self._create_figure(
                logging, loss, inp, tar, pred, v_loss, v_inp, v_tar, v_pred
            )

            os.makedirs(save_path, exist_ok=True)
            torch.save(
                self.state_dict(),
                os.path.join(
                    save_path, "model_weights_epoch{}.pt".format(epoch + 1)
                ),
            )
            logging.to_pickle(
                os.path.join(
                    save_path, "losses_epoch{}.pkl".format(epoch + 1)
                ),
            )

            if fig is not None:
                fig.savefig(
                    os.path.join(
                        save_path, "plot_epoch{}.png".format(epoch + 1)
                    ),
                    bbox_inches="tight",
                )

        return logging

    def _create_figure(
        self, logging, loss, inp, tar, pred, v_loss, v_inp, v_tar, v_pred
    ):
        """ Can be overwritten by child classes to plot training progress. """
        pass

    def _add_to_progress_bar(self, dict):
        """ Can be overwritten by child classes to add to progress bar. """
        return dict

    def _on_train_end(self, save_path, logging):
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            self.state_dict(), os.path.join(save_path, "model_weights.pt")
        )
        logging.to_pickle(os.path.join(save_path, "losses.pkl"),)

    def _print_info(self):
        """ Can be overwritten by child classes to print at epoch end. """
        pass

    def train_on(
        self,
        train_data,
        val_data,
        num_epochs,
        batch_size,
        loss_func,
        save_path,
        save_epochs=50,
        optimizer=torch.optim.Adam,
        optimizer_params={"lr": 2e-4, "eps": 1e-3},
        scheduler=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 1, "gamma": 1.0},
        acc_steps=1,
        train_transform=None,
        val_transform=None,
    ):
        optimizer = optimizer(self.parameters(), **optimizer_params)
        scheduler = scheduler(optimizer, **scheduler_params)

        logging = pd.DataFrame(
            columns=["loss", "val_loss", "rel_l2_error", "val_rel_l2_error"]
        )

        inp_train, tar_train = train_data
        inp_val, tar_val = val_data

        for epoch in range(num_epochs):
            permutation = torch.randperm(inp_train.shape[0])
            self.train()  # make sure we are in train mode
            t = tqdm(
                range(0, inp_train.shape[0], batch_size),
                desc="epoch {} / {}".format(epoch, num_epochs),
            )
            optimizer.zero_grad()
            loss = 0.0
            for i in t:
                indices = permutation[i : i + batch_size]
                batch = (
                    train_transform(inp_train[indices, ...])
                    if train_transform is not None
                    else inp_train[indices, ...],
                    tar_train[indices, ...],
                )
                loss_b, inp, tar, pred = self._train_step(
                    i, batch, loss_func, optimizer, batch_size, acc_steps
                )
                t.set_postfix(
                    **self._add_to_progress_bar({"loss": loss_b.item()})
                )
                with torch.no_grad():
                    loss += loss_b
            loss /= -(-inp_train.shape[0] // batch_size)

            with torch.no_grad():
                self.eval()  # make sure we are in eval mode
                scheduler.step()
                v_loss = 0.0
                for ii in range(0, inp_val.shape[0], batch_size):
                    v_batch = (
                        val_transform(inp_val[ii : ii + batch_size])
                        if val_transform is not None
                        else inp_val[ii : ii + batch_size],
                        tar_val[ii : ii + batch_size],
                    )
                    v_loss_b, v_inp, v_tar, v_pred = self._val_step(
                        ii, v_batch, loss_func
                    )
                    v_loss += v_loss_b
                v_loss /= -(-inp_val.shape[0] // batch_size)

                logging = self._on_epoch_end(
                    epoch,
                    save_epochs,
                    save_path,
                    logging,
                    loss,
                    inp,
                    tar,
                    pred,
                    v_loss,
                    v_inp,
                    v_tar,
                    v_pred,
                    val_data,
                )

        self._on_train_end(save_path, logging)
        return logging


# ----- ----- Iterative Networks ----- -----


class IterativeNet(InvNet):
    def __init__(
        self,
        subnet,
        operator,
        inverter,
        num_iter,
        lam,
        lam_learnable,
        final_dc=True,
        resnet_factor=1.0,
    ):
        super(IterativeNet, self).__init__()
        self.operator = operator
        self.inverter = inverter
        self.subnet = subnet
        self.num_iter = num_iter
        self.final_dc = final_dc
        self.resnet_factor = resnet_factor
        if not isinstance(lam, (list, tuple)):
            lam = [lam] * num_iter
        if not isinstance(lam_learnable, (list, tuple)):
            lam_learnable = [lam_learnable] * len(lam)

        self.lam = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.tensor(lam[it]), requires_grad=lam_learnable[it]
                )
                for it in range(len(lam))
            ]
        )

    def forward(self, inp):
        xinv = self.inverter(inp)
        for it in range(self.num_iter):
            # subnet step
            xinv = self.resnet_factor * xinv + self.subnet(xinv)

            # data consistency step
            if (self.final_dc) or (
                (not self.final_dc) and it < self.num_iter - 1
            ):
                xinv = xinv - self.lam[it] * self.operator.adj(
                    self.operator(xinv) - inp
                )

        return xinv

    def _print_info(self):
        print("Current lambda(s):")
        print([self.lam[it].item() for it in range(len(self.lam))])
        print([self.lam[it].requires_grad for it in range(len(self.lam))])

    def _create_figure(
        self, logging, loss, inp, tar, pred, v_loss, v_inp, v_tar, v_pred
    ):

        fig, subs = plt.subplots(1, 3, clear=True, num=1, figsize=(8, 6))

        inv = self.inverter(inp)
        v_inv = self.inverter(v_inp)

        # training and validation loss
        subs[0].set_title("losses")
        subs[0].semilogy(logging["loss"], label="train")
        subs[0].semilogy(logging["val_loss"], label="val")
        subs[0].legend()

        # validation
        subs[1].plot(v_tar[0, 0, :].detach().cpu(), label="sig")
        subs[1].plot(v_inv[0, 0, :].detach().cpu(), label="inv")
        subs[1].plot(v_pred[0, 0, :].detach().cpu(), label="rec")
        subs[1].legend()
        subs[1].set_title(
            "val output:\n ||x0-xrec||_2 / ||x0||_2\n = "
            "{:1.2e}".format(logging["val_rel_l2_error"].iloc[-1])
        )

        # training output
        subs[2].plot(tar[0, 0, :].detach().cpu(), label="sig")
        subs[2].plot(inv[0, 0, :].detach().cpu(), label="inv")
        subs[2].plot(pred[0, 0, :].detach().cpu(), label="rec")
        subs[2].legend()
        subs[2].set_title(
            "train output:\n ||x0-xrec||_2 / ||x0||_2\n = "
            "{:1.2e}".format(logging["rel_l2_error"].iloc[-1])
        )

        return fig


# ----- ----- U-Net ----- -----


class UNet(InvNet):
    """ U-Net implementation.

    Based on https://github.com/mateuszbuda/brain-segmentation-pytorch/
    and modified in agreement with their licence:

    -----

    MIT License

    Copyright (c) 2019 mateuszbuda

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

    """

    def __init__(
        self, in_channels=1, out_channels=1, base_features=32, drop_factor=0.0,
    ):
        # set properties of UNet
        super(UNet, self).__init__()

        self.encoder1 = UNet._conv_block(
            in_channels,
            base_features,
            drop_factor=drop_factor,
            block_name="encoding_1",
        )
        self.pool1 = torch.nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder2 = UNet._conv_block(
            base_features,
            base_features * 2,
            drop_factor=drop_factor,
            block_name="encoding_2",
        )
        self.pool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder3 = UNet._conv_block(
            base_features * 2,
            base_features * 4,
            drop_factor=drop_factor,
            block_name="encoding_3",
        )
        self.pool3 = torch.nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder4 = UNet._conv_block(
            base_features * 4,
            base_features * 8,
            drop_factor=drop_factor,
            block_name="encoding_4",
        )
        self.pool4 = torch.nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = UNet._conv_block(
            base_features * 8,
            base_features * 16,
            drop_factor=drop_factor,
            block_name="bottleneck",
        )

        self.upconv4 = torch.nn.ConvTranspose1d(
            base_features * 16, base_features * 8, kernel_size=2, stride=2,
        )
        self.decoder4 = UNet._conv_block(
            base_features * 16,
            base_features * 8,
            drop_factor=drop_factor,
            block_name="decoding_4",
        )
        self.upconv3 = torch.nn.ConvTranspose1d(
            base_features * 8, base_features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._conv_block(
            base_features * 8,
            base_features * 4,
            drop_factor=drop_factor,
            block_name="decoding_3",
        )
        self.upconv2 = torch.nn.ConvTranspose1d(
            base_features * 4, base_features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._conv_block(
            base_features * 4,
            base_features * 2,
            drop_factor=drop_factor,
            block_name="decoding_2",
        )
        self.upconv1 = torch.nn.ConvTranspose1d(
            base_features * 2, base_features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._conv_block(
            base_features * 2,
            base_features,
            drop_factor=drop_factor,
            block_name="decoding_1",
        )

        self.outconv = torch.nn.Conv1d(
            in_channels=base_features,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x):

        enc1 = self.encoder1(x)

        enc2 = self.encoder2(self.pool1(enc1))

        enc3 = self.encoder3(self.pool2(enc2))

        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.outconv(dec1)

    @staticmethod
    def _conv_block(in_channels, out_channels, drop_factor, block_name):
        return torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        block_name + "conv1",
                        torch.nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (block_name + "bn_1", torch.nn.BatchNorm1d(out_channels)),
                    (block_name + "relu1", torch.nn.ReLU(True)),
                    (block_name + "dr1", torch.nn.Dropout(p=drop_factor)),
                    (
                        block_name + "conv2",
                        torch.nn.Conv1d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (block_name + "bn_2", torch.nn.BatchNorm1d(out_channels)),
                    (block_name + "relu2", torch.nn.ReLU(True)),
                    (block_name + "dr2", torch.nn.Dropout(p=drop_factor)),
                ]
            )
        )


# ----- ----- Tiramisu Network ----- -----


class Tiramisu(InvNet):
    """ Tiramisu network implementation.

    Based on https://github.com/bfortuner/pytorch_tiramisu
    and modified in agreement with their licence:

    -----

    MIT License

    Copyright (c) 2018 Brendan Fortuner

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        drop_factor=0.0,
        down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5),
        pool_factors=(2, 2, 2, 2, 2),
        bottleneck_layers=5,
        growth_rate=8,
        out_chans_first_conv=16,
    ):
        super(Tiramisu, self).__init__()

        self.down_blocks = down_blocks
        self.up_blocks = up_blocks

        # init counts of channels
        cur_channels_count = 0
        skip_connection_channel_counts = []

        # First Convolution
        self.bn_layer = torch.nn.BatchNorm1d(out_chans_first_conv)
        self.add_module(
            "firstconv",
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_chans_first_conv,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )
        cur_channels_count = out_chans_first_conv

        # Downsampling path
        self.denseBlocksDown = torch.nn.ModuleList([])
        self.transDownBlocks = torch.nn.ModuleList([])
        for i in range(len(self.down_blocks)):
            self.denseBlocksDown.append(
                Tiramisu._DenseBlock(
                    cur_channels_count,
                    growth_rate,
                    self.down_blocks[i],
                    drop_factor,
                )
            )
            cur_channels_count += growth_rate * self.down_blocks[i]
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(
                Tiramisu._TransitionDown(
                    cur_channels_count, drop_factor, pool_factors[i]
                )
            )

        # Bottleneck
        self.add_module(
            "bottleneck",
            Tiramisu._Bottleneck(
                cur_channels_count,
                growth_rate,
                bottleneck_layers,
                drop_factor,
            ),
        )
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        # Upsampling path
        self.transUpBlocks = torch.nn.ModuleList([])
        self.denseBlocksUp = torch.nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(
                Tiramisu._TransitionUp(
                    prev_block_channels,
                    prev_block_channels,
                    pool_factors[-i - 1],
                )
            )
            cur_channels_count = (
                prev_block_channels + skip_connection_channel_counts[i]
            )

            self.denseBlocksUp.append(
                Tiramisu._DenseBlock(
                    cur_channels_count,
                    growth_rate,
                    up_blocks[i],
                    drop_factor,
                    upsample=True,
                )
            )
            prev_block_channels = growth_rate * self.up_blocks[i]
            cur_channels_count += prev_block_channels

        # Final DenseBlock
        self.transUpBlocks.append(
            Tiramisu._TransitionUp(
                prev_block_channels, prev_block_channels, pool_factors[0]
            )
        )
        cur_channels_count = (
            prev_block_channels + skip_connection_channel_counts[-1]
        )

        self.denseBlocksUp.append(
            Tiramisu._DenseBlock(
                cur_channels_count,
                growth_rate,
                self.up_blocks[-1],
                drop_factor,
                upsample=False,
            )
        )
        cur_channels_count += growth_rate * self.up_blocks[-1]

        # Final Conv layer
        self.finalConv = torch.nn.Conv1d(
            in_channels=cur_channels_count,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        out = self.bn_layer(self.firstconv((x)))

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        return out

    # ----- Blocks for Tiramisu -----

    class _DenseLayer(torch.nn.Sequential):
        def __init__(self, in_channels, growth_rate, p):
            super().__init__()
            self.add_module("bn", torch.nn.BatchNorm1d(in_channels))
            self.add_module("relu", torch.nn.ReLU(True))
            self.add_module(
                "conv",
                torch.nn.Conv1d(
                    in_channels,
                    growth_rate,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
            )
            self.add_module("drop", torch.nn.Dropout(p=p))

        def forward(self, x):
            return super().forward(x)

    class _DenseBlock(torch.nn.Module):
        def __init__(
            self, in_channels, growth_rate, n_layers, p, upsample=False
        ):
            super().__init__()
            self.upsample = upsample
            self.layers = torch.nn.ModuleList(
                [
                    Tiramisu._DenseLayer(
                        in_channels + i * growth_rate, growth_rate, p
                    )
                    for i in range(n_layers)
                ]
            )

        def forward(self, x):
            if self.upsample:
                new_features = []
                # we pass all previous activations to each dense layer normally
                # But we only store each layer's output in the new_features
                for layer in self.layers:
                    out = layer(x)
                    x = torch.cat([x, out], dim=1)
                    new_features.append(out)
                return torch.cat(new_features, dim=1)
            else:
                for layer in self.layers:
                    out = layer(x)
                    x = torch.cat([x, out], dim=1)  # 1 = channel axis
                return x

    class _TransitionDown(torch.nn.Sequential):
        def __init__(self, in_channels, p, pool_factor):
            super().__init__()
            self.add_module("bn", torch.nn.BatchNorm1d(in_channels))
            self.add_module("relu", torch.nn.ReLU(inplace=True))
            self.add_module(
                "conv",
                torch.nn.Conv1d(
                    in_channels,
                    in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
            )
            self.add_module("drop", torch.nn.Dropout(p))
            self.add_module(
                "maxpool",
                torch.nn.MaxPool1d(
                    kernel_size=pool_factor, stride=pool_factor
                ),
            )

        def forward(self, x):
            return super().forward(x)

    class _TransitionUp(torch.nn.Module):
        def __init__(self, in_channels, out_channels, pool_factor):
            super().__init__()
            self.convTrans = torch.nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=pool_factor,
                padding=0,
                bias=True,
            )

        def forward(self, x, skip):
            out = self.convTrans(x)
            out = Tiramisu._center_crop(out, skip.size(2))
            out = torch.cat([out, skip], dim=1)
            return out

    class _Bottleneck(torch.nn.Sequential):
        def __init__(self, in_channels, growth_rate, n_layers, p):
            super().__init__()
            self.add_module(
                "bottleneck",
                Tiramisu._DenseBlock(
                    in_channels, growth_rate, n_layers, p, upsample=True
                ),
            )

        def forward(self, x):
            return super().forward(x)

    def _center_crop(layer, max_width):
        _, _, w = layer.size()
        c = (w - max_width) // 2
        return layer[:, :, c : (c + max_width)]
