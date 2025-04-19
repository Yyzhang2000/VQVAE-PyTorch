import torch
import torch.nn as nn
import torch.nn.functional as F

from config import VQVAEConfig


class Encoder(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super().__init__()

        self.latent_dim = config.latent_dim

        self.encoder_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        config.convbn_channels[i],
                        config.convbn_channels[i + 1],
                        kernel_size=config.conv_kernel_size[i],
                        stride=config.conv_kernel_strides[i],
                        padding=1,
                    ),
                    nn.BatchNorm2d(config.convbn_channels[i + 1]),
                    nn.LeakyReLU(0.2),
                )
                for i in range(config.convbn_blocks - 1)
            ]
        )

        enc_last_idx = config.convbn_blocks
        self.encoder_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    config.convbn_channels[enc_last_idx - 1],
                    config.convbn_channels[enc_last_idx],
                    kernel_size=config.conv_kernel_size[enc_last_idx - 1],
                    stride=config.conv_kernel_strides[enc_last_idx - 1],
                    padding=1,
                ),
            )
        )

    def forward(self, x: torch.Tensor):
        for layer in self.encoder_layers:
            x = layer(x)
        return x
