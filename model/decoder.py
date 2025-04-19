import torch
import torch.nn as nn
import torch.nn.functional as F

from config import VQVAEConfig


class Decoder(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super().__init__()

        self.latent_dim = config.latent_dim

        self.layers = nn.ModuleList()

        # First layer: from embedding_dim → transpose_channels[0]
        self.latent_dim = config.latent_dim

        self.decoder_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        config.transposebn_channels[i],
                        config.transposebn_channels[i + 1],
                        kernel_size=config.transpose_kernel_size[i],
                        stride=config.transpose_kernel_strides[i],
                        padding=0,
                    ),
                    nn.BatchNorm2d(config.transposebn_channels[i + 1]),
                    nn.LeakyReLU(0.2),
                )
                for i in range(config.transpose_bn_blocks - 1)
            ]
        )

        dec_last_idx = config.transpose_bn_blocks
        self.decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    config.transposebn_channels[dec_last_idx - 1],
                    config.transposebn_channels[dec_last_idx],
                    kernel_size=config.transpose_kernel_size[dec_last_idx - 1],
                    stride=config.transpose_kernel_strides[dec_last_idx - 1],
                    padding=0,
                ),
                nn.Tanh(),
            )
        )

    def forward(self, x: torch.Tensor):
        for layer in self.decoder_layers:
            x = layer(x)

        return x
