import torch
import torch.nn as nn
import torch.nn.functional as F

from config import VQVAEConfig
from quantizer import Quantizer
from encoder import Encoder
from decoder import Decoder


class VQVAE(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super().__init__()

        self.encoder = Encoder(config)
        self.quantizer = Quantizer(config)
        self.decoder = Decoder(config)

        self.pre_quant_conv = nn.Conv2d(
            config.convbn_channels[-1], config.latent_dim, kernel_size=1
        )
        self.post_quant_conv = nn.Conv2d(
            config.latent_dim, config.transposebn_channels[0], kernel_size=1
        )

    def forward(self, x: torch.Tensor):
        # Encoder
        z_e = self.encoder(x)
        z_e = self.pre_quant_conv(z_e)
        # Quantization
        z_q, commitment_loss, encoding_indices = self.quantizer(z_e)

        # Decoder
        z_q = self.post_quant_conv(z_q)
        x_recon = self.decoder(z_q)

        return x_recon, commitment_loss, encoding_indices

    def decode_from_indices(self, indices: torch.Tensor):
        # Convert indices to one-hot encoding
        one_hot = F.one_hot(indices, self.quantizer.num_embeddings).float()
        z_q = torch.matmul(one_hot, self.quantizer.embeddings.weight)

        # Decode
        z_q = self.post_quant_conv(z_q)
        x_recon = self.decoder(z_q)

        return x_recon


if __name__ == "__main__":
    # Example usage
    config = VQVAEConfig()
    model = VQVAE(config)

    # Dummy input
    x = torch.randn(12, config.in_channels, 32, 32)  # Batch size of 8

    # Forward pass
    x_recon, commitment_loss, encoding_indices = model(x)

    print("Reconstructed shape:", x_recon.shape)
    print("Commitment loss:", commitment_loss)
    print("Encoding indices shape:", encoding_indices.shape)
