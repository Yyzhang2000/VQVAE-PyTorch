from dataclasses import dataclass, field
from typing import List, Optional


# @dataclass
# class VQVAEConfig:
#     """Configuration for VQVAE model."""

#     # Model parameters
#     num_embeddings: int = 512
#     embedding_dim: int = 64
#     commitment_cost: float = 0.25
#     decay: float = 0.99
#     epsilon: float = 1e-5

#     # Training parameters
#     batch_size: int = 128
#     learning_rate: float = 1e-3
#     num_epochs: int = 100
#     weight_decay: float = 1e-4

#     # Data parameters
#     image_size: tuple = (128, 128)
#     num_channels: int = 3

#     # Encoder
#     conv_kernel_size: List = field(default_factory=lambda: [3, 4, 4, 3, 3, 3])
#     conv_kernel_strides: List = field(default_factory=lambda: [1, 2, 2, 1, 1, 1])
#     conv_channels: List = field(default_factory=lambda: [16, 32, 64, 128, 64, 1])

#     # Codebook
#     codebook_size: int = 512
#     codebook_dim: int = 64
#     codebook_commitment_cost: float = 0.25

#     # Decoder
#     transpose_kernel_size: list = field(default_factory=lambda: [4, 4, 4, 4, 4])
#     transpose_channels: list = field(default_factory=lambda: [512, 512, 256, 128, 64])
#     latent_dim: int = 32

from dataclasses import dataclass, field
from typing import List


@dataclass
class VQVAEConfig:
    in_channels: int = 3
    convbn_blocks: int = 4
    conv_kernel_size: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    conv_kernel_strides: List[int] = field(default_factory=lambda: [2, 2, 1, 1])
    convbn_channels: List[int] = field(default_factory=lambda: [3, 16, 32, 8, 8])
    conv_activation_fn: str = "leaky"

    transpose_bn_blocks: int = 4
    transposebn_channels: List[int] = field(default_factory=lambda: [8, 8, 32, 16, 3])
    transpose_kernel_size: List[int] = field(default_factory=lambda: [2, 2, 1, 1])
    transpose_kernel_strides: List[int] = field(default_factory=lambda: [2, 2, 1, 1])
    transpose_activation_fn: str = "leaky"

    latent_dim: int = 8
    num_embeddings: int = 512
    beta: float = 0.25
