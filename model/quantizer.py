import torch
import torch.nn as nn


from config import VQVAEConfig


class Quantizer(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super().__init__()

        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.latent_dim
        self.beta = config.beta

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / self.num_embeddings, 1.0 / self.num_embeddings
        )

    def forward(self, ze: torch.Tensor):
        # ze: (B, latent_dim, H, W) -> (B * H *W , latent_dim)
        B, L, H, W = ze.shape
        ze_flatten = ze.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        # Calculate the distance between each z vector
        # and the embedding vectors
        # ||z - e||² = ||z||² + ||e||² - 2·z·eᵀ
        # ze: (B * H * W, latent_dim)
        # embedding: (num_embeddings, latent_dim)
        d = (
            torch.sum(ze_flatten.pow(2), dim=1, keepdim=True)
            + torch.sum(self.embedding.weight.pow(2), dim=1)
            - 2 * torch.matmul(ze_flatten, self.embedding.weight.t())
        )

        # find closest encodings
        # min_encoding_indices: (B * H * W, 1)
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        # convert to one-hot encoding
        # min_encodings: (B * H * W, num_embeddings)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.num_embeddings
        )  # Set the indices of the closest encodings to 1
        min_encodings.scatter_(dim=1, index=min_encoding_indices, value=1)

        # fetch quantized latent vectors
        z_q = (
            torch.matmul(min_encodings, self.embedding.weight)
            .view(B, H, W, L)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # compute loss for embedding
        # l2 error +  commitment loss
        loss = torch.mean((z_q.detach() - ze) ** 2) + self.beta * torch.mean(
            (z_q - ze.detach()) ** 2
        )

        # preserve gradients
        z_q = ze + (z_q - ze).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return z_q, loss, min_encoding_indices.view(-1, 1)


if __name__ == "__main__":
    # Example usage
    config = VQVAEConfig()
    model = Quantizer(config)

    # Dummy input
    x = torch.randn(
        2, config.latent_dim, 12, 12
    )  # Batch size of 2, 3 channels, 64x64 image
    z_q, loss, min_encoding_indcies = model(x)
    assert z_q.shape == x.shape
    print(z_q.shape)
