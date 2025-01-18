import torch
import torch.nn as nn
import torch.nn.functional as F

class FireStateEncoder(nn.Module):
    """
    CNN encoder for the fire state bitmask sequence.
    Encodes each timestep's 2D bitmask into a latent spatial embedding.
    """

    def __init__(self, in_channels, base_num_filters, depth, output_dim):
        """
        Args:
            in_channels (int): Number of input channels in a single bitmask.
            base_num_filters (int): Number of filters in the first conv layer.
            depth (int): How many convolution layers to stack.
            output_dim (int): Dimensionality of the final latent embedding.
        """
        super().__init__()

        # Example: A simple stack of Convs -> ReLU -> optional Pool
        layers = []
        current_channels = in_channels

        for i in range(depth):
            out_channels = base_num_filters * (2**i)  # e.g. double filters each layer
            layers.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            layers.append(nn.ReLU(inplace=True))
            # Optionally add BatchNorm, Dropout, or pooling
            # layers.append(nn.BatchNorm2d(out_channels))
            current_channels = out_channels

        # Final projection layer to get desired embedding dimension
        self.conv_layers = nn.Sequential(*layers)
        self.projection = nn.Conv2d(current_channels, output_dim, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (Tensor): Fire mask sequence with shape [B, T, in_channels, H, W]
                        or [B, T, H, W] if in_channels=1.

        Returns:
            embeddings (Tensor): Encoded features with shape [B, T, output_dim, H', W']
                                 or flattened if you prefer [B, T, output_dim].
        """
        # 1) If x has shape [B, T, H, W], add a channel dim.
        if x.dim() == 4:
            x = x.unsqueeze(2)  # Now [B, T, 1, H, W]

        B, T, C, H, W = x.shape
        # We'll encode each timestep individually.

        embeddings = []
        for t in range(T):
            # shape for that timestep: [B, in_channels, H, W]
            x_t = x[:, t, :, :, :]  # [B, C, H, W]
            out_t = self.conv_layers(x_t)      # [B, _, H', W']
            out_t = self.projection(out_t)     # [B, output_dim, H', W']
            embeddings.append(out_t)

        # Stack along time dimension => [B, T, output_dim, H', W']
        embeddings = torch.stack(embeddings, dim=1)

        return embeddings

