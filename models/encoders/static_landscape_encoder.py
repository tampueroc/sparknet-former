import torch
import torch.nn as nn

class StaticLandscapeEncoder(nn.Module):
    """
    CNN encoder for static landscape data.
    Encodes static layers (e.g. topography, vegetation) into a latent spatial embedding.
    """

    def __init__(self, in_channels, base_num_filters, depth, output_dim):
        super().__init__()

        layers = []
        current_channels = in_channels

        for i in range(depth):
            out_channels = base_num_filters * (2**i)
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
            current_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.projection = nn.Conv2d(current_channels, output_dim, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (Tensor): Static input with shape [B, in_channels, H, W].

        Returns:
            embedding (Tensor): Encoded static features, e.g. [B, output_dim, H', W'].
        """
        out = self.conv_layers(x)
        out = self.projection(out)
        out = torch.mean(out, dim=(2, 3))  # [B, output_dim]
        return out

