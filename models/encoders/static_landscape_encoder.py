import torch
import torch.nn as nn

class StaticLandscapeEncoder(nn.Module):
    """
    CNN encoder for static landscape data.
    Encodes static layers (e.g. topography, vegetation) into a latent spatial embedding.
    """

    def __init__(self, in_channels, base_num_filters, depth, output_dim, reduction_factor=4):
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
                    stride=2 if reduction_factor > 1 else 1,  # Reduce spatial dimensions progressively
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
            x (Tensor): Static input with shape [B, T=1, C, H, W].


        Returns:
            embedding (Tensor): Encoded static features with shape [B, T=1, output_dim, H', W'].
        """
        B, T, C, H, W = x.shape
        assert T == 1, "StaticLandscapeEncoder expects T=1 for static data."

        # Flatten temporal dimension and process the static landscape
        x = x.view(B * T, C, H, W)  # Shape becomes [B * T, C, H, W]
        out = self.conv_layers(x)  # [B * T, _, H', W']
        out = self.projection(out)  # [B * T, output_dim, H', W']
        out = self.conv_layers(x)

        # Reshape back to include the temporal dimension T=1
        out = out.view(B, T, -1, out.shape[2], out.shape[3])  # [B, T, output_dim, H', W']
        return out
