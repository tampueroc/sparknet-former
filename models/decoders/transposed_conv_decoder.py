import torch
import torch.nn as nn
import torch.nn.functional as F

class TransposedConvDecoder(nn.Module):
    """
    A decoder that uses a series of transposed convolutions (deconvs) to upsample
    a latent feature map to a full-resolution fire mask.
    """

    def __init__(self,
                 in_channels=128,    # latent dim coming in
                 base_num_filters=64,
                 depth=3,            # how many upsampling (deconv) blocks
                 out_channels=1,     # e.g. 1 for a binary fire mask
                 use_batchnorm=False,
                 final_activation=None):
        """
        Args:
            in_channels (int): Number of channels in the latent representation.
            base_num_filters (int): Number of filters in the first upsampling block.
            depth (int): Number of upsampling blocks. Each block typically doubles spatial size.
            out_channels (int): Number of channels in final output (1 for binary mask).
            use_batchnorm (bool): Whether to apply BatchNorm after each transposed conv.
            final_activation (str): e.g. 'sigmoid' or 'softmax' (optional).
        """
        super().__init__()

        # We'll build a list of upsampling layers
        # For each 'depth', we will do a ConvTranspose2d that doubles H' and W' (stride=2, kernel_size=2, etc.)
        # The number of filters can decrease as we upsample (similar to UNet).
        modules = []
        current_in = in_channels

        for i in range(depth):
            # For example, reduce filter size each upsampling step:
            # out = base_num_filters / 2^(i) – or keep it constant if you prefer.
            out_filters = base_num_filters // (2**i) if (base_num_filters // (2**i) > 0) else 1

            modules.append(
                nn.ConvTranspose2d(
                    in_channels=current_in,
                    out_channels=out_filters,
                    kernel_size=2,
                    stride=2,
                    padding=0
                )
            )
            if use_batchnorm:
                modules.append(nn.BatchNorm2d(out_filters))

            modules.append(nn.ReLU(inplace=True))

            current_in = out_filters

        # After the upsampling blocks, we produce the desired output channels
        self.upsampling = nn.Sequential(*modules)

        self.output_conv = nn.Conv2d(current_in, out_channels, kernel_size=1, stride=1, padding=0)

        # Optional final activation layer
        if final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif final_activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)  # if multi-channel
        else:
            self.final_activation = None

    def forward(self, x):
        """
        Args:
            x (Tensor): [B, in_channels, H', W'] – the latent feature map.

        Returns:
            out (Tensor): [B, out_channels, H, W] – the upsampled fire state prediction.
        """
        # 1) Sequentially apply upsampling (deconvs)
        out = self.upsampling(x)
        # 2) Final 1x1 conv to get the desired number of output channels
        out = self.output_conv(out)
        # 3) Optional final activation
        if self.final_activation:
            out = self.final_activation(out)
        # Resize to exact dimensions (512 -> 400)
        out = out[:, :, 56:456, 56:456]  # Crop the center region for 400x400
        return out

