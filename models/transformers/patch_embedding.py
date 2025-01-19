import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, d_model):
        """
        Args:
            in_channels (int): Number of input channels (e.g., d_model).
            patch_size (int): Size of the patches (e.g., 2, 4).
            d_model (int): Dimensionality of the patch embeddings.
        """
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            patches (Tensor): Patch embeddings of shape [B, H', W', d_model].
        """
        x = self.projection(x)  # [B, d_model, H', W']
        B, C, H, W = x.shape
        return x.view(B, C, H * W).permute(0, 2, 1)  # [B, H'*W', d_model]

