impor torch
import torch.nn as nn


class Learnable2DPositionalEncoding(nn.Module):
    def __init__(self, d_model, h, w):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, h * w, d_model))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [B, H'*W', d_model].

        Returns:
            Tensor: Input tensor with positional encodings added.
        """
        return x + self.pos_embedding
