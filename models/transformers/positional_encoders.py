import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Computes and adds sinusoidal positional embeddings along the temporal axis (T).
    """
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model (int): Dimensionality of the embeddings.
            max_len (int): Maximum sequence length (T) to precompute positions for.
        """
        super().__init__()
        # Create a long enough 'pe' matrix once in log space
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Even indices: sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # Odd indices: cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register pe as a buffer so it's not treated as a learnable parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input embeddings of shape [batch_size, T, d_model, H, W].

        Returns:
            Tensor: The same shape as x, with positional encodings added along T.
        """
        batch_size, T, d_model, H, W = x.shape

        # Validate the d_model matches
        if d_model != self.pe.size(1):
            raise ValueError(f"Input embedding dimension {d_model} does not match positional encoding dimension {self.pe.size(1)}.")

        # Ensure T doesn't exceed precomputed max_len
        if T > self.pe.size(0):
            raise ValueError(f"Temporal sequence length {T} exceeds the maximum length {self.pe.size(0)} of positional encoding.")

        # Extract required positional encodings
        pos_encoding = self.pe[:T, :].unsqueeze(1).unsqueeze(3).unsqueeze(4)  # [T, 1, d_model, 1, 1]

        # Add positional encodings along T, broadcasting over H and W
        x = x + pos_encoding
        return x

class LearnablePositionalEncoding:
    """
    Learnable positional encoding for temporal sequences.
    """
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, x):
        pass


class RelativePositionalEncoding:
    """
    Relative positional encoding for temporal sequences.
    """
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, x, attn_scores):
        pass

