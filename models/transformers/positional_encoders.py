import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    """
    Computes and adds sinusoidal positional embeddings to your input sequences.
    Useful for 1D sequences (e.g., time series, text).
    """

    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model (int): Dimensionality of the embeddings.
            max_len (int): Maximum sequence length to precompute positions for.
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
            x (Tensor): Input embeddings of shape [batch_size, seq_len, d_model].

        Returns:
            Tensor: The same shape as x, with positional encodings added.
        """
        seq_len = x.size(1)

        # pe[:seq_len] => [seq_len, d_model]
        # unsqueeze(0) => [1, seq_len, d_model] for broadcasting
        pos_encoding = self.pe[:seq_len, :].unsqueeze(0)

        # Add the positional encoding to each batch
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

