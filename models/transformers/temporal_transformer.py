import torch
import torch.nn as nn
from .positional_encoders import SinusoidalPositionalEncoding

class TemporalTransformerEncoder(nn.Module):
    """
    A temporal Transformer that processes spatiotemporal embeddings.
    Handles inputs with shape [B, T, C, H, W], preserving temporal order within each sequence.
    """

    def __init__(self,
                 d_model=128,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=512,
                 dropout=0.1,
                 activation='relu',
                 use_positional_encoding=True):
        """
        Args:
            d_model (int): Dimensionality of the embeddings/tokens.
            nhead (int): Number of attention heads in each layer.
            num_layers (int): Number of stacked TransformerEncoderLayers.
            dim_feedforward (int): Dimension of the feedforward network in each layer.
            dropout (float): Dropout probability in attention/feedforward sub-layers.
            activation (str): Activation function of the feedforward network.
            use_positional_encoding (bool): If True, apply sinusoidal positional encoding.
        """
        super().__init__()

        # Positional Encoding
        self.use_positional_encoding = use_positional_encoding
        if self.use_positional_encoding:
            self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=5000)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True  # Ensure batch-first input
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            x (Tensor): [B, T, C, H, W]
            src_mask (Tensor): Optional attention mask of shape [T, T].
            src_key_padding_mask (Tensor): Optional padding mask of shape [B * H * W, T].

        Returns:
            encoded (Tensor): [B, T, d_model, H, W],
                              the transformed spatiotemporal sequence embeddings.
        """
        B, T, C, H, W = x.shape

        # Reshape for processing each spatial position independently
        x = x.permute(0, 3, 4, 1, 2)  # [B, H, W, T, C]
        x = x.reshape(B * H * W, T, C)  # [B * H * W, T, C]

        # Optionally add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoder(x)  # [B * H * W, T, C]

        # Pass through the Transformer
        encoded = self.transformer_encoder(
            x, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )  # [B * H * W, T, d_model]

        # Reshape back to spatiotemporal form
        encoded = encoded.view(B, H, W, T, -1).permute(0, 3, 4, 1, 2)  # [B, T, d_model, H, W]

        return encoded
