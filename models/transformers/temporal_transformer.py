import torch
import torch.nn as nn

class TemporalTransformerEncoder(nn.Module):
    """
    A temporal Transformer that processes spatiotemporal embeddings.
    Handles inputs with shape [B, T, C, H, W].
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
            from .positional_encoders import SinusoidalPositionalEncoding
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
            x (Tensor): [batch_size, seq_len, channels, height, width]
            src_mask (Tensor): Optional attention mask of shape [seq_len, seq_len].
            src_key_padding_mask (Tensor): Optional padding mask of shape [batch_size, seq_len].

        Returns:
            encoded (Tensor): [batch_size, seq_len, d_model, height, width],
                              the transformed spatiotemporal sequence embeddings.
        """
        B, T, C, H, W = x.shape

        # Flatten spatial dimensions into sequence
        x = x.view(B, T, C, -1)  # [B, T, C, H*W]
        x = x.permute(0, 1, 3, 2)  # [B, T, H*W, C]
        x = x.flatten(1, 2)  # [B, T*H*W, C]

        # Optionally add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoder(x)

        # Pass through the Transformer
        encoded = self.transformer_encoder(
            x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )  # [B, T*H*W, d_model]

        # Reshape back to spatiotemporal form
        encoded = encoded.view(B, T, H, W, -1)  # [B, T, H, W, d_model]
        encoded = encoded.permute(0, 1, 4, 2, 3)  # [B, T, d_model, H, W]

        return encoded

