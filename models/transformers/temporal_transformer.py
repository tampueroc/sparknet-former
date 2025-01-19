import torch
import torch.nn as nn
from .positional_encoders import SinusoidalPositionalEncoding
from .patch_positional_encoders import Learnable2DPositionalEncoding
from .patch_embedding import PatchEmbedding

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
                 h=50,
                 w=50,
                 patch_size=5,
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
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(d_model, patch_size, d_model)
        if self.use_positional_encoding:
            self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=5000)
            self.patch_pos_encoder = Learnable2DPositionalEncoding(d_model, h // patch_size, w // patch_size)

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
            x (Tensor): [B, T, d_model, H, W] - Fused spatiotemporal embeddings.
            src_mask (Tensor): Optional attention mask of shape [T, T].
            src_key_padding_mask (Tensor): Optional padding mask of shape [B * H * W, T].

        Returns:
            encoded (Tensor): [B, T, d_model, H, W],
                              the transformed spatiotemporal sequence embeddings.
        """
        B, T, C, H, W = x.shape

        # Step 1: Add temporal positional encoding
        # Reshape to [B * H * W, T, d_model] for temporal positional encoding
        x_temporal = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)  # [B * H * W, T, d_model]
        if self.use_positional_encoding:
            x_temporal = self.pos_encoder(x_temporal)  # Add temporal positional encoding

        # Reshape back to [B, T, d_model, H, W]
        x_temporal = x_temporal.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)  # [B, T, d_model, H, W]

        # Step 2: Extract patches and add patch positional encoding
        patch_tokens = []
        H_prime, W_prime = H // self.patch_size, W // self.patch_size
        for t in range(T):
            # Extract patches for each timestep
            patches = self.patch_embed(x_temporal[:, t])  # [B, H' * W', d_model]

            # Add patch positional encoding
            if self.use_positional_encoding:
                patches = self.patch_pos_encoder(patches)  # [B, H' * W', d_model]

            patch_tokens.append(patches)

        # Step 3: Combine patch tokens across timesteps
        patch_tokens = torch.cat(patch_tokens, dim=1)  # [B, T * H' * W', d_model]
        patch_tokens = patch_tokens.permute(0, 2, 1, 3).reshape(B * H_prime * W_prime, T, -1)  # [B * H' * W', T, d_model]

        # Step 3: Adjust valid_tokens mask for spatial dimensions
        if src_key_padding_mask is not None:
            # src_key_padding_mask: [B, T]
            src_key_padding_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            src_key_padding_mask = src_key_padding_mask.expand(-1, H_prime, W_prime, -1)  # [B, H', W', T]
            src_key_padding_mask = src_key_padding_mask.reshape(B * H_prime * W_prime, T)  # [B * H' * W', T]

        # Step 4: Pass through the Transformer
        encoded = self.transformer_encoder(
            patch_tokens, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )  # [B, T * H' * W', d_model]

        # Step 5: Reshape back to spatiotemporal form
        encoded = encoded.view(B, T, H_prime, W_prime, -1).permute(0, 1, 4, 2, 3)  #

        return encoded
