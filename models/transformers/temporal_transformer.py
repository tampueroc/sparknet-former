import torch.nn as nn

class TemporalTransformerEncoder(nn.Module):
    """
    A temporal Transformer that processes a sequence of embeddings over time.
    Uses PyTorch's TransformerEncoder under the hood.
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

        # If you want optional sinusoidal positional encoding:
        self.use_positional_encoding = use_positional_encoding
        if self.use_positional_encoding:
            from .positional_encoders import SinusoidalPositionalEncoding
            self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=5000)

        # Build one TransformerEncoderLayer, then stack it num_layers times
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            x (Tensor): [batch_size, seq_len, d_model]
            src_mask (Tensor): Optional attention mask of shape [seq_len, seq_len].
            src_key_padding_mask (Tensor): Optional padding mask of shape [batch_size, seq_len].

        Returns:
            encoded (Tensor): [batch_size, seq_len, d_model],
                              the transformed sequence embeddings.
        """
        # Optionally add sinusoidal positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoder(x)

        # The PyTorch Transformer expects shape [seq_len, batch_size, d_model]
        x = x.permute(1, 0, 2)  # => [T, B, d_model]

        # Pass through the stacked TransformerEncoder
        # - src_mask is optional if you want to mask out certain timesteps
        # - src_key_padding_mask is for ignoring padded tokens in the batch
        encoded = self.transformer_encoder(
            x,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        # Permute back to [B, T, d_model]
        encoded = encoded.permute(1, 0, 2)
        return encoded

