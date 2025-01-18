import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    """
    Fuses fire state encodings, static landscape encodings,
    and wind inputs into a unified representation.
    """

    def __init__(self,
                 fire_dim,       # dimension of the fire state embedding
                 static_dim,     # dimension of the static landscape embedding
                 wind_dim,       # dimension of wind features
                 d_model=128,    # target dimension to project everything to
                 fusion_method='concat',
                 use_mlp=True):
        """
        Args:
            fire_dim (int): Dimension of fire encoding per timestep (e.g., 128).
            static_dim (int): Dimension of static landscape encoding (e.g., 128).
            wind_dim (int): Number of wind features per timestep (e.g., 2 or 4).
            d_model (int): The unified dimension to project everything into.
            fusion_method (str): 'concat' or 'sum'.
            use_mlp (bool): If True, apply an MLP after fusion.
        """
        super().__init__()
        self.fusion_method = fusion_method
        self.use_mlp = use_mlp

        # Projections: map all inputs to the same dimension (d_model)
        # If your fire_encodings are already at d_model, you can skip or set fire_dim=d_model.
        self.fire_projection = nn.Linear(fire_dim, d_model, bias=False) if fire_dim != d_model else nn.Identity()
        self.static_projection = nn.Linear(static_dim, d_model, bias=False) if static_dim != d_model else nn.Identity()
        self.wind_projection = nn.Linear(wind_dim, d_model, bias=False) if wind_dim != d_model else nn.Identity()

        # If using MLP, define a small feedforward to refine fused features
        if self.use_mlp:
            # If fusion_method='concat', the dimension might be d_model * 3. If 'sum', it stays d_model.
            if fusion_method == 'concat':
                in_features = d_model * 3
            elif fusion_method == 'sum':
                in_features = d_model
            else:
                raise ValueError(f"Unknown fusion_method: {fusion_method}")

            self.mlp = nn.Sequential(
                nn.Linear(in_features, d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(d_model, d_model),
            )

    def forward(self, fire_encodings, static_encoding, wind_inputs):
        """
        Args:
            fire_encodings (Tensor): [B, T, fire_dim],
                                     or possibly [B, T, d_model] if already projected.
            static_encoding (Tensor): [B, static_dim],
                                      or [B, d_model].
            wind_inputs (Tensor): [B, T, wind_dim].

        Returns:
            fused (Tensor): [B, T, d_model] fused features over time.
        """
        B, T, _ = fire_encodings.shape

        # 1) Project each to d_model
        fire_enc = self.fire_projection(fire_encodings)          # [B, T, d_model]
        wind_enc = self.wind_projection(wind_inputs)             # [B, T, d_model]

        # Static might need to be repeated for each timestep:
        static_enc = self.static_projection(static_encoding)      # [B, d_model]
        static_enc = static_enc.unsqueeze(1).expand(B, T, -1)     # [B, T, d_model]

        # 2) Fuse
        if self.fusion_method == 'concat':
            fused = torch.cat([fire_enc, static_enc, wind_enc], dim=-1)  # [B, T, 3*d_model]
        elif self.fusion_method == 'sum':
            # Must ensure they're the same shape => [B, T, d_model]
            fused = fire_enc + static_enc + wind_enc
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")

        # 3) Optional MLP to transform the fused feature
        if self.use_mlp:
            # Flatten time dimension, apply MLP, and reshape
            fused = self.mlp(fused.reshape(-1, fused.shape[-1]))  # [(B*T), fused_dim]
            fused = fused.view(B, T, -1)                          # [B, T, d_model]

        return fused

