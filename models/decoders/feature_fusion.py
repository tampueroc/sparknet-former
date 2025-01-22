import torch
import torch.nn as nn
from .attention_decoder import ChannelAttentionModule, SpatialAttentionModule

class FeatureFusion(nn.Module):
    def __init__(self,
                 fire_dim,
                 static_dim,
                 wind_dim,
                 d_model=128,
                 fusion_method='concat',
                 use_mlp=True):
        super().__init__()
        self.fusion_method = fusion_method
        self.use_mlp = use_mlp

        # Projections
        self.fire_projection = nn.Linear(fire_dim, d_model, bias=False) if fire_dim != d_model else nn.Identity()
        self.static_projection = nn.Linear(static_dim, d_model, bias=False) if static_dim != d_model else nn.Identity()
        self.wind_projection = nn.Linear(wind_dim, d_model, bias=False) if wind_dim != d_model else nn.Identity()

        # Attention modules
        self.cam = ChannelAttentionModule(d_model)
        self.sam = SpatialAttentionModule()

        # Optional MLP
        if self.use_mlp:
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
        B, T, _, H, W = fire_encodings.shape

        # Project fire encodings
        fire_enc = fire_encodings.permute(0, 1, 3, 4, 2).reshape(-1, fire_encodings.shape[2])
        fire_enc = self.fire_projection(fire_enc).view(B, T, H, W, -1).permute(0, 1, 4, 2, 3)

        # Broadcast static encoding
        static_enc = static_encoding.permute(0, 2, 3, 4, 1).reshape(-1, static_encoding.shape[2])
        static_enc = self.static_projection(static_enc).view(B, 1, H, W, -1).permute(0, 1, 4, 2, 3)
        static_enc = static_enc.expand(-1, T, -1, -1, -1)

        # Project wind inputs
        wind_enc = self.wind_projection(wind_inputs).unsqueeze(-1).unsqueeze(-1)
        wind_enc = wind_enc.expand(-1, -1, -1, H, W)

        # Fuse features
        if self.fusion_method == 'concat':
            fused = torch.cat([fire_enc, static_enc, wind_enc], dim=2)
        elif self.fusion_method == 'sum':
            fused = fire_enc + static_enc + wind_enc
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")

        # Apply channel attention
        fused = self.cam(fused)

        # Apply spatial attention
        fused = self.sam(fused)

        # Optional MLP
        if self.use_mlp:
            fused = fused.permute(0, 1, 3, 4, 2).reshape(-1, fused.shape[-1])
            fused = self.mlp(fused).view(B, T, H, W, -1).permute(0, 1, 4, 2, 3)

        return fused

