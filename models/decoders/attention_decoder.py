import torch
import torch.nn as nn

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, T, H, W = x.size()

        # Perform global pooling (output shape: B x C x 1 x 1 x 1)
        avg_out = self.avg_pool(x).view(B, C)  # B x C
        max_out = self.max_pool(x).view(B, C)  # B x C

        # Pass through shared FC layers
        avg_out = self.fc(avg_out).view(B, C, 1, 1, 1)
        max_out = self.fc(max_out).view(B, C, 1, 1, 1)

        # Combine and apply sigmoid
        out = self.sigmoid(avg_out + max_out)
        return x * out  # Scale input features by attention weights

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Average across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max across channels
        combined = torch.cat([avg_out, max_out], dim=1)  # Concatenate along channel dimension
        attention = self.sigmoid(self.conv(combined))
        return x * attention

