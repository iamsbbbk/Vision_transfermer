import os
import torch
import torch.nn as nn
from torchviz import make_dot

class AdaptiveAttentionFusionModule(nn.Module):
    def __init__(self, in_channels=2048):
        super(AdaptiveAttentionFusionModule, self).__init__()
        self.spatial_attention = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        spatial_attention = self.spatial_attention(x)
        channel_attention = self.channel_attention(x)
        attention = spatial_attention + channel_attention
        return x * attention

