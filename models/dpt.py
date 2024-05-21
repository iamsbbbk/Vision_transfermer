import torch
import torch.nn as nn
import torchvision.models as models
from .adaptive_attention import AdaptiveAttentionFusionModule
from .augmented_edge_gcn import AugmentedEdgeGraphConvolution

class DPT(nn.Module):
    def __init__(self, num_classes=21, backbone='resnet50', pretrained=True):
        super(DPT, self).__init__()

        # Backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError("Unsupported backbone")

        self.encoder = nn.Sequential(*list(self.backbone.children())[:-2])

        # Adaptive Attention Fusion Module
        self.aafm = AdaptiveAttentionFusionModule(in_channels=2048)

        # Augmented Edge Graph Convolution
        self.ae_gcn = AugmentedEdgeGraphConvolution(in_channels=2048, out_channels=512)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        features = self.encoder(x)

        # Adaptive Attention Fusion
        features = self.aafm(features)

        # Augmented Edge Graph Convolution
        edge_index = self.compute_edge_index(features)  # Placeholder for actual edge index computation
        features = self.ae_gcn(features, edge_index)

        # Decoder
        out = self.decoder(features)
        out = nn.functional.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=False)

        return out

    def compute_edge_index(self, x):
        # Placeholder function to compute edge indices for AE-GCN
        batch_size, channels, height, width = x.size()
        row = torch.arange(height).repeat_interleave(width).view(1, -1)
        col = torch.arange(width).repeat(height).view(1, -1)
        edge_index = torch.cat([row, col], dim=0)
        return edge_index