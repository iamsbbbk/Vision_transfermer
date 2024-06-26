import os
import torch
import torch.nn as nn
from torchviz import make_dot

class AugmentedEdgeGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AugmentedEdgeGraphConvolution, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, edge_index):
        edge_features = self.compute_edge_features(x, edge_index)
        x = self.conv1(x)
        edge_features = self.conv2(edge_features)
        # 调整 edge_features 形状以与 x 兼容
        edge_features = edge_features.view_as(x)
        return x + edge_features

    def compute_edge_features(self, x, edge_index):
        # 假设 edge_index 是一个形状为 (2, num_edges) 的张量
        batch_size, channels, height, width = x.size()
        row, col = edge_index
        row = row.view(-1)
        col = col.view(-1)
        # 提取边缘特征
        edge_features = x[:, :, row, col]  # (batch_size, channels, num_edges)
        # 将 edge_features 调整为 (batch_size, channels, 1, 1)
        edge_features = edge_features.mean(dim=2, keepdim=True).unsqueeze(-1)
        # 扩展 edge_features 以匹配输入特征图的形状
        edge_features = edge_features.expand(batch_size, channels, height, width)
        return edge_features

