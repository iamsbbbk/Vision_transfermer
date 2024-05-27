import torch
import torch.nn as nn
import os
from torchviz import make_dot


def kmeans(X, n_clusters=5, n_iters=100, tol=1e-4):

    centroids = X[torch.randperm(X.size(0))[:n_clusters]]

    for _ in range(n_iters):
        # 计算所有点到每个质心的距离
        distances = torch.cdist(X, centroids)  # 使用torch.cdist计算距离矩阵

        # 为每个点分配最近的质心
        labels = torch.argmin(distances, dim=1)

        # 更新质心
        new_centroids = []
        for j in range(n_clusters):
            # 选取属于当前簇的所有点
            members = X[labels == j]
            if len(members) > 0:
                new_centroids.append(members.mean(dim=0))
            else:
                # 如果某个簇为空，则随机重新选择一个质心
                new_centroids.append(X[torch.randint(0, X.size(0), (1,))])

        new_centroids = torch.stack(new_centroids)

        # 检查收敛（质心变化是否小于容忍度）
        centroid_shift = torch.norm(new_centroids - centroids, dim=1).sum()
        if centroid_shift <= tol:
            break

        centroids = new_centroids

    return labels, centroids


class CandidateRegionGenerator(nn.Module):
    def __init__(self, in_channels, num_subclasses=5):
        super(CandidateRegionGenerator, self).__init__()
        self.num_subclasses = num_subclasses
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        features = self.bn3(self.conv3(x))

        batch_size, channels, height, width = features.size()
        features_flat = features.view(batch_size, channels, -1).permute(0, 2, 1).contiguous()

        candidate_regions = []
        for i in range(batch_size):
            # 使用自定义的PyTorch KMeans
            labels, _ = kmeans(features_flat[i], n_clusters=self.num_subclasses)
            candidate_region = torch.zeros_like(features_flat[i])
            for j in range(self.num_subclasses):
                candidate_region[labels == j] = torch.mean(features_flat[i][labels == j], dim=0)
            candidate_regions.append(candidate_region)
        candidate_regions = torch.stack(candidate_regions).permute(0, 2, 1).view(batch_size, channels, height, width)

        return candidate_regions


