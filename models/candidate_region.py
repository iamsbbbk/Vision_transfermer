import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

class CandidateRegionGenerator(nn.Module):
    def __init__(self, in_channels, num_subclasses=5):
        super(CandidateRegionGenerator, self).__init__()
        self.num_subclasses = num_subclasses

        # Feature extraction layers
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        # Feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        features = self.bn3(self.conv3(x))

        # Flatten the spatial dimensions
        batch_size, channels, height, width = features.size()
        features_flat = features.view(batch_size, channels, -1).permute(0, 2, 1).contiguous()  # (batch_size, height * width, channels)

        # Generate candidate regions using KMeans clustering
        candidate_regions = []
        for i in range(batch_size):
            kmeans = KMeans(n_clusters=self.num_subclasses, random_state=0).fit(features_flat[i].cpu().detach().numpy())
            labels = kmeans.labels_
            candidate_region = torch.zeros_like(features_flat[i])
            for j in range(self.num_subclasses):
                candidate_region[labels == j] = torch.mean(features_flat[i][labels== j], dim=0)
            candidate_regions.append(candidate_region)
        candidate_regions = torch.stack(candidate_regions).permute(0, 2, 1).view(batch_size, channels, height, width)

        return candidate_regions