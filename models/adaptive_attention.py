import torch
import torch.nn as nn

class SpatialAttentionModule(nn.Module):

    def __init__(self, in_channels):

        super(SpatialAttentionModule, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)

        self.LeakyRelu = nn.LeakyReLU(0.01, inplace=True)

        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):

        print(x.shape)

        x_ = self.conv1(x)

        print(x_.shape)

        pooling1 = self.max_pool(x_)

        pooling2 = self.max_pool(x_)

        pool_cat = torch.mul(pooling1, pooling2)

        print(pool_cat.shape)

        conv_1 = self.conv1(x_)

        conv_2 = torch.mul(conv_1, pool_cat)

        print(conv_2.shape)

        out = torch.add(conv_2, x_)

        out = self.conv1(out)

        out = torch.add(x, out)

        out = self.LeakyRelu(out)

        out = self.conv1(out)

        out = self.Sigmoid(out)

        return out

class ChannelAttentionModule(nn.Module):

    def __init__(self, in_channels):
        super(ChannelAttentionModule, self).__init__()
        # 自适应最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 自适应平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #1*1卷积核
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        #激活函数
        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        max_pool = self.max_pool(x)

        avg_pool = self.avg_pool(x)

        pool_cat = torch.mul(max_pool, avg_pool)

        # 重塑
        pool_mul = pool_cat.view(pool_cat.size(0), pool_cat.size(1), 1, 1)

        # 扩展
        fc_out = self.conv1(pool_mul)
        # 激活函数
        fc_out = self.relu(fc_out)

        fc_out = self.conv1(fc_out)

        out = self.sigmoid(fc_out)

        return out


class AdaptiveAttentionFusionModule(nn.Module):

    def __init__(self, in_channels):
        super(AdaptiveAttentionFusionModule, self).__init__()

        self.SpatialAttentionModule = SpatialAttentionModule(in_channels)

        self.ChannelAttentionModule = ChannelAttentionModule(in_channels)

    def forward(self, x):
        x1 = self.SpatialAttentionModule(x)

        x2 = self.ChannelAttentionModule(x)

        out = torch.add(x1, x2)

        return out



