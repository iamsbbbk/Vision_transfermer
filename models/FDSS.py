import torch
import torch.nn as nn

class FDSS_nbt(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FDSS_nbt, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.Channel_split = in_channels // 2

        self.dwconv3_1 = nn.Conv2d(self.Channel_split, self.Channel_split, (3, 1), groups=self.Channel_split, padding=(1, 0))
        self.dwconv1_3 = nn.Conv2d(self.Channel_split, self.Channel_split, (1, 3), groups=self.Channel_split, padding=(0, 1))

        self.ddwconv3_1 = nn.Conv2d(self.Channel_split, self.Channel_split, (3, 1), groups=self.Channel_split, padding=(1, 0))
        self.ddwconv1_3 = nn.Conv2d(self.Channel_split, self.Channel_split, (1, 3), groups=self.Channel_split, padding=(0, 1))

    def Channel_shuffle(self, x, groups):
        batch_size, channels, height, width = x.shape
        assert channels % groups == 0
        group_channels = channels // groups

        x = x.view(batch_size, groups, group_channels, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, channels, height, width)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x_ = self.conv2(x)
        channels_half = x_.size(1) // 2

        x_1 = x_[:, :channels_half, :, :]
        x_2 = x_[:, channels_half:, :, :]

        x_1 = self.dwconv3_1(x_1)
        x_1 = self.dwconv1_3(x_1)

        x_2 = self.ddwconv3_1(x_2)
        x_2 = self.ddwconv1_3(x_2)

        x__ = torch.cat((x_1, x_2), dim=1)

        out = x + x__  # Residual connection
        out = self.Channel_shuffle(out, groups=2)

        return out

if __name__ == '__main__':
    in_channels = 64
    out_channels = 64
    batch_size = 1
    height = 32
    width = 32

    tensor = torch.randn(batch_size, in_channels, height, width)
    model = FDSS_nbt(in_channels, out_channels)
    out = model(tensor)
    print(out.shape)