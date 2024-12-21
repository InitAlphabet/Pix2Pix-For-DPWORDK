import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    # 空间注意力
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv1(x_out)
        attention = self.sigmoid(attention)
        return x * attention


class ChannelAttention(nn.Module):
    # 通道注意力
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)
        avg_out = self.fc2(self.fc1(avg_pool.view(avg_pool.size(0), -1)))
        max_out = self.fc2(self.fc1(max_pool.view(max_pool.size(0), -1)))
        out = avg_out + max_out
        return x * self.sigmoid(out).view(out.size(0), out.size(1), 1, 1)


class DualAttention(nn.Module):
    # 综合空间注意力，通道注意力

    def __init__(self, in_channels):
        super(DualAttention, self).__init__()
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(in_channels)

    def forward(self, x):
        x = self.spatial_attention(x)
        x = self.channel_attention(x)
        return x


class SelfAttention(nn.Module):
    # 自注意力
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query_conv(x).view(batch_size, C, -1)
        key = self.key_conv(x).view(batch_size, C, -1)
        value = self.value_conv(x).view(batch_size, C, -1)
        attention_map = torch.bmm(query.transpose(1, 2), key)
        attention_map = torch.softmax(attention_map, dim=-1)
        out = torch.bmm(value, attention_map.transpose(1, 2))
        out = out.view(batch_size, C, H, W)
        return x + self.gamma * out
