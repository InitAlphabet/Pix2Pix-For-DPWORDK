import torch.nn as nn
import torch
from tool import ParameterError
from extra_networks import DualAttention


def weights_init(m):
    className = m.__class__.__name__
    if className.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif className.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def downsample(in_channels, out_channels, normalize=True):
    # 下采样用于逐步降低输入的分辨率，同时提取特征
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))  # 默认都是要归一化的
    layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)


def upsample(in_channels, out_channels, dropout=False):
    # 上采样层，逐步恢复图像的分辨率，同时结合跳跃连接传递的低层特征
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    ]
    # ConvTranspose2d：转置卷积操作，将尺寸扩大一倍
    # BatchNorm2d：归一化层
    # ReLU：激活函数

    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


def block(in_channels, out_channels, normalize=True, stride=2):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False)]
    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)


class ResnetBlock_Standard(nn.Module):
    # 残差模块，标砖版本
    def __init__(self, in_channels, out_channels):
        super(ResnetBlock_Standard, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)  # 通道数转换

    def forward(self, x):
        return self.block(x) + self.skip_connection(x)


class ResnetBlock_Simple(nn.Module):
    # 残差模块，简化版本
    def __init__(self, in_channels, out_channels, bottleneck_channels):
        super(ResnetBlock_Simple, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, padding=0, stride=1)
        )
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)  # 通道数转换

    def forward(self, x):
        return self.block(x) + self.skip_connection(x)


class Generator(nn.Module):
    def __init__(self, model='common'):
        super(Generator, self).__init__()
        """
        :param model: 'common','resnet','attention','all'
        :return:
        """
        model_parameters = ['common', 'resnet', 'attention', 'all']
        if model not in model_parameters:
            raise ParameterError("model must be ['common','resnet','attention','all']")
        if model == 'common':
            self.model = UNetGenerator()
        elif model == 'resnet':
            self.model = UNetGenerator1()
        elif model == 'attention':
            self.model = UNetGenerator2()
        else:
            self.model = UNetGenerator3()

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):

    # 标准版
    # 与传统判别器不同，PatchGAN 不是对整张图像进行真伪判别，而是对图像的局部区域（patch）进行真伪判断
    # 专注于局部特征，可以更有效地捕捉细节
    def __init__(self, model='common'):
        super(Discriminator, self).__init__()
        """
        :param model: 'common','resnet','attention','all'
        :return:
        """
        model_parameters = ['common', 'resnet', 'attention', 'all']
        if model not in model_parameters:
            raise ParameterError("model must be ['common','resnet','attention','all']")

        modules = [block(6, 64, normalize=False),
                   block(64, 128),
                   block(128, 256),
                   block(256, 512, stride=1)]

        if model == 'common':
            pass
        elif model == 'resnet':
            modules += [
                ResnetBlock_Standard(512, 512),
                ResnetBlock_Standard(512, 512),
                ResnetBlock_Standard(512, 512)
            ]
        elif model == 'attention':
            modules += [
                DualAttention(512)
            ]
        else:
            modules += [
                ResnetBlock_Standard(512, 512),
                ResnetBlock_Standard(512, 512),
                ResnetBlock_Standard(512, 512),
                DualAttention(512)
            ]
        modules.append(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*modules)

    def forward(self, x, y):
        # x为输入图像，y为目标图像(真实图像)
        xy = torch.cat([x, y], dim=1)  # [batch_size, 6, H, W]，拼接的目的是让判别器同时比较输入图像和目标图像，判断它们的相似性
        return self.model(xy)  # 最终输出为一个二维矩阵，每个值对应输入图像的一块区域的真伪评分


class UNetGenerator(nn.Module):
    # 基础版本
    # 生成器是基于unet的架构
    def __init__(self):
        super(UNetGenerator, self).__init__()
        # ----------------------下采样层
        self.down = nn.ModuleList([
            downsample(3, 64, normalize=False),  # 128
            downsample(64, 128),  # 64
            downsample(128, 256),  # 32
            downsample(256, 512),  # 16
            downsample(512, 512),  # 8
            downsample(512, 512),  # 4
            downsample(512, 512),  # 2
        ])
        self.up = nn.ModuleList([
            upsample(512, 512),  # 4
            upsample(1024, 512),  # 8
            upsample(1024, 512),  # 16
            upsample(1024, 256),  # 32
            upsample(512, 128),  # 64
            upsample(256, 64),  # 128
        ])

        self.final = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)  # 将最后的特征映射回 3 通道（RGB）的输出图像
        self.tanh = nn.Tanh()

    def forward(self, x):
        skips = []
        for layer in self.down:
            x = layer(x)
            skips.append(x.clone())  # 确保跳跃连接存储为张量
        # 输入经过每个下采样层，逐层提取特征并降低分辨率。保存跳跃连接特征 skips，用于后续解码阶段
        skips = skips[:-1][::-1]  # 跳过最后一层，倒序
        # 在解码阶段，将上采样的特征与跳跃连接的特征拼接在一起
        for i, layer in enumerate(self.up):
            x = layer(x)
            if i < len(skips):
                # 确保跳跃连接形状匹配
                if x.shape[2:] != skips[i].shape[2:]:
                    x = torch.nn.functional.interpolate(x, size=skips[i].shape[2:], mode='nearest')
                x = torch.cat((x, skips[i]), dim=1)

        x = self.final(x)
        return self.tanh(x)


class UNetGenerator1(nn.Module):
    # 残差强化版
    def __init__(self):
        super(UNetGenerator1, self).__init__()
        # ----------------------下采样层
        self.down = nn.ModuleList([
            downsample(3, 64, normalize=False),  # 128
            downsample(64, 128),  # 64
            downsample(128, 256),  # 32
            downsample(256, 512),  # 16
            downsample(512, 512),  # 8
            downsample(512, 512),  # 4
        ])
        self.resnets = nn.Sequential(
            ResnetBlock_Standard(512, 512),
            ResnetBlock_Standard(512, 512),
            ResnetBlock_Standard(512, 512),
            ResnetBlock_Standard(512, 512),
            ResnetBlock_Standard(512, 512),
            ResnetBlock_Standard(512, 512),
            ResnetBlock_Standard(512, 512),
            ResnetBlock_Standard(512, 512),
        )
        self.up = nn.ModuleList([
            upsample(512, 512),  # 8
            upsample(1024, 512),  # 16
            upsample(1024, 256),  # 32
            upsample(512, 128),  # 64
            upsample(256, 64),  # 128
        ])

        self.final = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)  # 将最后的特征映射回 3 通道（RGB）的输出图像
        self.tanh = nn.Tanh()

    def forward(self, x):
        skips = []
        for layer in self.down:
            x = layer(x)
            skips.append(x.clone())  # 确保跳跃连接存储为张量
        # 输入经过每个下采样层，逐层提取特征并降低分辨率。保存跳跃连接特征 skips，用于后续解码阶段
        skips = skips[:-1][::-1]  # 跳过最后一层，倒序
        x = self.resnets(x)
        # 在解码阶段，将上采样的特征与跳跃连接的特征拼接在一起
        for i, layer in enumerate(self.up):
            x = layer(x)
            if i < len(skips):
                # 确保跳跃连接形状匹配
                if x.shape[2:] != skips[i].shape[2:]:
                    x = torch.nn.functional.interpolate(x, size=skips[i].shape[2:], mode='nearest')
                x = torch.cat((x, skips[i]), dim=1)

        x = self.final(x)
        return self.tanh(x)


class UNetGenerator2(nn.Module):
    # 注意力强化版本
    def __init__(self):
        super(UNetGenerator2, self).__init__()
        # ----------------------下采样层
        self.down = nn.ModuleList([
            downsample(3, 64, normalize=False),  # 128
            downsample(64, 128),  # 64
            downsample(128, 256),  # 32
            downsample(256, 512),  # 16
            downsample(512, 512),  # 8
            downsample(512, 512),  # 4
        ])

        self.resnets = DualAttention(512)
        self.up = nn.ModuleList([
            upsample(512, 512),  # 8
            upsample(1024, 512),  # 16
            upsample(1024, 256),  # 32
            upsample(512, 128),  # 64
            upsample(256, 64),  # 128
        ])

        self.final = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)  # 将最后的特征映射回 3 通道（RGB）的输出图像
        self.tanh = nn.Tanh()

    def forward(self, x):
        skips = []
        for layer in self.down:
            x = layer(x)
            skips.append(x.clone())  # 确保跳跃连接存储为张量
        # 输入经过每个下采样层，逐层提取特征并降低分辨率。保存跳跃连接特征 skips，用于后续解码阶段
        skips = skips[:-1][::-1]  # 跳过最后一层，倒序
        x = self.resnets(x)
        # 在解码阶段，将上采样的特征与跳跃连接的特征拼接在一起
        for i, layer in enumerate(self.up):
            x = layer(x)
            if i < len(skips):
                # 确保跳跃连接形状匹配
                if x.shape[2:] != skips[i].shape[2:]:
                    x = torch.nn.functional.interpolate(x, size=skips[i].shape[2:], mode='nearest')
                x = torch.cat((x, skips[i]), dim=1)

        x = self.final(x)
        return self.tanh(x)


class UNetGenerator3(nn.Module):
    # 注意力加残差强化版
    def __init__(self):
        super(UNetGenerator3, self).__init__()
        # ----------------------下采样层
        self.down = nn.ModuleList([
            downsample(3, 64, normalize=False),  # 128
            downsample(64, 128),  # 64
            downsample(128, 256),  # 32
            downsample(256, 512),  # 16
            downsample(512, 512),  # 8
            downsample(512, 512),  # 4
        ])

        self.resnets = nn.Sequential(
            ResnetBlock_Standard(512, 512),
            ResnetBlock_Standard(512, 512),
            ResnetBlock_Standard(512, 512),
            DualAttention(512))
        self.up = nn.ModuleList([
            upsample(512, 512),  # 8
            upsample(1024, 512),  # 16
            upsample(1024, 256),  # 32
            upsample(512, 128),  # 64
            upsample(256, 64),  # 128
        ])

        self.final = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)  # 将最后的特征映射回 3 通道（RGB）的输出图像
        self.tanh = nn.Tanh()

    def forward(self, x):
        skips = []
        for layer in self.down:
            x = layer(x)
            skips.append(x.clone())  # 确保跳跃连接存储为张量
        # 输入经过每个下采样层，逐层提取特征并降低分辨率。保存跳跃连接特征 skips，用于后续解码阶段
        skips = skips[:-1][::-1]  # 跳过最后一层，倒序
        x = self.resnets(x)
        # 在解码阶段，将上采样的特征与跳跃连接的特征拼接在一起
        for i, layer in enumerate(self.up):
            x = layer(x)
            if i < len(skips):
                # 确保跳跃连接形状匹配
                if x.shape[2:] != skips[i].shape[2:]:
                    x = torch.nn.functional.interpolate(x, size=skips[i].shape[2:], mode='nearest')
                x = torch.cat((x, skips[i]), dim=1)

        x = self.final(x)
        return self.tanh(x)
