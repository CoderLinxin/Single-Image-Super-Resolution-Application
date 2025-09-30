import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import numpy as np
from huggingface_hub import PyTorchModelHubMixin


class dwconv(nn.Module):
    def __init__(self, hidden_features):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultipleSizeConvExtract(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        self.out_channels = out_channels

        # 来自 Single image super-resolution using multi-scale feature enhancement attention residual net
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, 3 // 2)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 5, 1, 5 // 2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, 7, 1, 7 // 2)
        self.conv9 = nn.Conv2d(in_channels, out_channels, 9, 1, 9 // 2)
        self.conv_x = nn.Conv2d(3, out_channels, 1, 1, 0)

        # self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.norm = nn.LayerNorm(out_channels)

        #  用于调整特征的通道维度
        self.conv_last = nn.Conv2d(4 * out_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        """
        x: shape = (b,c,h,w)
        """

        x3 = self.conv3(x)  # (b,out_channels,h,w)
        x5 = self.conv5(x)  # (b,out_channels,h,w)
        x7 = self.conv7(x)  # (b,out_channels,h,w)
        x9 = self.conv9(x)  # (b,out_channels,h,w)
        x = self.conv_x(x)

        # B, C, H, W = x.size()

        # x = x.view(B, self.out_channels, -1)  # (b,out_channels,h*w)

        # 计算各个尺寸卷积核的特征图结果与y的点积得到注意力权重(不要对点积进行 sum, 效果会变差)
        x3_attention = torch.sigmoid(x * x3)  # (b,out_channels,h,w)
        x5_attention = torch.sigmoid(x * x5)  # (b,out_channels,h,w)
        x7_attention = torch.sigmoid(x * x7)  # (b,out_channels,h,w)
        x9_attention = torch.sigmoid(x * x9)  # (b,out_channels,h,w)

        # 加权(加了残差性能更好)
        x3 = x3 * x3_attention + x3  # (b,out_channels,h,w)
        x5 = x5 * x5_attention + x5  # (b,out_channels,h,w)
        x7 = x7 * x7_attention + x7  # (b,out_channels,h,w)
        x9 = x9 * x9_attention + x9  # (b,out_channels,h,w)

        # scale = H * W
        # x3 = (((x3.view(B, self.out_channels, -1).permute(0, 2, 1) @ x) / scale) @ x3.view(B, self.out_channels, -1).permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W)
        # x5 = (((x5.view(B, self.out_channels, -1).permute(0, 2, 1) @ x) / scale) @ x5.view(B, self.out_channels, -1).permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W)
        # x7 = (((x7.view(B, self.out_channels, -1).permute(0, 2, 1) @ x) / scale) @ x7.view(B, self.out_channels, -1).permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W)
        # x9 = (((x9.view(B, self.out_channels, -1).permute(0, 2, 1) @ x) / scale) @ x9.view(B, self.out_channels, -1).permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W)

        return self.conv_last(torch.cat((x3, x5, x7, x9), dim=1))


# 计算出 x 的联合注意力权重(联合了 C、H、W的注意力计算)
class UnionAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # self.relu = torch.relu
        self.conv1 = nn.Conv2d(2, 1, 3, 1, 3 // 2)
        self.conv2 = nn.Conv2d(2, 1, 3, 1, 3 // 2)
        self.conv3 = nn.Conv2d(2, 1, 3, 1, 3 // 2)
        self.conv_last = nn.Conv2d(in_channels, in_channels, 3, 1, 3 // 2)

    def forward(self, x: torch.Tensor):
        """
        x: shape = (b,c,h,w)
        """
        b, c, h, w = x.size()

        # 对 C 维度计算最大池化、平均池化
        avg_out_c = torch.mean(x, dim=1, keepdim=True)  # (b,1,h,w)
        max_out_c = torch.max(x, dim=1, keepdim=True)[0]  # (b,1,h,w)
        c_attention = self.conv1(torch.cat((avg_out_c, max_out_c), dim=1))  # (b,1,h,w)
        # 对 H 维度计算最大池化、平均池化
        avg_out_h = torch.mean(x, dim=2, keepdim=True).reshape(b, 1, c, w)  # (b,c,1,w)->(b,1,c,w)
        max_out_h = torch.max(x, dim=2, keepdim=True)[0].reshape(b, 1, c, w)  # (b,c,1,w)->(b,1,c,w)
        h_attention = self.conv2(torch.cat((avg_out_h, max_out_h), dim=1)).reshape(b, c, 1, w)  # (b,c,1,w)
        # 对 W 维度计算最大池化、平均池化
        avg_out_w = torch.mean(x, dim=3, keepdim=True).reshape(b, c, 1, h).reshape(b, 1, c, h)  # (b,c,h,1)->(b,c,1,h)->(b,1,c,h)
        max_out_w = torch.max(x, dim=3, keepdim=True)[0].reshape(b, c, 1, h).reshape(b, 1, c, h)  # (b,c,h,1)->(b,c,1,h)->(b,1,c,h)
        w_attention = self.conv3(torch.cat((avg_out_w, max_out_w), dim=1)).reshape(b, c, 1, h).reshape(b, c, h, 1)  # (b,c,h,1)

        # 计算联合了 C、H、W 的注意力权重
        return self.conv_last(c_attention + w_attention + h_attention)


class Fusion(nn.Module):
    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.union_attention1 = UnionAttention(out_channels)
        self.union_attention2 = UnionAttention(out_channels)
        self.union_attention3 = UnionAttention(out_channels)
        # self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 3 // 2)
        self.sigmoid = torch.sigmoid

    def forward(self, shallow, deep):
        """
        shallow: shape = (b,c,h,w) 浅层特征
        deep = shape = (b,c,h,w) 深层特征
        """
        # 计算注意力权重
        shallow_attention = self.union_attention1(shallow)
        attention = self.sigmoid(self.union_attention2(shallow + deep))
        deep_attention = self.union_attention3(deep)

        shallow_attention = shallow_attention * attention
        deep_attention = deep_attention * (1 - attention)

        # 注意力加权
        shallow_weighting = shallow * self.sigmoid(shallow_attention)
        deep_weighting = deep * self.sigmoid(deep_attention)

        return shallow_weighting + deep_weighting


class DFE(nn.Module):
    """ Dual Feature Extraction
    Args:
        in_features (int): Number of input channels.
        out_features (int): Number of output channels.
    """

    def __init__(self,
                 in_features,  # 输入特征维度
                 out_features  # 输出特征维度
                 ):
        super().__init__()

        self.out_features = out_features

        # 卷积操作用于提取空间信息
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, in_features // 5, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_features // 5, in_features // 5, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_features // 5, out_features, 1, 1, 0)
        )

        # 非线性变换(1x1卷积)用于提取通道信息
        self.linear = nn.Conv2d(in_features, out_features, 1, 1, 0)

    def forward(self, x, x_size):
        """
        x.size() = (B, H*W, C)
        """
        B, L, C = x.shape
        H, W = x_size
        # (B, H*W, C) -> (B,C,H*W) -> (B,C,H,W)
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        # 将 1x1 卷积结果和 3x3 卷积结果想乘
        x = self.conv(x) * self.linear(x)  # (B,C,H,W)
        # (B,C,H,W) -> (B,C,H*W) -> (B, H*W, C)
        x = x.view(B, -1, H * W).permute(0, 2, 1).contiguous()

        return x  # (B, H*W, C)


class Mlp(nn.Module):
    """ MLP-based Feed-Forward Network
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (tuple): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    # num_windows * window_size[0] * window_size[1] = H*W
    # (H*W*B)/(H*W) = B
    B = int(windows.shape[0] * (window_size[0] * window_size[1]) / (H * W))
    # (b,h // window_size,w // window_size,window_size,window_size,c)
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    # (b,h // window_size,w // window_size,window_size,window_size,c)
    # -> (b,h // window_size,window_size,w // window_size,window_size,c)
    # -> (b,h,w,c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x  # (b,h,w,c)


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of heads for spatial self-correlation.
        residual (bool):  If True, use residual strage to connect conv.
    """

    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)  # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


# 空间通道注意力模块(加了这个模块,时间复杂度变高了不少,大约变高了 15%)
class SpatialChannelAttention(nn.Module):
    def __init__(self, dim):
        """
        dim: 输入特征图维度
        """
        super().__init__()

        self.linear1 = nn.Conv2d(1, dim, 3, 1, 1)
        self.linear2 = nn.Conv2d(1, dim, 3, 1, 1)
        # self.linear_last = nn.Conv2d(dim, dim, 1, 1, 0)

        # 下面 linear 层是参数量多少的关键(为了节省参数量,做了一个通道压缩)
        self.linear1_first = nn.Linear(dim, dim // 10)
        self.linear1_second = nn.Linear(dim // 10, dim)
        self.linear2_first = nn.Linear(dim, dim // 10)
        self.linear2_second = nn.Linear(dim // 10, dim)
        # 池化操作非常费时间
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        """
        x.size() = (b,c,h,w)
        """

        channel_avg = torch.mean(x, dim=1, keepdim=True)  # (b,1,h,w)
        channel_max = torch.max(x, dim=1, keepdim=True)[0]  # (b,1,h,w)
        channel_attn1 = self.relu(self.linear1(channel_avg))  # (b,c,h,w)
        channel_attn2 = self.relu(self.linear2(channel_max))  # (b,c,h,w)

        spatial_avg = self.avg(x)  # (b,c,1,1)
        spatial_max = self.max(x)  # (b,c,1,1)
        spatial_attn1 = self.linear1_second(self.linear1_first(
            spatial_avg.permute(0, 2, 3, 1)
        )).permute(0, 3, 1, 2)  # (b,c,1,1)
        spatial_attn2 = self.linear2_second(self.linear2_first(
            spatial_max.permute(0, 2, 3, 1)
        )).permute(0, 3, 1, 2)  # (b,c,1,1)

        attn = (channel_attn1 * spatial_attn1 + channel_attn2 * spatial_attn2) / 2.

        return attn + x  # (b,c,h,w)


class SCC(nn.Module):
    """ Spatial-Channel Correlation.
    Args:
        dim (int): Number of input channels.
        base_win_size (tuple[int]): The height and width of the base window.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of heads for spatial self-correlation.
        value_drop (float, optional): Dropout ratio of value. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 is_channel_spatial_attn,
                 dim,  # 输入特征的维度 embed_dim
                 base_win_size,  # 基础窗口大小
                 window_size,  # 实际使用的窗口大小
                 num_heads,  # 注意力头的个数
                 value_drop=0.,
                 proj_drop=0.
                 ):
        super().__init__()
        # parameters
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        # feature projection
        # DFE 由两个分支组成,分别用于提取通道信息和空间信息
        # self.qkv = DFE(dim, dim)
        # 卷积操作用于提取空间信息(这里如果直接使用3x3卷积的话参数量较大)
        # self.conv = nn.Sequential(nn.Conv2d(dim, dim // 7, 1, 1, 0),
        #                           nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                           nn.Conv2d(dim // 7, dim // 7, 3, 1, 1),
        #                           nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                           nn.Conv2d(dim // 7, dim, 1, 1, 0))
        # # 非线性变换(1x1卷积)用于提取通道信息
        # self.linear = nn.Conv2d(dim, dim, 1, 1, 0)
        self.qkv = nn.Identity()
        # 空间通道注意力
        if is_channel_spatial_attn:
            self.qkv = SpatialChannelAttention(dim)
            print('深层特征提取模块中qkv的计算使用空间通道注意力模块')
        else:
            print('深层特征提取模块中qkv的计算不使用空间通道注意力模块')

        self.proj = nn.Linear(dim, dim)

        # dropout
        self.value_drop = nn.Dropout(value_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # base window size
        # 获取 RHTB 中使用的一系列窗口中的最小值(论文中设置的层级窗口只有一个窗口比基础窗口小,故此代码获取的就是层级窗口中的最小窗口)
        min_h = min(self.window_size[0], base_win_size[0])
        min_w = min(self.window_size[1], base_win_size[1])
        self.base_win_size = (min_h, min_w)

        # normalization factor and spatial linear layer for S-SC
        head_dim = dim // (2 * num_heads)
        self.scale = head_dim
        self.spatial_linear = nn.Linear((self.window_size[0] * self.window_size[1]) // (self.base_win_size[0] * self.base_win_size[1]), 1)

        # self.head_dim_half1 = (self.window_size[0] * self.window_size[1]) // 2
        # self.head_dim_half2 = self.window_size[0] * self.window_size[1] - self.head_dim_half1
        # self.k_generate1 = nn.Linear(head_dim, self.head_dim_half1)
        # self.k_generate2 = nn.Linear(head_dim, self.head_dim_half2)
        self.k_generate1 = nn.Linear(head_dim, head_dim)
        self.k_generate2 = nn.Linear(head_dim, head_dim)

        # define a parameter table of relative position bias
        self.H_sp, self.W_sp = self.window_size
        self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

    def spatial_linear_projection(self, x):
        """
        x.size()=(num_windows*B, num_heads, window_size*window_size, C//(2*num_heads))
        """
        B, num_h, L, C = x.shape
        # 当前使用的窗口
        H, W = self.window_size
        # 获取层级窗口中的最小窗口
        map_H, map_W = self.base_win_size

        # 如果当前使用的窗口大小比基础窗口大小大,那么将其映射为基础窗口大小
        # 相当于池化操作,对感受野比较大的深层特征使用池化操作提取特征
        # (num_windows*B, num_heads, window_size*window_size, C//(2*num_heads))
        # -> (num_windows*B, num_heads, map_H, window_size//map_H, map_W, window_size//map_W, C//(2*num_heads))
        # -> (num_windows*B, num_heads, map_H, map_W, C//(2*num_heads), window_size//map_H, window_size//map_W)
        # -> (num_windows*B, num_heads, map_H*map_W, C//(2*num_heads), (window_size//map_H)*(window_size//map_W))
        x = x.view(B, num_h, map_H, H // map_H, map_W, W // map_W, C).permute(0, 1, 2, 4, 6, 3, 5).contiguous().view(B, num_h, map_H * map_W, C, -1)
        # 直接将最后一个维度吸收为 1
        # (num_windows*B, num_heads, map_H*map_W, C//(2*num_heads), 1)
        # -> (num_windows*B, num_heads, map_H*map_W, C//(2*num_heads))
        x = self.spatial_linear(x).view(B, num_h, map_H * map_W, C)
        return x

    def spatial_self_correlation(self, q, k, v):
        """
        q.size()=k.size()=v.size() = (num_windows*B, num_heads, window_size*window_size, C//(2*num_heads))
        """
        B, num_head, L, C = q.shape

        # spatial projection
        # S-Linear(用于概括空间信息),相当于池化操作,如果当前窗口比基础窗口大,那么映射为基础窗口大小,多余的像素通过线性层吸收掉
        # 相当于压缩了词向量的数量
        # (num_windows*B, num_heads, window_size_map*window_size_map, C//(2*num_heads))
        # 注意这一步如果省略的话,那么计算量会飙升
        v = self.spatial_linear_projection(v)
        k = self.spatial_linear_projection(k)

        # compute correlation map
        # 计算 qk 的相关性矩阵
        # (num_windows*B, num_heads, window_size*window_size, window_size_map*window_size_map)
        corr_map = (q @ k.transpose(-2, -1)) / self.scale

        # add relative position bias
        # generate mother-set
        position_bias_h = torch.arange(1 - self.H_sp, self.H_sp, device=v.device)
        position_bias_w = torch.arange(1 - self.W_sp, self.W_sp, device=v.device)
        biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
        rpe_biases = biases.flatten(1).transpose(0, 1).contiguous().float()
        pos = self.pos(rpe_biases)

        # select position bias
        coords_h = torch.arange(self.H_sp, device=v.device)
        coords_w = torch.arange(self.W_sp, device=v.device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.H_sp - 1
        relative_coords[:, :, 1] += self.W_sp - 1
        relative_coords[:, :, 0] *= 2 * self.W_sp - 1
        relative_position_index = relative_coords.sum(-1)
        relative_position_bias = pos[relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.base_win_size[0], self.window_size[0] // self.base_win_size[0], self.base_win_size[1], self.window_size[1] // self.base_win_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(0, 1, 3, 5, 2, 4).contiguous().view(
            self.window_size[0] * self.window_size[1], self.base_win_size[0] * self.base_win_size[1], self.num_heads, -1).mean(-1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # 添加相对位置编码
        corr_map = corr_map + relative_position_bias.unsqueeze(0)

        # transformation
        v_drop = self.value_drop(v)
        # 相关性矩阵与 v 进行矩阵乘法
        # (num_windows*B, num_heads, window_size*window_size, C//(2*num_heads))
        # ->(num_windows*B, window_size*window_size, num_heads, C//(2*num_heads))
        # ->(num_windows*B, window_size*window_size, C//2)
        x = (corr_map @ v_drop).permute(0, 2, 1, 3).contiguous().view(B, L, -1)

        return x  # (num_windows*B, window_size*window_size, C//2)

    def channel_self_correlation(self, q, k, v):
        """
        q.size() = k.size() = v.size() = (num_windows*B, num_heads, window_size*window_size, C//(2*num_heads))
        """
        B, num_head, L, C = q.shape

        # 单头策略
        # (num_windows*B, window_size*window_size, num_heads, C//(2*num_heads))
        # -> (num_windows*B, window_size*window_size, C//2)
        q = q.permute(0, 2, 1, 3).contiguous().view(B, L, num_head * C)
        k = k.permute(0, 2, 1, 3).contiguous().view(B, L, num_head * C)
        v = v.permute(0, 2, 1, 3).contiguous().view(B, L, num_head * C)

        # compute correlation map
        # 计算通道相关性矩阵
        # (num_windows*B, C//2, C//2)
        corr_map = (q.transpose(-2, -1) @ k) / L

        # transformation
        v_drop = self.value_drop(v)
        # 将通道相关性矩阵与 v 进行矩阵乘法
        # (num_windows*B, C//2, window_size*window_size)
        # -> (num_windows*B, window_size*window_size, C//2)
        x = (corr_map @ v_drop.transpose(-2, -1)).permute(0, 2, 1).contiguous().view(B, L, -1)  # 这个 view 操作是多余的

        return x  # (num_windows*B, window_size*window_size, C//2)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        xB, xH, xW, xC = x.shape
        # 计算通道空间特征得到 q、v
        # (B, H*W, C) -> (B,H,W,C)
        # qkv = self.qkv(x.view(xB, -1, xC), (xH, xW)).view(xB, xH, xW, xC)

        # 计算通道空间特征得到 qkv
        # (B, H*W, C) -> (B,C,H*W) -> (B,C,H,W)
        h = x.view(xB, -1, xC).permute(0, 2, 1).contiguous().view(xB, xC, xH, xW)
        # (B,C,H,W) -> (B,C,H*W) -> (B, H*W, C) -> (B,H,W,C)
        # qkv = (self.conv(h) * self.linear(h)).view(xB, -1, xH * xW).permute(0, 2, 1).contiguous().view(xB, xH, xW, xC)
        qkv = self.qkv(h).view(xB, -1, xH * xW).permute(0, 2, 1).contiguous().view(xB, xH, xW, xC)

        # window partition
        # 由于需要在窗口中计算注意力,所以需要先进行窗口划分
        qkv = window_partition(qkv, self.window_size)  # (num_windows*B,window_size,window_size,C)
        qkv = qkv.view(-1, self.window_size[0] * self.window_size[1], xC)  # (num_windows*B,window_size*window_size,C)

        # qkv splitting
        B, L, C = qkv.shape
        # 在通道上划分 qkv
        # (num_windows*B, window_size*window_size, 2, num_heads, C//(2*num_heads))
        # -> (2, num_windows*B, num_heads, window_size*window_size, C//(2*num_heads))
        qkv = qkv.view(B, L, 2, self.num_heads, C // (2 * self.num_heads)).permute(2, 0, 3, 1, 4).contiguous()
        q, v = qkv[0], qkv[1]  # q.size() = k.size() = v.size() = (num_windows*B, num_heads, window_size*window_size, C//(2*num_heads))
        # k = (self.k_generate(q) + self.k_generate(v)) / 2.
        k = (self.k_generate1(q) + self.k_generate2(v)) / 2.
        # k = torch.cat((self.k_generate1(q), self.k_generate2(v)), dim=-1) 不应该操作最后一维(词向量维度),而是要操作 window_size*window_size(词向量个数)
        # k = self.k_generate(torch.cat((q[:, :, :, 0:self.head_dim_half1], v[:, :, :, 0:self.head_dim_half2]), dim=-1)) 不应该操作最后一维(词向量维度),而是要操作 window_size*window_size(词向量个数)
        # k1 = self.k_generate1(torch.cat((q[:, :, 0:self.head_dim_half1, :], v[:, :, 0:self.head_dim_half2, :]), dim=2))
        # k2 = self.k_generate2(torch.cat((q[:, :, self.head_dim_half1 - 1:-1, :], v[:, :, self.head_dim_half2 - 1:-1, :]), dim=2))

        # spatial self-correlation (S-SC)
        x_spatial = self.spatial_self_correlation(q, k, v)  # (num_windows*B, window_size*window_size, C//2)
        x_spatial = x_spatial.view(-1, self.window_size[0], self.window_size[1], C // 2)  # (num_windows*B, window_size, window_size, C//2)
        x_spatial = window_reverse(x_spatial, (self.window_size[0], self.window_size[1]), xH, xW)  # (B,H,W,C//2)

        # channel self-correlation (C-SC)
        x_channel = self.channel_self_correlation(q, k, v)  # (num_windows*B, window_size*window_size, C//2)
        x_channel = x_channel.view(-1, self.window_size[0], self.window_size[1], C // 2)  # (num_windows*B, window_size, window_size, C//2)
        x_channel = window_reverse(x_channel, (self.window_size[0], self.window_size[1]), xH, xW)  # (B,H,W,C//2)

        # (num_windows*B, num_heads, window_size*window_size, C//(3*num_heads))
        # -> (num_windows*B, window_size*window_size, num_heads, C//(3*num_heads))
        # -> (num_windows*B, window_size*window_size, C//3)
        # k = k.permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        # k = k.view(-1, self.window_size[0], self.window_size[1], C // 3)  # (num_windows*B, window_size, window_size, C//3)
        # k = window_reverse(k, (self.window_size[0], self.window_size[1]), xH, xW)  # (B,H,W,C//3)

        # spatial-channel information fusion,将 k 也 cat 进来
        x = torch.cat([x_spatial, x_channel], -1)  # (B,H,W,C)
        x = self.proj_drop(self.proj(x))  # (B,H,W,C)

        return x  # (B,H,W,C)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class HierarchicalTransformerBlock(nn.Module):
    """ Hierarchical Transformer Block.
        STL

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of heads for spatial self-correlation.
        base_win_size (tuple[int]): The height and width of the base window.
        window_size (tuple[int]): The height and width of the window.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        value_drop (float, optional): Dropout ratio of value. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 is_channel_spatial_attn,
                 dim,  # 输入特征的维度 embed_dim
                 # 输入特征图的分辨率 (img_size[0]/patch_size[0],img_size[1]/patch_size[1]), swinir 配置中的 patch_size = 1
                 # 这里 input_resolution = (img_size[0],img_size[1])
                 input_resolution,
                 num_heads,  # 注意力头个数
                 base_win_size,  # 基础窗口大小
                 window_size,  # 当前使用的窗口大小
                 mlp_ratio=4.,
                 drop=0.,
                 value_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        # check window size
        # 当前使用的窗口大小比基础窗口大小大:确保当前使用的窗口大小是基础窗口大小的整数倍
        if (window_size[0] > base_win_size[0]) and (window_size[1] > base_win_size[1]):
            assert window_size[0] % base_win_size[0] == 0, "please ensure the window size is smaller than or divisible by the base window size"
            assert window_size[1] % base_win_size[1] == 0, "please ensure the window size is smaller than or divisible by the base window size"

        self.norm1 = norm_layer(dim)
        # hitsir 最主要的创新点
        self.correlation = SCC(
            is_channel_spatial_attn,
            dim, base_win_size=base_win_size, window_size=self.window_size, num_heads=num_heads,
            value_drop=value_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def check_image_size(self, x, win_size):
        """
        x.size() = (b,h,w,c)
        """
        x = x.permute(0, 3, 1, 2).contiguous()  # (b,c,h,w)
        _, _, h, w = x.size()
        mod_pad_h = (win_size[0] - h % win_size[0]) % win_size[0]
        mod_pad_w = (win_size[1] - w % win_size[1]) % win_size[1]
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')  # (b,c,h_pad,w_pad)
        x = x.permute(0, 2, 3, 1).contiguous()  # (b,h_pad,w_pad,c)
        return x

    def forward(self, x, x_size, win_size):
        """
             x.size() = (b,h*w,c), h、w 不需要是 window_size 的整数倍
             x_size 是 x 的高度和宽度(h,w)
             win_size 为当前使用的窗口大小(其实可以不用传)
        """
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = x.view(B, H, W, C)  # (b,h,w,c)

        # 将 x padding(如果需要的话) 为 window_size 的整数倍
        x = self.check_image_size(x, win_size)  # (b,h_pad,w_pad,c)
        _, H_pad, W_pad, _ = x.shape  # shape after padding

        # 计算空间通道相关性
        x = self.correlation(x)  # (b,h_pad,w_pad,c)

        # unpad
        x = x[:, :H, :W, :].contiguous()  # (b,h,w,c)

        # norm
        x = x.view(B, H * W, C)  # (b,h*w,c)
        x = self.norm1(x)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.norm2(self.mlp(x, x_size)))

        return x  # (b,h*w,c)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Module):
    """ A basic Hierarchical Transformer layer for one stage.
        对应了 swin transformer 的一个阶段,包含了(depth)(可以不是偶数)个 swin transformer block,这些 block 使用的是层级窗口

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of heads for spatial self-correlation.
        base_win_size (tuple[int]): The height and width of the base window.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        value_drop (float, optional): Dropout ratio of value. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        hier_win_ratios (list): hierarchical window ratios for a transformer block. Default: [0.5,1,2,4,6,8].
    """

    def __init__(self,
                 is_channel_spatial_attn,
                 dim,  # 输入特征维度 embed_dim
                 # 输入特征图的分辨率 (img_size[0]/patch_size[0],img_size[1]/patch_size[1]), swinir 配置中的 patch_size = 1
                 # 这里 input_resolution = (img_size[0],img_size[1])
                 input_resolution,
                 depth,  # swing transformer block 的个数
                 num_heads,  # 多头注意力的头数
                 base_win_size,  # 基础窗口大小(8,8)
                 mlp_ratio=4.,
                 drop=0.,
                 value_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 # 是否在 stage 末尾进行下采样操作(swin transformer 中对应了 patch merging), swinir 配置中为 None,即不进行下采样操作
                 downsample=None,
                 use_checkpoint=False,
                 hier_win_ratios=[0.5, 1, 2, 4, 6, 8]  # 层级窗口设置
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # 获取层级窗口
        self.win_hs = [int(base_win_size[0] * ratio) for ratio in hier_win_ratios]
        self.win_ws = [int(base_win_size[1] * ratio) for ratio in hier_win_ratios]

        # build blocks
        self.blocks = nn.ModuleList([
            HierarchicalTransformerBlock(
                is_channel_spatial_attn=is_channel_spatial_attn,
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads,
                base_win_size=base_win_size,
                window_size=(self.win_hs[i], self.win_ws[i]),
                mlp_ratio=mlp_ratio,
                drop=drop, value_drop=value_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer(swin ir 中不使用 patch merging)
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        """
        x.size() = (b,h*w,c),h和w可以不是window_size的整数倍
        x_size 是 x 的高度和宽度(h,w)
        """
        i = 0
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size, (self.win_hs[i], self.win_ws[i]))
            else:
                # block 内部一开始会将 x padding 为 window_size 整数倍
                # 传入 (self.win_hs[i], self.win_ws[i]) 指定窗口大小(主要是为了根据窗口大小对 x padding 为窗口整数倍)
                x = blk(x, x_size, (self.win_hs[i], self.win_ws[i]))  # (b,h*w,c)
            i = i + 1  # (b,h*w,c)

        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class RHTB(nn.Module):
    """Residual Hierarchical Transformer Block (RHTB).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of heads for spatial self-correlation.
        base_win_size (tuple[int]): The height and width of the base window.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        value_drop (float, optional): Dropout ratio of value. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
        hier_win_ratios (list): hierarchical window ratios for a transformer block. Default: [0.5,1,2,4,6,8].
    """

    def __init__(self,
                 is_channel_spatial_attn,  # qkv 计算是否使用通道空间注意力
                 dim,  # 输入特征的维度 embed_dim
                 # 输入特征图的分辨率 (img_size[0]/patch_size[0],img_size[1]/patch_size[1]), swinir 配置中的 patch_size = 1
                 # 这里 input_resolution = (img_size[0],img_size[1])
                 input_resolution,
                 depth,  # STL 的个数
                 num_heads,  # 注意力头的个数
                 base_win_size,  # 基础窗口大小
                 mlp_ratio=4.,  # 多层感知机隐藏层的维度和嵌入层的比.
                 drop=0.,  # 随机神经元丢弃率
                 value_drop=0.,
                 drop_path=0.,  # 深度随机丢弃率
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,  # swinir 默认配置为 (64,64)
                 patch_size=4,  # swinir 默认配置为 1
                 resi_connection='1conv',
                 hier_win_ratios=[0.5, 1, 2, 4, 6, 8]  # RHTB 中 STL 使用的层级窗口
                 ):
        super(RHTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        # depth 个 STL
        self.residual_group = BasicLayer(
            is_channel_spatial_attn=is_channel_spatial_attn,
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            base_win_size=base_win_size,
            mlp_ratio=mlp_ratio,
            drop=drop, value_drop=value_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            hier_win_ratios=hier_win_ratios)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        """
          x.size() = (b,h*w,c),h和w需要是window_size的整数倍
          x_size 是 x 的高度和宽度(h,w)
        """
        # (b,h*w,c) -> (b,c,h,w) -> (b,h*w,c)
        return self.patch_embed(self.conv(self.patch_unembed(
            self.residual_group(x, x_size), x_size)  # (b,h*w,c)
        )) + x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
        将 2 维图像转变成 1 维 patch embeddings(原始 2 维图像 (特征图的一个像素点对应的所有 channel(这里并没有分块)) 转变为 1 维的 patch embeddings)

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 img_size=224,  # 输入图像的大小(swinir 配置中是 64)(无实际用处)
                 patch_size=4,  # Patch token 的大小(swinir 配置中是 1)(无实际用处)
                 in_chans=3,  # 输入图像的通道数(swinir 配置中是 embed_dim)
                 embed_dim=96,  # 线性 projection 输出的通道数(swinir 配置中是 embed_dim)
                 norm_layer=None  # 归一化层
                 ):
        super().__init__()
        img_size = to_2tuple(img_size)  # (img_size,img_size)
        patch_size = to_2tuple(patch_size)  # (patch_size,patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # 原始图片编码成一个个 patch 后特征图的分辨率 img_size/patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # patch 的总个数

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        x.size() = (b,c,h,w)
        """
        # 由于 swinir 中 patch_size 为 1, 故这里直接将输入 x 重塑为目标形状
        x = x.flatten(2).transpose(1, 2)  # [b,c,h,w] -> [b,c,h*w] -> [b,h*w,c] = (b,词向量个数,词向量维度)
        if self.norm is not None:
            x = self.norm(x)
        return x  # [b,h*w,c]


class PatchUnEmbed(nn.Module):
    r""" 将 1 维 patch embeddings 转变为 2 维图像。
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 img_size=224,  # 输入图像大小,无实际用处
                 patch_size=4,  # patch 大小,无实际用处
                 in_chans=3,  # 输入图像的通道数(swinir 配置中是 embed_dim)
                 embed_dim=96,  # 线性 projection 输出的通道数(swinir 配置中是 embed_dim)
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        """
        x.size() = (b,h*w,c)
        """
        B, HW, C = x.shape
        # (b,h*w,c) -> (b,c,h*w) -> (b,c,x_size[0],x_size[1])
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


class HiT_SIR(nn.Module, PyTorchModelHubMixin):
    """ HiT-SIR network.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Transformer block.
        num_heads (tuple(int)): Number of heads for spatial self-correlation in different layers.
        base_win_size (tuple[int]): The height and width of the base window.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        value_drop_rate (float): Dropout ratio of value. Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale (int): Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range (float): Image range. 1. or 255.
        upsampler (str): The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection (str): The convolutional block before residual connection. '1conv'/'3conv'
        hier_win_ratios (list): hierarchical window ratios for a transformer block. Default: [0.5,1,2,4,6,8].
    """

    def __init__(
            self,
            is_mult_size_conv_feat_extract: bool,
            is_channel_spatial_attn: bool,
            is_fusion: bool,
            # 输入图像的大小(在swinir中无实际用处,在 swintransformer 中对应了预训练输入图像的大小,
            # 用于设置绝对位置编码(大小为 num_patchs),但swinir中无绝对位置编码,故此参数无意义)
            img_size=64,
            # patch 的大小(在swinir中无实际用处,因为压根没有分块)
            patch_size=1,
            in_chans=3,  # 输入图像的通道数
            embed_dim=60,  # Patch embedding 的维度(中间特征通道数),注意 embed_dim 必须是 num_heads[i] * 2 的整数倍
            depths=[6, 6, 6, 6],  # 元组的元素个数len(depths)对应了 RHTB 的个数, 元素值对应了 RHTB 中 STL 的个数
            num_heads=[6, 6, 6, 6],  # 在各个 RHTB 中 STL 的注意力头的个数
            base_win_size=[8, 8],  # 基础窗口大小
            mlp_ratio=2.,  # MLP隐藏层特征图通道与嵌入层特征图通道的比
            drop_rate=0.,  # 随机丢弃神经元
            value_drop_rate=0.,
            drop_path_rate=0.,  # 深度随机丢弃率(hitsir不使用)
            norm_layer=nn.LayerNorm,  # 归一化操作
            ape=False,  # 是否给 patch embedding 添加绝对位置 embedding
            patch_norm=True,  # 是否在 PatchEmbed 后添加归一化操作
            use_checkpoint=False,  # 是否使用 checkpointing 来节省显存
            upscale=4,  # 放大因子， 2/3/4/8 适合图像超分, 1 适合图像去噪和 JPEG 压缩去伪影
            img_range=1.,  # 灰度值范围， 1 或者 255
            upsampler='pixelshuffledirect',  # 图像重建方法的选择模块，可选择 pixelshuffle, pixelshuffledirect, nearest+conv 或 None.
            resi_connection='1conv',  # 残差连接之前的卷积块，可选择 1conv 或 3conv.
            # hier_win_ratios 元素个数必须等于 depths 的元素值(所有元素值相等),如默认 depths[i] = 6,这 6 个 STL使用的窗口大小为: base_win_size * hier_win_ratios[i]
            hier_win_ratios=[0.5, 1, 2, 4, 6, 8],
            **kwargs):
        super(HiT_SIR, self).__init__()
        num_in_ch = in_chans  # 输入图片通道数
        num_out_ch = in_chans  # 输出图片通道数
        num_feat = 64  # 上采样模块中特征图的通道数
        self.img_range = img_range  # 灰度值范围:[0, 1] or [0, 255]
        if in_chans == 3:  # 如果输入是RGB图像
            # rgb_mean = (0.4488, 0.4371, 0.4040)
            rgb_mean = (0.485, 0.456, 0.4060)  # imagenet 数据集的 rgb 均值
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)  # 转为[1, 3, 1, 1]的张量
        else:  # 否则灰度图
            self.mean = torch.zeros(1, 1, 1, 1)  # 构造[1, 1, 1, 1]的张量
        self.upscale = upscale  # 图像放大倍数，超分(2/3/4/8),去噪(1)
        self.upsampler = upsampler  # 上采样方法
        self.base_win_size = base_win_size  # 基础窗口大小

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        # 浅层特征提取(将输入图片转换为 patch_embedding)
        self.conv_first = None
        if is_mult_size_conv_feat_extract:
            self.conv_first = MultipleSizeConvExtract(num_in_ch, embed_dim)
            print('浅层特征提取模块使用不同尺寸卷积核提取特征')
        else:
            self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
            print('浅层特征提取模块只使用3x3卷积提取特征')

        # 特征融合模块
        self.fusion = None
        if is_fusion:
            self.fusion = Fusion(embed_dim)
            print('添加深层特征浅层特征融合模块')
        else:
            self.fusion = lambda x, y: x + y
            print('不添加深层特征浅层特征融合模块')

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)  # RHTB 的个数
        self.embed_dim = embed_dim  # 嵌入层特征图的通道数
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim  # 特征图的通道数
        self.mlp_ratio = mlp_ratio

        # 将图像展平成一维 (b,c,h,w) -> (b,h*w,c)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        # 分割得到patch的个数(无实际用处),由于 patch_size = 1,这里 num_patches = img_size[0] * img_size[1]
        num_patches = self.patch_embed.num_patches
        # 分割得到patch的分辨率(无实际用处),由于 patch_size = 1,这里 patches_resolution = (img_size[0],img_size[1])
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # (b,h*w,c) -> (b,c,h,w), 其中 h 和 w 需要在调用 patch_unembed.forward 的第二个参数上指定
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # 绝对位置嵌入
        if self.ape:
            # 结构为 [1，patch个数， 嵌入层特征图的通道数] 的参数
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            # 截断正态分布，限制标准差为0.02
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # 以drop_rate为丢弃率随机丢弃神经元，默认不丢弃
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度衰减规律，默认为 [0, drop_path_rate] 进行 sum(depths) 等分后的列表
        # 假设 depths = [6,6,6,6], 那么 dpr 为 24 个元素的列表
        # sum(depths) 为 24, 对应着总共 24 个 STL,而 dpr 的 24 个元素就用于这 24 个 STL
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Hierarchical Transformer blocks (RHTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHTB(
                is_channel_spatial_attn=is_channel_spatial_attn,
                dim=embed_dim,
                input_resolution=(patches_resolution[0],
                                  patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                base_win_size=base_win_size,
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate, value_drop=value_drop_rate,
                # 取出 dpr 中的 depths[i] 个元素用于 RSTB 中的每个 STL
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                hier_win_ratios=hier_win_ratios
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # 为了减少参数使用和节约显存，采用瓶颈结构
            # (b,c,h,w) -> (b,c//4,h,w) -> (b,c//4,h,w) -> (b,c,h,w)
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),  # 先将特征图维度转换为 num_feat(64)
                nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters) 适合轻量级充分，可以减少参数量(一步是实现既上采样也降维)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'  # 目前仅支持4倍超分重建
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),  # 先将特征图维度转换为 num_feat(64)
                nn.LeakyReLU(inplace=True))
            # 第一次上采样卷积(直接对输入做最近邻插值变为2倍图像)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            # 第二次上采样卷积(直接对输入做最近邻插值变为2倍图像)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            # 对上采样完成的图像再做最后的调整
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        # 初始化网络参数
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        """
        x.size() = (b,c,h,w)
        深层特征提取网络的前向传播
        """
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)  # (b,h*w,c)
        if self.ape:  # 绝对位置 embedding
            # 这里有问题,在 swin transformer 源码中是将 absolute_pos_embed(num_patchs大小) 插值到 x(h*w大小) 大小再相加的
            x = x + self.absolute_pos_embed  # x 加上对应的绝对位置 embedding(默认配置中并不会添加)
        x = self.pos_drop(x)  # 随机将x中的部分元素置 0(默认配置中并不会置0)

        for layer in self.layers:
            x = layer(x, x_size)  # (b,h*w,c),x 通过多个串联的 RHTB

        x = self.norm(x)  # (b,h*w,c),对 RHTB 的输出进行归一化
        x = self.patch_unembed(x, x_size)  # (b,c,h,w)

        return x  # (b,c,h,w)

    def forward(self, x):
        """
        x.size() = (B,3,H,W)
        """
        H, W = x.shape[2:]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)  # (B,embed_dim,H,W)
            # x = self.conv_after_body(self.forward_features(x)) + x  # (B,embed_dim,H,W)
            x = self.fusion(self.conv_after_body(self.forward_features(x)), x)  # (B,embed_dim,H,W)
            x = self.conv_before_upsample(x)  # (B,num_feat,H,W)
            x = self.conv_last(self.upsample(x))  # (B,3,4H,4W)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            # x = self.conv_after_body(self.forward_features(x)) + x
            x = self.fusion(self.conv_after_body(self.forward_features(x)), x)
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            # x = self.conv_after_body(self.forward_features(x)) + x
            x = self.fusion(self.conv_after_body(self.forward_features(x)), x)
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            # res = self.conv_after_body(self.forward_features(x_first)) + x_first
            res = self.fusion(self.conv_after_body(self.forward_features(x_first)), x_first)
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, :H * self.upscale, :W * self.upscale]


if __name__ == '__main__':
    upscale = 4
    base_win_size = [8, 8]
    height = (1024 // upscale // base_win_size[0] + 1) * base_win_size[0]
    width = (720 // upscale // base_win_size[1] + 1) * base_win_size[1]

    ## HiT-SIR
    model = HiT_SIR(upscale=4, img_size=(height, width),
                    base_win_size=base_win_size, img_range=1., depths=[6, 6, 6, 6],
                    embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')

    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("params: ", params_num)
