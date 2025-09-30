import collections.abc
import math
import torch
import torchvision
import warnings
from distutils.version import LooseVersion
from itertools import repeat
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from basicsr.utils import get_root_logger


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()

        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)  # 输入输出宽高不变
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)  # 输入输出宽高不变
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x  # (b,num_feat,h,w)
        out = self.conv2(self.relu(self.conv1(x)))  # (b,num_feat,h,w)
        return identity + out * self.res_scale  # (b,num_feat,h,w)


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
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.
    本方法属于 back_ward warp: 遍历目标图像所有像素坐标位置,其每个位置对应的像素值在 x 上根据 flow 所指示的位置取

    Args:
        x (Tensor): Tensor with size (b, c, h, w). 支持帧
        flow (Tensor): Tensor with size (b, h, w, 2), normal value. 参考帧 -> 支持帧的光流
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    # 确保 (x.h, x.w) == (flow.h, flow.w)
    assert x.size()[-2:] == flow.size()[1:3]

    _, _, h, w = x.size()
    # 创建 hxw 大小的网格坐标
    grid_y, grid_x = torch.meshgrid(
        [torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x)],
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), 2).float()  # shape = (h,w,2)
    grid.requires_grad = False

    # 获取添加了光流向量后发生偏移的网格坐标(这里会触发广播)
    vgrid = grid + flow  # (b,h,w,2)
    # scale grid to [-1,1]
    # 这里考虑的是 grid + flow 的数值范围不会超出图像边界，也就是说 grid + flow 的数值范围和 grid 的数值范围假设是一样的
    # 因为超出图像边界的点坐标我们是直接通过填充0或填充边界像素值这样特殊处理的，故不需要考虑这些非法坐标的规范化，
    # 如果下述规范化公式考虑了 flow 的数值范围的话，会导致原本溢出图像边界的那些非法坐标变得合法，而那些原本合法的坐标
    # 则会偏离原本正确的位置，这样会使warp后的图像产生缩放，这肯定不是我们想要的
    # 由于 grid 的 w 方向(x方向)的数值范围为 [0,w-1], 则 vgrid_x 的数值范围为: 2*[0,w-1]/(w-1)-1 = [-1,1]
    # 由于 grid 的 h 方向(y方向)的数值范围为 [0,h-1], 则 vgrid_y 的数值范围为: 2*[0,h-1]/(h-1)-1 = [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0  # (b,h,w)
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0  # (b,h,w)
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)  # (b,h,w,2)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    return output  # (b, c, h, w)


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale ** 2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    ``Paper: Delving Deep into Deformable Alignment in Video Super-Resolution``
    """

    # 这里的 offset_feat 用于计算 offset, shape = (b, num_feat, h, w)
    # 这里的 x 是待应用可变形卷积的特征图, shape = (b, num_feat, h, w)
    def forward(self, x, offset_feat):
        # 根据 offset_feat 求 offset、mask
        out = self.conv_offset(offset_feat)  # (b, deformable_groups*3N, h, w), N为卷积核参数总数
        o1, o2, mask = torch.chunk(out, 3, dim=1)  # o1.shape=o2.shape=mask.shape=(b, deformable_groups*N, h, w)
        offset = torch.cat((o1, o2), dim=1)  # (b, deformable_groups*2N, h, w)
        mask = torch.sigmoid(mask)  # 由于mask实现的是门机制,故需要用sigmoid激活

        # 计算offset的绝对值的平均值，评判形变大小。太大的话，则一般是跑崩了
        offset_absmean = torch.mean(torch.abs(offset))

        # 由于DCN训练的时候无法稳定的收敛，导致整个网络不好收敛，这是作者认识到的一个问题，也是这个模型存在的训练上的缺陷。也是改进的一个方向。
        # 在实验中，如果训练不稳定则会输出 Offset mean is larger than 100(下面的代码是50)
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')
            # 相当于下面这句
            # print(f'Offset abs mean is {offset_absmean}, larger than 50.')

        # forward的最后一部分，是选择使用哪个DCN模型
        # 获取torchvision的版本号，选择使用哪个DCN模型
        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            # 采用pytorch官方的API
            return torchvision.ops.deform_conv2d(
                x,
                offset,
                self.weight,  # 卷积核的参数
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                mask
            )  # (b, num_feat, h, w)
        else:
            # 采用Basicsr这个库实现的API
            return modulated_deform_conv(
                x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                self.dilation, self.groups, self.deformable_groups
            )


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# From PyTorch
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
