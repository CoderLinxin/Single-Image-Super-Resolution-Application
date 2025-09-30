import math
import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import flow_warp


class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        # 代表一个 Gk, 通道数为 {32,64,32,16,2}
        self.basic_module = nn.Sequential(
            # 由于 stride = 1, padding = kernel_size // 2, 故下述卷积操作后的特征图宽高保持不变
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, tensor_input: torch.Tensor):
        """
        Args:
            tensor_input: shape = (b,8,h,w)
            tensor_input 为: torch.cat([本层帧1, warp(本层帧2,上层光流估计的上采样结果), 上层光流估计的上采样结果])
                                通道数:    3                    3                            2
        Returns:

        """
        return self.basic_module(tensor_input)  # (b,2,h,w)


@ARCH_REGISTRY.register()
class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
    """

    def __init__(self, load_path: str = None):
        super(SpyNet, self).__init__()
        # 六层金字塔
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            # 修改 map_location 为 device(todo)
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

        # 对输入数据进行预处理的正态分布均值和方差定义(取自大型数据集)
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    # 对数据进行规范化预处理
    def preprocess(self, tensor_input: torch.Tensor):
        """
        Args:
            tensor_input: shape = (b,3,h,w)
        """
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output  # (b,3,h,w)

    # 推理过程(根据参考帧和支持帧获取对应的光流估计)
    def process(self, ref: torch.Tensor, supp: torch.Tensor):
        """
        Args:
            ref: 参考帧, shape = (b,3,h,w)
            supp: 支持帧, shape = (b,3,h,w)
        """
        flow = []

        ref = [self.preprocess(ref)]  # (b,3,h,w)
        supp = [self.preprocess(supp)]  # (b,3,h,w)

        # 使用均值滤波下采样操作以构建 6 层金字塔(原图大小作为一层)
        # 输入图像宽高尽量是 2^5 = 32 的整数倍,否则由于输出形状大小默认向下取整,某些数据会被丢弃
        # [ref↓32, ref↓16, ref↓8, ref↓4, ref↓2, ref] 记为 {refIk}
        # [supp↓32, supp↓16, supp↓8, supp↓4, supp↓2, supp] 记为 {suppIk}
        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        # 初始光流 V-1 为全 0
        # 由于这里还有个最底层再2倍下采样的光流估计,故输入图像宽高尽量是 2^6 = 64 的整数倍
        flow = ref[0].new_zeros([
            ref[0].size(0),
            2,
            int(math.floor(ref[0].size(2) / 2.0)),
            int(math.floor(ref[0].size(3) / 2.0))
        ])  # (b,2,h/64,w/64)

        # 遍历金字塔
        # flow: (b,2,h/64,w/64) -> (b,2,h,w)
        for level in range(len(ref)):
            # 对光流进行上采样得到宽高等于当前层特征图宽高的光流估计 u(Vk-1)
            # 由于上采样后数据尺度(尺寸)变大2倍(运动的区域放大了),故相应的数值部分也要*2来匹配(光流向量的模长也要跟着放大)
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            # 由于 V-1 的宽高可能出现向下取整的情况: 7 // 2 = 3 -> 3 * 2 + 1 = 7
            # 故这里确保 upsampled_flow 的宽高等于当前层特征图宽高
            if upsampled_flow.size(2) != ref[level].size(2):
                # pad=(左边填充数,右边填充数, 上边填充数,下边填充数),这里的填充指的是针对最后两维(2D图)的填充
                #          调整列数(宽)        调整行数(高)
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            # Gk 接收: 通道cat([refIk,   warp(suppIk,u(Vk-1)),       u(Vk-1)])
            #         通道cat([本层ref, 本层supp经过u(Vk-1)的warp结果, u(Vk-1)])
            # 得到本层的光流估计残差预测: vk
            # 最后得到本层的光流估计预测: Vk = vk + u(Vk-1)
            flow = self.basic_module[level](
                torch.cat([
                    ref[level],  # (b,3,hk,wk)
                    flow_warp(
                        supp[level],
                        upsampled_flow.permute(0, 2, 3, 1),  # (b,hk,wk,2)
                        interp_mode='bilinear',
                        padding_mode='border'
                    ),  # (b,3,hk,wk)
                    upsampled_flow  # (b,2,hk,wk)
                ], 1)  # (b,8,hk,wk)
            ) + upsampled_flow  # (b,2,hk,wk)

        # 得到的光流: ref -> supp
        return flow  # (b,2,h,w)

    def forward(self, ref: torch.Tensor, supp: torch.Tensor):
        """
        Args:
            ref: (b,3,h,w)
            supp: (b,3,h,w)
        """
        assert ref.size() == supp.size()

        # 为了构建金字塔,需要确保图片宽高是 2^5 = 32 的整数倍(可能会对图片宽高进行放大)
        h, w = ref.size(2), ref.size(3)
        h_ceil = math.ceil(h / 32.0) * 32
        w_ceil = math.ceil(w / 32.0) * 32

        # 对图片宽高大小进行插值调整
        ref = F.interpolate(
            input=ref, size=(h_ceil, w_ceil), mode='bilinear', align_corners=False
        )  # (b,3,h_ceil,w_ceil)
        supp = F.interpolate(
            input=supp, size=(h_ceil, w_ceil), mode='bilinear', align_corners=False
        )  # (b,3,h_ceil,w_ceil)

        # 根据 ref、supp 获取 flow
        flow = self.process(ref, supp)  # (b,2,h_ceil,w_ceil)
        # 将 flow 的宽高调整成 hxw,由于此时 flow 的宽高大小可能发生变化,即尺度发生变化,相应地,数据也要进行匹配(后续代码)
        flow = F.interpolate(input=flow, size=(h, w), mode='bilinear', align_corners=False)  # (b,2,h,w)

        # 由于 w_ceil 可能大于 w, h_ceil 可能大于 h
        # (h_ceil*(h/h_ceil), w_ceil*(w/w_ceil)) = (h,w)
        # h_scaling_factor = h/h_ceil, w_scaling_factor = w/w_ceil
        # 故经过插值调整前后的 flow 宽高大小(尺度)可能是不一样的,因此相应的数值尺度也要乘以对应的缩放因子以匹配
        flow[:, 0, :, :] *= float(w) / float(w_ceil)  # 第0个通道对应x方向,即宽方向
        flow[:, 1, :, :] *= float(h) / float(h_ceil)  # 第1个通道对应y方向,即高方向

        return flow  # (b,2,h,w)
