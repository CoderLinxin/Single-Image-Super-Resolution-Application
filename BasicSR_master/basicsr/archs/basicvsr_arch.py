import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
from .edvr_arch import PCDAlignment, TSAFusion
from .spynet_arch import SpyNet


@ARCH_REGISTRY.register()
class BasicVSR(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=15, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        # 输入通道数为 num_feat + 3,输出通道数为 num_feat
        # 这里对应后向/前向分支中的特征矫正模块
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)  # 输入输出宽高不变
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)  # 输入输出宽高不变

        # reconstruction
        # fusion 采用的仍然是简单的按通道连接
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)  # 输入输出宽高不变
        # 进行 pixel_shuffle 上采样前的通道数扩充
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)  # 输入输出宽高不变
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)  # 输入输出宽高不变
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)  # 输入输出宽高不变
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)  # 输入输出宽高不变

        # 2倍上采样
        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        """
        Args:
            x: (b, n, c, h, w), n 可以看成整个 clip 中的帧总数, 即 x 由整个序列的所有帧组成
        """
        b, n, c, h, w = x.size()

        # 取头帧到倒数第二帧
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)  # (b*(n-1),c,h,w)
        # 取第二帧到尾帧
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)  # (b*(n-1),c,h,w)

        # 求反向传播的光流: 反向传播的光流估计模块输入为 xi、xi+1 输出光流估计值 fxi->xi+1,
        # 目的是为了让 xi+1(或者说hi+1) 对齐 xi,后续 warp 的输入为 fxi->xi+1、hi+1
        # xi  : [    第0帧,               第1帧,          ...,      倒数第二帧(第n-2帧) ]
        # xi+1: [    第1帧,               第2帧,          ...,        尾帧(第n-1帧)   ]
        # 输出: [(第0帧->第1帧)的光流, (第1帧->第2帧)的光流, ..., (倒数第二帧->尾帧)的光流]
        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)  # (b, n - 1, 2, h, w)
        # 求正向传播的光流: 正向传播的光流估计模块输入为 xi、xi-1 输出光流估计值 fxi->xi-1,
        # 目的是为了让 xi-1(或者说hi-1) 对齐 xi,后续 warp 的输入为 fxi->xi-1、hi-1
        # xi  : [        第1帧,            第2帧,         ...,        尾帧(第n-1帧)   ]
        # xi-1: [        第0帧,            第1帧,         ...,    倒数第二帧(第n-2帧) ]
        # 输出: [(第1帧->第0帧)的光流, (第2帧->第1帧)的光流, ..., (尾帧->倒数第二帧)的光流]
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)  # (b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, x):
        """Forward function of BasicVSR.
        Args:
            x: Input frames with shape (b, n, c, h, w). n 为 clip 中的帧总数. c = 3
        """
        flows_forward, flows_backward = self.get_flow(x)  # (b,n-1,2,h,w),(b,n-1,2,h,w)
        b, n, _, h, w = x.size()

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)  # (b,num_feat,h,w)
        for i in range(n - 1, -1, -1):  # 遍历 [n-1,n-2,...,0], 即从尾帧遍历到首帧
            # 获取当前帧 xi
            x_i = x[:, i, :, :, :]  # (b,3,h,w)
            # 尾帧(索引为n-1)作为xi时,由于不存在xi+1,故跳过对齐过程
            # 当xi不是尾帧时,需要将xi+1(或者说hi+1)对齐到xi上
            if i < n - 1:
                # 获取光流fxi->xi+1
                flow = flows_backward[:, i, :, :, :]  # (b,2,h,w)
                # 对xi+1(或者说hi+1)根据光流fxi->xi+1进行warp对齐到xi上得到hi_bar
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))  # (b,num_feat,h,w)
            # 将xi与hi_bar进行concat然后送入特征矫正模块得到hi,hi包含了xi后面所有帧的信息
            # hi在下一轮循环中对应hi+1
            feat_prop = torch.cat([x_i, feat_prop], dim=1)  # (b,3+num_feat,h,w)
            feat_prop = self.backward_trunk(feat_prop)  # (b,num_feat,h,w)
            out_l.insert(0, feat_prop)  # [h0,h1,...,hn-1] 其中索引为i的元素表示对齐到xi的特征图,其包含了xi后面所有帧的信息

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)  # (b,num_feat,h,w)
        for i in range(0, n):  # 遍历 [0,1,...,n-1], 即从首帧遍历到尾帧
            # 获取当前帧 xi
            x_i = x[:, i, :, :, :]  # (b,3,h,w)
            # 首帧作为xi时,由于不存在xi-1,故跳过对齐过程
            # 当xi不是首帧时,需要将xi-1(或者说hi-1)对齐到xi上
            if i > 0:
                # 获取光流fxi->xi-1
                flow = flows_forward[:, i - 1, :, :, :]  # (b,2,h,w)
                # 对x-1(或者说hi-1)根据光流fxi->xi-1进行warp对齐到xi上得到hi_bar
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))  # (b,num_feat,h,w)
            # 将xi与hi_bar进行concat然后送入特征矫正模块得到hi,hi包含了xi前面所有帧的信息
            # hi在下一轮循环中对应hi-1
            feat_prop = torch.cat([x_i, feat_prop], dim=1)  # (b,3+num_feat,h,w)
            feat_prop = self.forward_trunk(feat_prop)  # (b,num_feat,h,w)

            # 重建阶段
            # 将后向分支的hi和前向分支的hi在通道上concat(early fusion)后送入特征融合模块
            # 由于后面实际上学习的是xi的残差,故这里特征融合不需要concat上xi(如果最终学习的不是残差,则这里最好再concat上xi)
            out = torch.cat([out_l[i], feat_prop], dim=1)  # (b,2*num_feat,h,w)
            out = self.lrelu(self.fusion(out))  # (b,num_feat,h,w)

            # 对融合的特征图进行4倍上采样
            out = self.lrelu(self.pixel_shuffle(
                self.upconv1(out)  # (b,4*num_feat,h,w)
            ))  # (b,num_feat,2*h,2*w)
            out = self.lrelu(self.pixel_shuffle(
                self.upconv2(out)  # (b,64*4,2*h,2*w)
            ))  # (b,64,4*h,4*w)

            # 对上采样后的特征图进行最后的调整
            out = self.lrelu(self.conv_hr(out))  # (b,64,4*h,4*w)
            out = self.conv_last(out)  # (b,3,4*h,4*w)

            # 对xi使用插值放大4倍然后与out进行残差连接
            base = F.interpolate(x_i, scale_factor=4, mode='bicubic', align_corners=False)
            out_l[i] = out + base  # (b,3,4*h,4*w)

        # 这里可以看到BasicVSR输入一个帧序列然后输出一个帧序列,重建效率明显提高很多
        return torch.stack(out_l, dim=1)  # (b,n,3,4*h,4*w)


class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            # 先用一个卷积层将输入特征图的通道数转换为 num_out_ch
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True),  # 输入输出宽高不变
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # 堆积 num_block 个残差块
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch)  # 输入输出宽高通道数不变
        )

    def forward(self, fea):
        """
        Args:
            fea: (b,num_in_ch,h,w)
        """
        return self.main(fea)  # (b,num_out_ch,h,w)


@ARCH_REGISTRY.register()
class IconVSR(nn.Module):
    """IconVSR, proposed also in the BasicVSR paper.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15.
        keyframe_stride (int): Keyframe stride(在相邻帧集合中确定关键帧所依据的间隔). Default: 5.
        temporal_padding (int): Temporal padding Default: 2,
            这个参数决定了在提取某个关键帧特征时所选择的相邻帧数量的一半(左右各 temporal_padding 个)
            2 表示选取的相邻帧为当前关键帧的左边2帧加右边2帧
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(
            self,
            num_feat=64,
            num_block=15,
            keyframe_stride=5,
            temporal_padding=2,
            spynet_path=None,
    ):
        super().__init__()

        self.num_feat = num_feat
        self.temporal_padding = temporal_padding
        self.keyframe_stride = keyframe_stride

        # 关键帧特征提取模块(前向后向分支中都使用此edvr模块)
        # 提取某个关键帧时输入的帧为: 关键帧左边 temporal_padding 个相邻帧、关键帧、关键帧右边 temporal_padding 个相邻帧
        # 因此输入通道数为 temporal_padding * 2 + 1
        self.edvr = EDVRFeatureExtractor(temporal_padding * 2 + 1, num_feat)
        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        # 对应后向/前向分支中的特征矫正模块
        # backward_trunk 的特征矫正模块需要同时接收:
        # xi(通道数为3)
        # hi_bar(通道数为num_feat)
        # 故输入通道为 num_feat + 3
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)  # 输入输出宽高不变
        # forward_trunk 的特征矫正模块需要同时接收:
        # xi(通道数为3)
        # hi_bar(通道数为num_feat)
        # edvr模块的输出特征(通道数为num_feat,IConVSR中新增)
        # 故输入通道为 2*num_feat + 3
        self.forward_trunk = ConvResidualBlocks(2 * num_feat + 3, num_feat, num_block)  # 输入输出宽高不变

        # fusion 采用的仍然是简单的按通道连接
        self.backward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)  # 输入输出宽高不变
        self.forward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)  # 输入输出宽高不变

        # reconstruction(同BasicVSR)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)  # 输入输出宽高不变
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)  # 输入输出宽高不变
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)  # 输入输出宽高不变
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)  # 输入输出宽高不变

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    # 对 x 进行宽高上的补充,使其宽高能够被 4 所整除以支持任意宽高大小的 x 能够输入 EDVR 模块
    def pad_spatial(self, x):
        """
        Apply padding spatially.
        Since the PCD module in EDVR requires that the resolution is a multiple
        of 4, we apply padding to the input LR images if their resolution is
        not divisible by 4.

        Args:
            x (Tensor): Input LR sequence with shape (n, t, c, h, w).
        Returns:
            Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).
        """
        n, t, c, h, w = x.size()

        pad_h = (4 - h % 4) % 4  # 最右边的 %4 是防止 h % 4 为 0 时, pad_h 的值能为 0
        pad_w = (4 - w % 4) % 4

        # padding
        x = x.view(-1, c, h, w)  # (n*t, c, h, w)
        # pad=(左边填充数,右边填充数, 上边填充数,下边填充数),这里的填充指的是针对最后两维(2D图)的填充
        #          调整列数(宽)        调整行数(高)
        # 之所以选择右边和下边进行填充是因为之后重建完成需要丢弃 padding 的部分，也就是丢弃右边和下边的内容
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')  # (n*t, c, h_pad, w_pad)

        return x.view(n, t, c, h + pad_h, w + pad_w)  # (n, t, c, h_pad, w_pad)

    # 与 BasicVSR.get_flow 实现一致
    def get_flow(self, x):
        """
        Args:
            x: (b, n, c, h, w), n 可以看成整个 clip 中的帧总数, 即 x 由整个序列的所有帧组成
        """
        b, n, c, h, w = x.size()

        # 取头帧到倒数第二帧
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)  # (b*(n-1),c,h,w)
        # 取第二帧到尾帧
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)  # (b*(n-1),c,h,w)

        # 求反向传播的光流: 反向传播的光流估计模块输入为 xi、xi+1 输出光流估计值 fxi->xi+1,
        # 目的是为了让 xi+1(或者说hi+1) 对齐 xi,后续 warp 的输入为 fxi->xi+1、hi+1
        # xi  : [    第0帧,               第1帧,          ...,      倒数第二帧(第n-2帧) ]
        # xi+1: [    第1帧,               第2帧,          ...,        尾帧(第n-1帧)   ]
        # 输出: [(第0帧->第1帧)的光流, (第1帧->第2帧)的光流, ..., (倒数第二帧->尾帧)的光流]
        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)  # (b, n - 1, 2, h, w)
        # 求正向传播的光流: 正向传播的光流估计模块输入为 xi、xi-1 输出光流估计值 fxi->xi-1,
        # 目的是为了让 xi-1(或者说hi-1) 对齐 xi,后续 warp 的输入为 fxi->xi-1、hi-1
        # xi  : [        第1帧,            第2帧,         ...,        尾帧(第n-1帧)   ]
        # xi-1: [        第0帧,            第1帧,         ...,    倒数第二帧(第n-2帧) ]
        # 输出: [(第1帧->第0帧)的光流, (第2帧->第1帧)的光流, ..., (尾帧->倒数第二帧)的光流]
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)  # (b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    # 获取所有关键帧的特征
    def get_keyframe_feature(self, x, keyframe_idx):
        """
        Args:
            x: 输入序列, shape = (b,t,c,h,w)
            keyframe_idx: 输入序列 x 中那些属于关键帧的索引组成的序列 shape = (len(keyframe_idx),)
        """
        # 由于关键帧必包含输入序列 x 的首帧和尾帧,但是首帧左边没有相邻帧,尾帧则右边没有相邻帧
        # 因此对关键帧进行特征提取时,需要对首帧左边和尾帧右边进行填充 temporal_padding 个帧
        if self.temporal_padding == 2:
            # 这里选择的填充方式为: reflection_circle 模式
            # 如输入序列 x 的索引为:    0、1、2、3、4、5、6、7、8、9
            # 则填充后为:        4、3、0、1、2、3、4、5、6、7、8、9、5、6
            # 关键帧间隔为 3,关键帧索引为: 0、3、6、9
            # 对原本索引为 0、3、6、9 的关键帧进行特征提取: [4、3、0、1、2], [1、2、3、4、5], [4、5、6、7、8], [7、8、9、5、6]
            # (以填充首帧索引 0 为例:
            # circle 表示取 0 右边 temporal_padding 个相邻帧后面的 temporal_padding 个索引来进行填充: 3、4、0、1、2
            # reflection 表示翻转索引,即 4、3、0、1、2
            x = [x[:, [4, 3]], x, x[:, [-4, -5]]]  # [(b,2,c,h,w),(b,t,c,h,w),(b,2,c,h,w)]
        elif self.temporal_padding == 3:
            x = [x[:, [6, 5, 4]], x, x[:, [-5, -6, -7]]]  # [(b,3,c,h,w),(b,t,c,h,w),(b,3,c,h,w)]
        x = torch.cat(x, dim=1)  # (b,t+2*self.temporal_padding,c,h,w)

        num_frames = 2 * self.temporal_padding + 1
        feats_keyframe = {}
        for i in keyframe_idx:
            # 对所有关键帧进行特征提取: feats_keyframe[i] 对应原本输入序列(b,t,c,h,w) x 中索引为 i 的关键帧特征
            feats_keyframe[i] = self.edvr(x[:, i:i + num_frames].contiguous())
        return feats_keyframe  # (b,len(keyframe_idx),c,h,w)

    def forward(self, x):
        """
        Args:
            x: 输入帧序列, shape = (b,n,c,h,w)
        """
        b, n, _, h_input, w_input = x.size()

        # 如果 x 的宽高不能被 4 整除,就要进行填充
        x = self.pad_spatial(x)  # (b,n,c,h_pad,w_pad)
        h, w = x.shape[3:]

        # 根据间隔 keyframe_stride 在输入序列 x 中挑选关键帧(索引)
        keyframe_idx = list(range(0, n, self.keyframe_stride))
        # 确保输入序列 x 的最后一帧也是关键帧
        if keyframe_idx[-1] != n - 1:
            keyframe_idx.append(n - 1)  # last frame is a keyframe

        # compute flow and keyframe features
        flows_forward, flows_backward = self.get_flow(x)  # (b,n-1,2,h_pad,w_pad), (b,n-1,2,h_pad,w_pad)
        # feats_keyframes 包含了所有关键帧的特征
        feats_keyframe = self.get_keyframe_feature(x, keyframe_idx)  # # (b,len(keyframe_idx),c,h_pad,w_pad)

        # backward branch(基本上与basicVSR中的实现一致)
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))  # hi_bar
            # 不同之处: 如果 xi 是关键帧,则额外对其进行特征提取
            if i in keyframe_idx:
                # 将 hi_bar 与预先提取好的关键帧的特征 feats_keyframe[i] 进行 cat 并进行特征提取
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.backward_fusion(feat_prop)
            # 同 basicVSR
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch(基本上与basicVSR中的实现一致)
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            # 不同之处: 仍然增加了对 xi 为关键帧时的额外特征提取
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.forward_fusion(feat_prop)

            # 不同之处, 除了 concat[x_i、feat_prop] 外，还有后向分支的输出 out_l[i]
            feat_prop = torch.cat([x_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)  # (b,num_feat,h_pad,w_pad)

            # upsample
            out = self.lrelu(self.pixel_shuffle(
                self.upconv1(feat_prop)  # (b,4*num_feat,h_pad,w_pad)
            ))  # (b,num_feat,2*h_pad,2*w_pad)
            out = self.lrelu(self.pixel_shuffle(
                self.upconv2(out)  # (b,4*64,2*h_pad,2*w_pad)
            ))  # (b,64,4*h_pad,4*w_pad)

            out = self.lrelu(self.conv_hr(out))  # (b,64,4*h_pad,4*w_pad)
            out = self.conv_last(out)  # (b,3,4*h_pad,4*w_pad)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out_l[i] = out + base  # (b,3,4*h_pad,4*w_pad)

        # 如果 xi 不能整除 4, 那么经过宽高上的 padding 进行重建后的结果会舍弃掉 padding 的部分(右边和下边)
        return torch.stack(out_l, dim=1)[..., :4 * h_input, :4 * w_input]  # (b,n,3,4*h,4*w)


# 注意 EDVR 中使用了特征金字塔会进行4倍下采样,且没有对下采样结果宽高大小可能出现的向下取整进行相关补充措施,故需要确保输入特征图的宽高是4的整数倍
class EDVRFeatureExtractor(nn.Module):
    """EDVR feature extractor used in IconVSR.

    Args:
        num_input_frame (int): Number of input frames.
        num_feat (int): Number of feature channels
    """

    def __init__(self, num_input_frame, num_feat):
        super(EDVRFeatureExtractor, self).__init__()

        # 获取中间帧(参考帧)索引
        self.center_frame_idx = num_input_frame // 2

        # extract pyramid features
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)  # 输入输出宽高不变
        self.feature_extraction = make_layer(ResidualBlockNoBN, 5, num_feat=num_feat)  # 输入输出宽高不变
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)  # 输出宽高变为输入宽高的一半
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 输入输出宽高不变
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)  # 输出宽高变为输入宽高的一半
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 输入输出宽高不变

        # pcd and tsa module
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=8)
        self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_input_frame, center_frame_idx=self.center_frame_idx)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        b, n, c, h, w = x.size()

        # 对输入的所有帧构建特征金字塔(准备输入PCA模块)
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))  # (b*n,c,h,w)
        feat_l1 = self.feature_extraction(feat_l1)  # (b*n,c,h,w)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))  # (b*n,c,h/2,w/2)
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))  # (b*n,c,h/2,w/2)
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))  # (b*n,c,h/4,w/4)
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))  # (b*n,c,h/4,w/4)

        feat_l1 = feat_l1.view(b, n, -1, h, w)  # (b,n,c,h,w)
        feat_l2 = feat_l2.view(b, n, -1, h // 2, w // 2)  # (b,n,c,h/2,w/2)
        feat_l3 = feat_l3.view(b, n, -1, h // 4, w // 4)  # (b,n,c,h/4,w/4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(),  # (b,c,h,w)
            feat_l2[:, self.center_frame_idx, :, :, :].clone(),  # (b,c,h/2,w/2)
            feat_l3[:, self.center_frame_idx, :, :, :].clone()  # (b,c,h/4,w/4)
        ]
        aligned_feat = []
        for i in range(n):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(),
                feat_l2[:, i, :, :, :].clone(),
                feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b,n,c,h,w)

        # TSA fusion
        return self.fusion(aligned_feat)  # (b, c, h, w)
