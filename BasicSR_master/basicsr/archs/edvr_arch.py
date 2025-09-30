import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import DCNv2Pack, ResidualBlockNoBN, make_layer


class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.
    ``Paper: EDVR: Video Restoration with Enhanced Deformable Convolutional Networks``

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        # Pyramids
        for i in range(3, 0, -1):  # 从金字塔顶端往回走: 3->2->1
            level = f'l{i}'  # l3,l2,l1

            # 两帧 cat 求 offset(的特征), 故输入通道数为 num_feat * 2
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)  # 输入输出宽高不变

            if i == 3:  # 由于不用合并上一层(没有再上一层)的帧,所以这里输入输出通道都是 num_feat
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 输入输出宽高不变
            else:  # 否则，就先经过一次卷积将通道减半，变回num_feat。然后再用一次卷积(输入和输出通道都是num_feat)操作
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)  # 输入输出宽高不变
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 输入输出宽高不变

            # 将本层最终offset(的特征)(这里说是最终offset,是说这是求offset前的最后操作了。其实应该是feat,
            # 这个feat是根据参考帧与支持帧的feat)
            # 将这个offset(feat)与需要align的支持帧feat送入dcn
            # 注意真正的offset是在DCNv2Pack里面求的
            self.dcn_pack[level] = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)  # 输入输出宽高不变

        # Cascading dcn
        # 将最终生成的对齐feat,与L1层的参考帧特征cat.所以输入通道数加倍
        # 通过cas_offset_conv1。使得通道数变回num_feat
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)  # 输入输出宽高不变
        # 通过cas_offset_conv2。通道数不变
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 输入输出宽高不变
        # 求解最终的对齐特征
        self.cas_dcnpack = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)

        # 对L3和L2的特征与offset，都要进行2倍的上采样，用于和下一层进行拼接
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, num_feat, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, num_feat, h, w).

        Returns:
            Tensor: Aligned features.
        """
        # Pyramids
        upsampled_offset, upsampled_feat = None, None

        for i in range(3, 0, -1):  # 3->2->1
            level = f'l{i}'

            # 将第i层(索引为i-1)的参考帧特征与支持帧特征在通道维度上拼接
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)  # (b, num_feat*2, h, w)
            # 将通道数减半，变回num_feat
            offset = self.lrelu(self.offset_conv1[level](offset))  # (b, num_feat, h, w)

            if i == 3:  # 如果是第三层,则直接求offset(的特征)(其他层是要接受上一层的offset和特征,进行拼接的)
                offset = self.lrelu(self.offset_conv2[level](offset))  # (b, num_feat, h, w)
            else:  # 不是第三层,需要先将上一层的offset上采样与本层offset(的特征)拼接,之后再求本层offset(的特征)。
                # 先将通道数减半: num_feat*2->num_feat
                offset = self.lrelu(
                    self.offset_conv2[level](
                        torch.cat([offset, upsampled_offset], dim=1)  # (b, num_feat*2, h, w)
                    )
                )  # (b, num_feat, h, w)
                # 求offset(的特征)
                offset = self.lrelu(self.offset_conv3[level](offset))  # (b, num_feat, h, w)

            # 将本层求得的offset(的特征)与需要align的支持帧送入dcn
            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)  # (b, num_feat, h, w)

            if i < 3:
                # 同样，如果是L2和L1层。需要将本层的通过dcn求得的align feature与上一层的align feature(上采样)拼接
                # 拼接后，通道数加倍。则经过feat_conv，通道数变回num_feat
                # 这样才是得到了本层最终的align feature
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1)  # (b, num_feat*2, h, w)
                )  # (b, num_feat, h, w)
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:
                # 对 offset and features 进行上采样
                # 当我们对偏移量offset进行上采样时,我们还应该放大幅度(*2)
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # 将金字塔最终生成的对齐feat,与L1层的参考帧特征cat
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)  # (b, num_feat*2, h, w)
        # 再一次计算对齐feat与参考帧的offset(的特征)
        offset = self.lrelu(self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))  # (b, num_feat, h, w)
        # 将feat和offset(的特征)输入dcn对feat进一步的调整校正
        feat = self.lrelu(self.cas_dcnpack(feat, offset))  # (b, num_feat, h, w)
        return feat


class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(TSAFusion, self).__init__()

        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 输入输出宽高不变
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 输入输出宽高不变
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)  # 1x1卷积输入输出宽高不变

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)  # stride=2的跨步卷积,输出宽高减少为输入的一半
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)  # stride=2的跨步卷积,输出宽高减少为输入的一半
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)  # 1x1卷积,输入输出宽高不变
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)  # 1x1卷积,输入输出宽高不变
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 输入输出宽高不变
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)  # 输入输出宽高不变
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 输入输出宽高不变
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)  # 1x1卷积,输入输出宽高不变
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)  # 输入输出宽高不变
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 输入输出宽高不变
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)  # 1x1卷积,输入输出宽高不变
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)  # 1x1卷积,输入输出宽高不变

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # 2倍上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, num_feat, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, num_feat, h, w).
        """
        b, t, c, h, w = aligned_feat.size()  # c 默认是 num_feat

        # temporal attention
        # 先对参考帧和支持帧(也包括参考帧)进行 embedding 操作(实际上就是卷积运算)
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())  # (b,num_feat,h,w)
        embedding = self.temporal_attn2(
            aligned_feat.view(-1, c, h, w)  # (b*t,num_feat,h,w)
        )  # (b*t,num_feat,h,w)
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, num_feat, h, w)

        # 存放各个支持帧与参考帧相关性矩阵的列表
        corr_l = []
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]  # (b, num_feat, h, w)
            # 计算支持帧与参考帧的相关性(点积)并在通道上求和,计算的是空间注意力
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(
                corr.unsqueeze(1)  # (b, 1, h, w)
            )  # 总共t个列表元素

        # 由于实现的是门机制,故注意力权重需要进行sigmoid激活
        corr_prob = torch.sigmoid(
            torch.cat(corr_l, dim=1)  # (b, t, h, w)
        )  # (b, t, h, w)
        # 为了与 aligned_feat 进行逐元素乘,需要扩展维度(同一位置的所有通道使用相同的注意力权重)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)  # (b, t, 1, h, w) -> # (b, t, num_feat, h, w)
        # 计算完基于时间的(空间)注意力权重后,时间维度t可消去
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*num_feat, h, w)
        # 将时间注意力权重与对齐特征图进行逐元素乘
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob  # (b, t*num_feat, h, w)

        # 进行时间融合特征提取(消去通道上的时间t)(金字塔第0层)
        feat = self.lrelu(self.feat_fusion(aligned_feat))  # (b, num_feat, h, w)

        # spatial attention
        # 根据 aligned_feat(其实也可以是feat) 计算 attn(金字塔第1层)
        attn = self.lrelu(self.spatial_attn1(
            aligned_feat  # (b, t*num_feat, h, w)
        ))  # (b, num_feat, h, w)
        attn_max = self.max_pool(attn)  # (b, num_feat, h/2, w/2)
        attn_avg = self.avg_pool(attn)  # (b, num_feat, h/2, w/2)
        attn = self.lrelu(self.spatial_attn2(
            torch.cat([attn_max, attn_avg], dim=1)  # (b, 2*num_feat, h/2, w/2)
        ))  # (b, num_feat, h/2, w/2)

        # 根据 attn 计算 attn_level(金字塔第2层)
        attn_level = self.lrelu(self.spatial_attn_l1(attn))  # (b, num_feat, h/2, w/2)
        attn_max = self.max_pool(attn_level)  # (b, num_feat, h/4, w/4)
        attn_avg = self.avg_pool(attn_level)  # (b, num_feat, h/4, w/4)
        attn_level = self.lrelu(self.spatial_attn_l2(
            torch.cat([attn_max, attn_avg], dim=1)  # (b, 2*num_feat, h/4, w/4)
        ))  # (b, num_feat, h/4, w/4)

        # 将 attn_level 上采样并与 attn 相加得到 attn_middle
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))  # (b, num_feat, h/4, w/4)
        attn_level = self.upsample(attn_level)  # (b, num_feat, h/2, w/2)
        attn_middle = self.lrelu(self.spatial_attn3(attn)) + attn_level  # (b, num_feat, h/2, w/2)

        # 对 attn_middle 进行上采样得到 attn_middle_up
        attn_middle = self.lrelu(self.spatial_attn4(attn_middle))  # (b, num_feat, h/2, w/2)
        attn_middle_up = self.upsample(attn_middle)  # (b, num_feat, h, w)
        attn_middle_up = self.spatial_attn5(attn_middle_up)  # (b, num_feat, h, w)

        # 将 attn_middle_up 分成两部分,一部分直接与feat相加,一部分作为feat的空间注意力权重
        attn_add = self.spatial_attn_add2(self.lrelu(self.spatial_attn_add1(attn_middle_up)))  # (b, num_feat, h, w)
        attn_weight = torch.sigmoid(attn_middle_up)  # (b, num_feat, h, w)

        # after initialization, * 2 makes (attn_weight * 2) to be close to 1.
        feat = feat * attn_weight * 2 + attn_add  # (b, num_feat, h, w)
        return feat  # (b, num_feat, h, w)


class PredeblurModule(nn.Module):
    """Pre-dublur module.

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        hr_in (bool): Whether the input has high resolution. Default: False.
    """

    def __init__(self, num_in_ch=3, num_feat=64, hr_in=False):
        super(PredeblurModule, self).__init__()
        self.hr_in = hr_in

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)  # 输入输出宽高不变
        if self.hr_in:
            # 如果输入的是 hr 图像，则通过 stride=2 的跨步卷积将宽高缩小 4 倍
            self.stride_conv_hr1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)  # 输出宽高变为输入的一半
            self.stride_conv_hr2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)  # 输出宽高变为输入的一半

        # 使用 stride=2 的跨步卷积生成特征金字塔
        self.stride_conv_l2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)  # 输出宽高变为输入的一半
        self.stride_conv_l3 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)  # 输出宽高变为输入的一半

        # 对特征金字塔使用一系列残差块进行特征提取
        self.resblock_l3 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_1 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_2 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l1 = nn.ModuleList([ResidualBlockNoBN(num_feat=num_feat) for i in range(5)])

        # 双线性插值上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        # 设 x.shape = (b*t,c,h,w)
        # 若 x 是 HR 图像，则 shape = (b*t,c,H,W), 其中 H=4*h, W=4*w

        # 首先对输入 x 进行特征提取得到 l1
        feat_l1 = self.lrelu(self.conv_first(x))  # (b*t,num_feat,h,w)

        # 如果 x 是 hr 图像则宽高缩小 4 倍
        if self.hr_in:
            feat_l1 = self.lrelu(self.stride_conv_hr1(feat_l1))  # (b*t,num_feat,H/2,W/2)
            feat_l1 = self.lrelu(self.stride_conv_hr2(feat_l1))  # (b*t,num_feat,H/4,W/4)

        # 生成特征金字塔 l2、l3
        feat_l2 = self.lrelu(self.stride_conv_l2(feat_l1))  # (b*t,num_feat,h/2,w/2)
        feat_l3 = self.lrelu(self.stride_conv_l3(feat_l2))  # (b*t,num_feat,h/4,w/4)

        # 对特征金字塔进行特征提取
        feat_l3 = self.upsample(self.resblock_l3(feat_l3))  # (b*t,num_feat,h/2,w/2)
        feat_l2 = self.resblock_l2_1(feat_l2) + feat_l3  # (b*t,num_feat,h/2,w/2)
        feat_l2 = self.upsample(self.resblock_l2_2(feat_l2))  # (b*t,num_feat,h,w)

        # 对 l1 特征串联两个残差块提取特征
        for i in range(2):
            feat_l1 = self.resblock_l1[i](feat_l1)  # (b*t,num_feat,h,w)

        # 将 l1 特征与特征金字塔提取的 l2 特征相加
        feat_l1 = feat_l1 + feat_l2  # (b*t,num_feat,h,w)

        # 再串联三个残差块提取特征
        for i in range(2, 5):
            feat_l1 = self.resblock_l1[i](feat_l1)  # (b*t,num_feat,h,w)
        return feat_l1  # (b*t,num_feat,h,w)


# 整体顺序: 特征提取 -> 对齐 -> 融合 -> 重建
@ARCH_REGISTRY.register()
class EDVR(nn.Module):
    """EDVR network structure for video super-resolution.Now only support X4 upsampling factor.
    ``Paper: EDVR: Video Restoration with Enhanced Deformable Convolutional Networks``

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_out_ch (int): Channel number of output image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_frame (int): Number of input frames. Default: 5.
        deformable_groups (int): Deformable groups. Defaults: 8.
        num_extract_block (int): Number of blocks for feature extraction.
            Default: 5.
        num_reconstruct_block (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame(参考帧索引). Frame counting from
            0. Default: Middle of input frames.
        hr_in (bool): Whether the input has high resolution(如果是视频去模糊任务输入的是HR图像，需经过下采样变成LR图像再输入EDVR中). Default: False.
        with_predeblur (bool): Whether has predeblur module.
            Default: False.
        with_tsa (bool): Whether has TSA module. Default: True.
    """

    def __init__(
            self,
            num_in_ch: int = 3,
            num_out_ch: int = 3,
            num_feat: int = 64,
            num_frame: int = 5,
            deformable_groups: int = 8,
            num_extract_block: int = 5,
            num_reconstruct_block: int = 10,
            center_frame_idx: int = None,
            hr_in: bool = False,
            with_predeblur: bool = False,
            with_tsa: bool = True
    ):
        super(EDVR, self).__init__()

        # 默认选取中间帧作为参考帧
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        self.hr_in = hr_in
        self.with_predeblur = with_predeblur
        self.with_tsa = with_tsa

        # 对输入的每一帧提取特征
        if self.with_predeblur:  # 包含去模糊步骤(注意视频去模糊任务(输入为HR)和视频超分任务(输入为LR)都可以使用去模糊模块)
            # 先进行去模糊(如果输入为 HR, 则去模糊模块的输出宽高会缩小为 1/4)
            self.predeblur = PredeblurModule(num_feat=num_feat, hr_in=self.hr_in)
            # 再提取特征
            self.conv_1x1 = nn.Conv2d(num_feat, num_feat, 1, 1)  # 输入输出宽高不变
        else:  # 不包含去模糊步骤
            # 直接进行特征提取
            # 根据卷积输出特征图大小计算公式:
            # W = (W-F+2P) / S + 1
            # H = (H-F+2P) / S + 1
            # 若 S = stride = 1，则 W = W-F+2P + 1, H = H-F+2P + 1
            # 若 F = kernel_size = 奇数，P = padding = F // 2
            # 则 -F+2P+1 = 0
            # 则下述卷积层输入和输出的 H、W 均保持不变
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)  # 输入输出宽高不变

        # 提取金字塔特征(适用于PCD模块)
        # L1 层: 使用 5 个残差块提取特征(make_layer串联起5个残差块)
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        # L2 层: 使用跨步卷积(stride=2)将特征图 H、W 缩小一半
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)  # 输出宽高减小为输入宽高的一半
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 输入输出宽高不变
        # L3 层: 使用跨步卷积(stride=2)将特征图 H、W 缩小一半
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)  # 输出宽高减小为输入宽高的一半
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 输入输出宽高不变

        # pcd 对齐模块
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)

        # tsa 特征融合模块
        if self.with_tsa:
            self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_frame, center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)  # 1x1卷积,输入输出宽高不变

        # 重建模块
        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)

        # 上采样操作前进行特征提取调整通道数 num_feat * 上采样倍率^2
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)  # 输入输出宽高不变
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)  # 输入输出宽高不变

        # 2倍上采样
        self.pixel_shuffle = nn.PixelShuffle(2)

        # 上采样完成后得到 hr 图像再进行两次特征提取
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)  # 输入输出宽高不变
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)  # 输入输出宽高不变

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        # 设 x.shape = (b,t,c,h,w), c 默认为 3
        # 若 x 是 HR 图像，则 shape = (b,t,c,H,W), 其中 H=4*h, W=4*w, c 默认为 3

        b, t, c, h, w = x.size()
        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, ('The height and width must be multiple of 16.')
        else:
            assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        # 提取中间帧(参考帧)
        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()  # (b,c,h,w)

        # extract features for each frame
        # 由于卷积操作的输入 shape 仅支持四个维度 len(x.shape) = 4, 默认为 (b,c,h,w)
        # 而视频序列 x 的输入 shape 有 5 个维度, 即 (b,t,c,h,w), 所以对 x 提取特征前需要先通过 view 调整 shape 为四个维度
        # L1
        if self.with_predeblur:  # 此时 x.shape = (b,t,c,H,W)
            feat_l1 = self.conv_1x1(self.predeblur(x.view(-1, c, h, w)))  # (b*t,num_feat,h,w)
            if self.hr_in:
                h, w = h // 4, w // 4
        else:
            feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))  # (b*t,num_feat,h,w)
        feat_l1 = self.feature_extraction(feat_l1)  # (b*t,num_feat,h,w)

        # 金字塔特征提取
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))  # (b*t,num_feat,h/2,w/2)
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))  # (b*t,num_feat,h/2,w/2)
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))  # (b*t,num_feat,h/4,w/4)
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))  # (b*t,num_feat,h/4,w/4)

        # 特征提取完毕, 重新调整为五个维度
        feat_l1 = feat_l1.view(b, t, -1, h, w)  # (b,t,num_feat,h,w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)  # (b,t,num_feat,h/2,w/2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)  # (b,t,num_feat,h/4,w/4)

        # PCD 对齐
        # 存放金字塔 (L1,L2,L3) 参考帧特征 list.一定要注意顺序
        ref_feat_l = [
            feat_l1[:, self.center_frame_idx, :, :, :].clone(),  # (b, num_feat, h, w)
            feat_l2[:, self.center_frame_idx, :, :, :].clone(),  # (b, num_feat, h/2, w/2)
            feat_l3[:, self.center_frame_idx, :, :, :].clone()  # (b, num_feat, h/4, w/4)
        ]
        # 存放对齐后的支持帧特征列表
        aligned_feat = []

        # 对每一个 i (即每一个支持帧), 都会存取其 L1,L2,L3 的特征，并与 ref_feat_l 一起送入对齐模块，实现了特征对齐
        # 每次对齐一个支持帧
        # 金字塔特征对齐中, 参考帧特征都是固定的, 每一次对不同 t 的支持帧特征进行对齐
        for i in range(t):
            # 存放当前 i 所对应的金字塔支持帧(包含参考帧与参考帧对齐)特征数据
            nbr_feat_l = [
                feat_l1[:, i, :, :, :].clone(),  # (b, num_feat, h, w)
                feat_l2[:, i, :, :, :].clone(),  # (b, num_feat, h/2, w/2)
                feat_l3[:, i, :, :, :].clone()  # (b, num_feat, h/4, w/4)
            ]
            # 支持帧 i 与参考帧进行对齐
            # 随着循环结束，存入了 5 帧的对齐特征
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))  # 共有 t 个列表元素,每个元素 shape = (b, num_feat, h, w)

        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, num_feat, h, w)

        # 不需要 tsa 模块(说明时间维度t已经消去了), 后续直接进行卷积特征提取(因此需要将维度调整为4维(注意消去的是时间维度t)才能输入卷积模块)
        if not self.with_tsa:
            aligned_feat = aligned_feat.view(b, -1, h, w)  # (b, num_feat*t, h, w)

        # 进行特征融合(tsa(输入为5维)或普通的卷积特征提取(输入为4维)),消去时间维度t
        feat = self.fusion(aligned_feat)  # (b, num_feat, h, w)

        # 进行图像重建
        out = self.reconstruction(feat)  # (b, num_feat, h, w)

        # 4倍上采样
        out = self.lrelu(self.pixel_shuffle(
            self.upconv1(out)  # (b, num_feat*4, h, w)
        ))  # (b, num_feat, 2*h, 2*w)
        out = self.lrelu(self.pixel_shuffle(
            self.upconv2(out)  # (b, 64*4, 2*h, 2*w)
        ))  # (b, 64, 4*h, 4*w)

        # 对放大后的图像再进行特征提取
        out = self.lrelu(self.conv_hr(out))  # (b, 64, 4*h, 4*w)
        out = self.conv_last(out)  # (b, 3, 4*h, 4*w)

        if self.hr_in:
            base = x_center  # (b,c,4*h,4*w), c默认为3
        else:
            # 由于后续需要进行残差连接，若输入的不是 hr 图像则需要将输入的参考帧(lr)放大4倍
            base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)  # (b,c,4*h,4*w), c默认为3

        # 进行残差连接: 学习的残差信息(out) + 参考帧(base)
        out += base  # (b, 3, 4*h, 4*w)
        return out  # (b, 3, 4*h, 4*w)
