import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import warnings

from basicsr.archs.arch_util import flow_warp
from basicsr.archs.basicvsr_arch import ConvResidualBlocks
from basicsr.archs.spynet_arch import SpyNet
from basicsr.ops.dcn import ModulatedDeformConvPack
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class BasicVSRPlusPlus(nn.Module):
    """BasicVSR++ network structure.

    Support either x4 upsampling or same size output. Since DCN is used in this
    model, it can only be used with CUDA enabled. If CUDA is not enabled,
    feature alignment will be skipped. Besides, we adopt the official DCN
    implementation and the version of torch need to be higher than 1.9.

    ``Paper: BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment``

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10. 表示可变形卷积模块中 offset 的残差(C0)缩放系数
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
        cpu_cache_length (int, optional): When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100.
    """

    def __init__(
            self,
            mid_channels=64,
            num_blocks=7,
            max_residue_magnitude=10,
            is_low_res_input=True,
            spynet_path=None,
            cpu_cache_length=100
    ):

        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length

        # optical flow
        self.spynet = SpyNet(spynet_path)

        # feature extraction module(xi经过特征提取网络得到gi)
        if is_low_res_input:  # 输入的是低分辨率图像
            self.feat_extract = ConvResidualBlocks(3, mid_channels, 5)  # 输入输出宽高不变,输出通道数变为 mid_channels
        else:  # 输入的是与高分辨率图像相同大小的图像,则需要先进行4倍下采样
            self.feat_extract = nn.Sequential(
                # 输出宽高变为输入宽高的一半
                nn.Conv2d(3, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                # 输出宽高变为输入宽高的一半
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ConvResidualBlocks(mid_channels, mid_channels, 5)  # 输入输出宽高不变
            )

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.deform_align[module] = SecondOrderDeformableAlignment(
                    # DCN 模块主要对 cat[fi-1,fi-2] 应用 oi、mi 进行可变形卷积采样
                    # in_channel 对应 cat[fi-1,fi-2] 的通道数 2*mid_channels
                    2 * mid_channels,
                    # DCN 输出为 fi_^, 通道数为 mid_channels
                    mid_channels,
                    # 可变形卷积核默认为 (3,3)
                    3,
                    padding=1,  # 确保输入输出宽高不变
                    deformable_groups=16,
                    max_residue_magnitude=max_residue_magnitude
                )
            # 对 cat[fij-1,fij_^] 进行残差块调整输出 fij
            # 使用的是 BasicVSR 的策略(所有分支的输出进行融合)而不是 IconVSR 的策略(后向->前向->后向->前向,不需要融合)
            # 第一个后向分支的残差块输入 cat(gi, fij_^), shape = (n,2*mid_channels,h,w)
            # 第二个前向分支的残差块输入 cat(gi, fij-1, fij_^), shape = (n,3*mid_channels,h,w)
            # 第三个后向分支的残差块输入 cat(gi, fij-1, fij-2, fij_^), shape = (n,4*mid_channels,h,w)
            # 第四个前向分支的残差块输入 cat(gi, fij-1, fij-2, fij-3, fij_^), shape = (n,5*mid_channels,h,w)
            self.backbone[module] = ConvResidualBlocks((2 + i) * mid_channels, mid_channels, num_blocks)

        # upsampling module
        # 对于某个特定时间步 i 来说,输入的是 gi 和所有分支的输出 fij, 共 5 个特征图
        self.reconstruction = ConvResidualBlocks(5 * mid_channels, mid_channels, 5)

        self.upconv1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)  # 输入输出宽高不变
        self.upconv2 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)  # 输入输出宽高不变

        self.pixel_shuffle = nn.PixelShuffle(2)

        # 重建的最后调整
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)  # 输入输出宽高不变
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)  # 输入输出宽高不变
        # 使用原始图像上采样作为残差连接
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        if len(self.deform_align) > 0:
            self.is_with_alignment = True
        else:
            self.is_with_alignment = False
            warnings.warn('Deformable alignment module is not added. '
                          'Probably your CUDA is not configured correctly. DCN can only '
                          'be used with CUDA enabled. Alignment is skipped now.')

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:  # 时间 t 为 偶数，则说明可以进行拆分成 2 部分
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            # 判断输入序列是否经过时间 t 的翻转(数据增强)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:  # 判断 L2 范数是否为 0
                self.is_mirror_extended = True

    # 与 BasicVSR 实现一致
    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the flows used for forward-time propagation \
                (current to previous). 'flows_backward' corresponds to the flows used for backward-time \
                propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        # 求反向传播的光流: 反向传播的光流估计模块输入为 xi、xi+1 输出光流估计值 fxi->xi+1,
        # 目的是为了让 xi+1(或者说hi+1) 对齐 xi,后续 warp 的输入为 fxi->xi+1、hi+1
        # xi  : [    第0帧,               第1帧,          ...,      倒数第二帧(第n-2帧) ]
        # xi+1: [    第1帧,               第2帧,          ...,        尾帧(第n-1帧)   ]
        # 输出: [(第0帧->第1帧)的光流, (第1帧->第2帧)的光流, ..., (倒数第二帧->尾帧)的光流]
        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)  # (n, t - 1, 2, h, w)

        if self.is_mirror_extended:
            # 此时 flows_forward、flows_backward 的 shape 应为 # (n, 2*t - 1, 2, h, w)
            flows_forward = flows_backward.flip(1)
        else:
            # 求正向传播的光流: 正向传播的光流估计模块输入为 xi、xi-1 输出光流估计值 fxi->xi-1,
            # 目的是为了让 xi-1(或者说hi-1) 对齐 xi,后续 warp 的输入为 fxi->xi-1、hi-1
            # xi  : [        第1帧,            第2帧,         ...,        尾帧(第n-1帧)   ]
            # xi-1: [        第0帧,            第1帧,         ...,    倒数第二帧(第n-2帧) ]
            # 输出: [(第1帧->第0帧)的光流, (第2帧->第1帧)的光流, ..., (尾帧->倒数第二帧)的光流]
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward  # (n, t - 1, 2, h, w), (n, t - 1, 2, h, w)

    def propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
                feats['spatial'] = [g0,g1,...gi,...], 有 t 个元素, gi.shape = (n,c,h,w)
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
                backward_flows or forward_flows
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated \
                features. Each key in the dictionary corresponds to a \
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()  # 这里的 t 实际上是 t-1

        frame_idx = range(0, t + 1)  # [0,1,...,t-1]
        flow_idx = range(-1, t)  # [-1,0,1,...,t-2],实际只遍历[0,1,...,t-2]
        mapping_idx = list(range(0, len(feats['spatial'])))  # [0,1,...,t-1]
        # 这里主要是针对使用了 flip 进行数据增强的遍历情况, 此时的 frame_idx 应为 # [0,1,...,t-1,...,2t-1]
        mapping_idx += mapping_idx[::-1]  # 将倒序的 mapping_idx 也进行合并: [0,1,...,t-1, t-1,...,1,0]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]  # 如果是 backward 分支,则需要从尾帧遍历到首帧 [t-1,...,0]
            # 如果是 backward 分支,光流也需要逆序遍历 [t-1,t-2,...,0](其中尾帧不需要对齐)
            # 实际只遍历[t-2,...,0]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)  # (n,mid_channels,h,w)
        # backward、forward 的遍历合并为下述一个 for 循环
        for i, idx in enumerate(frame_idx):
            # 取出 gi
            feat_current = feats['spatial'][mapping_idx[idx]]  # (n,mid_channels,h,w)
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
            # second-order deformable alignment
            # 无论是 forward 还是 backward,第一次遍历时都不需要对齐(即i>=1才进行对齐),
            # forward对应首帧不需要对齐,backward对应尾帧不需要对齐,对齐是在第二次遍历才开始的
            if i > 0 and self.is_with_alignment:
                # 取出光流 fxi->xi-1 或 fxi->xi+1
                flow_n1 = flows[:, flow_idx[i], :, :, :]  # (n, 2, h, w)
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                # 首先进行一阶格点传播: 将 fi+1 对齐到 xi 上或将 fi-1 对齐到 xi 上得到 fi-1_bar 或 fi+1_bar
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))  # (n,mid_channels,h,w)

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                # 第三次以上遍历时才会存在二阶格点传播
                if i > 1:  # second-order features
                    # 取出 fi-2 或 fi+2
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()
                    # 取出光流 fxi-1->xi-2 或 fxi+1->xi+2
                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    # 注意不能直接将 flow_n1 和 flow_n2 相加,因为同一位置上的光流并不是对应的
                    # 首先需要通过 warp 操作将 flow_n2 对齐到 flow_n1 上(变换到flow_n1的坐标系中)
                    # 对于某一个坐标(x0,y0)而言, warp 操作相当于通过 flow_n1 从 xi 走到 xi-1 或 xi+1,
                    # 这对应到 flow_n2 的某个坐标(x1,y1)上,然后在 flow_n2 的(x1,y1)坐标上进行采样,
                    # 从而得到结果图上坐标(x0,y0)表示从 xi-1 走到 xi-2 或从 xi+1 走到 xi+2 的光流信息
                    # 这样一来才能将其与 flow_n1 进行相加最终得到光流 fxi->xi-2 或 fxi->xi+2
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    # 将 fi-2 对齐到 xi 上或将 fi+2 对齐到 xi 上得到 fi-2_bar 或 fi+2_bar
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))  # (n,mid_channels,h,w)

                # flow-guided deformable convolution
                # cat[gi,fi-1_bar,fi-2_bar] 或 cat[gi,fi+1_bar,fi+2_bar]
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)  # (n,3*mid_channels,h,w)
                # cat[fi-1,fi-2] 或 cat[fi+1,fi+2]
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                # fi_^=D(c(fi-1,fi-2),oi,mi) 或 fi_^=D(c(fi+1,fi+2),oi,mi)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)

            # concatenate and residual blocks
            # 由于是顺序遍历四个分支
            # 因此在遍历 feats 时,只要对应的 key 有值 'backward_1', 'forward_1', 'backward_2' 说明对应的分支都已经传播完毕了
            # feats[module_name] 只在分支传播完毕后进行赋值(在下面可以看到)
            # [gi, 其他分支的fij-1,fij+1,..., fi_^], 这里 fij 统一用 fi 表示, fij_^ 统一用 fi_^ 表示
            feat = [feat_current] + [feats[k][idx] for k in feats if k not in ['spatial', module_name]] + [feat_prop]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            # backward_1: cat[gi,fi_^], shape = (n,2*mid_channels,h,w)
            # forward_1: cat[gi,fij-1,fi_^], shape = (n,3*mid_channels,h,w)
            # backward_2: cat[gi,fij-1,fij-2,fi_^], shape = (n,4*mid_channels,h,w)
            # forward_2: cat[gi,fij-1,fij-2,fij-3,fi_^], shape = (n,5*mid_channels,h,w)
            feat = torch.cat(feat, dim=1)
            # 经过一系列残差块提取特征得到 fi
            # feat_prop(本次遍历对应 fi) 下一次遍历就变成了 fi-1 或 fi+1
            feat_prop = feat_prop + self.backbone[module_name](feat)
            # 添加到 feats[module_name] 中
            # 后向对应: [..,fi,fi-1,..,f0]
            # 前向对应: [f0,..,fi-1,fi..,]
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        # 将后向的 feats 进行逆序排列,使得所有分支输出的 feats 都对应起来
        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        # feats = {
        #   'spatial':     [g0,    g1,  ..., gi,  ...], t 个元素,每个元素 shape = (n,mid_channels,h,w)
        #   'backward_1': [fi0j0,fi1j0, ...,fij0, ...], t 个元素,每个元素 shape = (n,mid_channels,h,w)
        #   'forward_1':  [fi0j1,fi1j1, ...,fij1, ...], t 个元素,每个元素 shape = (n,mid_channels,h,w)
        #   'backward_2': [fi0j1,fi1j2, ...,fij2, ...], t 个元素,每个元素 shape = (n,mid_channels,h,w)
        #   'forward_2':  [fi0j3,fi1j3, ...,fij3, ...], t 个元素,每个元素 shape = (n,mid_channels,h,w)
        # }
        return feats  # 这里其实也可以不需要 return

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propagation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        outputs = []
        num_outputs = len(feats['spatial'])  # t(没使用flip数据增强的情况下)

        mapping_idx = list(range(0, num_outputs))
        # 用作遍历的 index(包含了使用flip和不使用flip提供数据的情况)
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            # 取出当前时刻 i, 所有分支的输出 fij
            # [fij0,fij1,fij2,fij3]
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            # [gi,fij0,fij1,fij2,fij3]
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)  # (n,5*middle_channel,h,w)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)  # (n,middle_channel,h,w)
            hr = self.lrelu(self.pixel_shuffle(
                self.upconv1(hr)  # (n,4*middle_channel,h,w)
            ))  # (n,middle_channel,2*h,2*w)
            hr = self.lrelu(self.pixel_shuffle(
                self.upconv2(hr)  # (n,4*middle_channel,2*h,2*w)
            ))  # (n,middle_channel,4*h,4*w)
            hr = self.lrelu(self.conv_hr(hr))  # (n,64,4*h,4*w)
            hr = self.conv_last(hr)  # (n,3,4*h,4*w)

            # 与原始输入的图像 xi 作残差连接
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])  # (n,3,4*h,4*w)
            else:
                hr += lqs[:, i, :, :, :]  # (n,3,4*h,4*w)

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)  # t 个 元素,每个元素 (n,3,4*h,4*w)

        return torch.stack(outputs, dim=1)  # (n, t, 3, 4*h, 4*w)

    def forward(self, lqs):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        # 确保输入序列为低分辨率图像,注意 lqs_downsample(相当于确定了分辨率的 xi) 只作用于重建模块后面的残差连接
        # 特征提取针对的是 lqs
        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            # 手动进行插值缩放为 1/4
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25, mode='bicubic'
            ).view(n, t, c, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        feats = {}
        # 对 lqs(不确定分辨率的xi) 进行特征提取(得到 gi)
        # compute spatial features
        # feats['spatial'] 保存了输入序列的特征列表 [g0,g1,...,gi,...]
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].insert(0, feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs
        # 光流估计网络 spynet 中的金字塔最多会出现 1/64 的下采样,故要求 lqs_downsample 宽高至少 >= 64
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        # 根据 list[xi] 计算光流
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # feature propgation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                else:
                    flows = flows_forward

                feats = self.propagate(feats, flows, module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        # feats 包含了所有的 gi 以及所有分支所有时刻的输出 fij
        return self.upsample(lqs_downsample, feats)  # (n,3,4*h,4*w)


class SecondOrderDeformableAlignment(ModulatedDeformConvPack):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        # offsets 和 modulation masks 的学习是共享的，都使用的是 conv_offset
        # 对最后的输出进行通道上的split就能得到offsets和modulation masks
        self.conv_offset = nn.Sequential(
            # 输入为 cat[gi,fi-1_bar,fi-2_bar,si->i-1,si->i-2] 或 cat[gi,fi+1_bar,fi+2_bar,si->i+1,si->i+2]
            # 其中 gi.shape=fi-1_bar.shape=fi-2_bar.shape=(n,middle_channel,h,w)
            # si->i-1.shape=si->i-2.shape=(n,2,h,w)
            # self.out_channels在构造时被赋值为middle_channel,因此输入通道数为 3*middle_channel+2*2
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),  # 输入输出宽高不变
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),  # 输入输出宽高不变
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),  # 输入输出宽高不变
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # 这里对可变形卷积做了分组卷积处理,最终生成 deformable_groups 组 offset 和 modulation masks
            # 这也就意味着将 offset 和 modulation masks 应用 DCN 上也是分成 deformable_groups 组进行的
            # (学习DCN理论时,是输入特征图的每个通道都使用一组 offset 和 modulation masks,也就是deformable_groups=in_channels的情况)
            # (64, 16(deformable_groups) * 27(2*9+9=2N+N) = 288+144, 3, 1, 1)
            # 最后一个卷积模块的输出在通道上可拆分为 offset 和 modulation masks
            nn.Conv2d(self.out_channels, 27 * self.deformable_groups, 3, 1, 1),  # 输入输出宽高不变
        )

        self.init_offset()

    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        _constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        """
        Args:
            x: cat[fi-1,fi-2], shape=(b,2*middle_channels,h,w)
            extra_feat: cat[gi,fi-1_bar,fi-2_bar], shape=(b,3*middle_channels,h,w)
            flow_1: si->i-1, shape = (b,2,h,w)
            flow_2: si->i-2, shape = (b,2,h,w)
        """
        # cat[gi,fi-1_bar,fi-2_bar,si->i-1,si->i-2] 或 cat[gi,fi+1_bar,fi+2_bar,si->i+1,si->i+2]
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)  # shape=(b,3*middle_channels+4,h,w)
        out = self.conv_offset(extra_feat)  # shape=(b,deformable_groups*3N,h,w)
        # o1(x方向).shape=o2(y方向).shape=mask.shape=(b, deformable_groups*N, h, w)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # 对 o1、o2 进行tanh激活以及乘以对应的残差系数
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))  # (b, deformable_groups*2N, h, w)
        # offset_1.shape = offset_2.shape = (b, deformable_groups*N, h, w)
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)  # offset_1.shape=offset_2.shape=(b,deformable_groups*N,h,w)

        # 如果直接将光流flow_1、flow_2与offset相加,最终进行concat得到的目标shape=(b,deformable_groups*4*N,h,w)
        # 但我们需要的目标shape=(b,deformable_groups*2*N,h,w)
        # 因此才将offset按通道拆分再分别与flow_1和flow_2相加
        # 这里只能将flow_1理解成x=cat[fi-1,fi-2]在x方向的光流
        # 将flow_2理解成x=cat[fi-1,fi-2]在y方向的光流
        # (其实这里感觉还是太牵强)

        # oi->i-1 = si->i-1 + C0(cat[gi,fi-1_bar,fi-2_bar,si->i-1,si->i-2])
        # si->i-1=flow_1: (n,2,h,w)->(n,deformable_groups*N,h,w)
        offset_1 = flow_1.repeat(1, offset_1.size(1) // 2, 1, 1) + offset_1  # (b, deformable_groups*N, h, w)
        # oi->i-2 = si->i-2 + C0(cat[gi,fi-1_bar,fi-2_bar,si->i-1,si->i-2])
        offset_2 = flow_2.repeat(1, offset_2.size(1) // 2, 1, 1) + offset_2  # (b, deformable_groups*N, h, w)
        # 得到 oi = cat(oi->i-1,oi->i-2)
        # shape = (b, deformable_groups*2*N, h, w)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        # 进行DCN卷积
        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                             self.dilation, mask)
