import torch


# 对图像进行分块处理
def image_block_handle(
        img: torch.Tensor,
        inference_function,
        pre_callback,
        per_block_callback,
        tile: int = 256,
        tile_overlap: int = 32
):
    """
    :param img: 输入图像 (c,h,w)
    :param inference_function: 前向推理函数
    :param pre_callback: 执行分块处理逻辑前的回调(总分块数)
    :param per_block_callback: 每个分块处理完毕的回调(总分块数,当前处理的分块索引(从0开始))
    :param tile: 分块大小,每个块的边长
    :param tile_overlap: 相邻切块之间的重叠区域
    :return 输出超分结果 (c,4*h,4*w)
    """
    window_size = 8  # 确保分块大小是 window_size 的倍数

    # 输入图像处理逻辑
    c, h, w = img.size()  # 获取输入图像的高度、宽度和通道数
    tile = min(tile, h, w)  # 切块的大小，不能超过图像的高度和宽度
    assert tile % window_size == 0, "tile size should be a multiple of window_size"  # 确保切块大小是 window_size 的倍数
    stride = tile - tile_overlap  # 每次切块移动的步长，确保切块间存在重叠区域    256-32 = 224

    # 切块会从图像的左上角开始，按步长（224）逐步生成切块的位置。对于 1024x1280 的图像，这样会生成多个 tile，每次切块的起始位置会偏移 224 像素
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]  # 高度方向的切块索引
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]  # 宽度方向的切块索引

    # 创建用于存储结果的张量
    E = torch.zeros((1, c, h * 4, w * 4)).to(img.device)  # 存储最终结果
    W = torch.zeros_like(E)  # 存储权重

    # 需要处理的分块总数
    block_count = len(h_idx_list) * len(w_idx_list)
    current_block_index = 0

    pre_callback(block_count)

    # 遍历所有切块
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            # 提取当前切块
            in_patch = img[:, h_idx:h_idx + tile, w_idx:w_idx + tile]  # (c, tile, tile)
            # 对当前块进行超分
            out_patch = inference_function(in_patch)  # (1, c, tile*4, tile*4)
            # 生成与切块输出同形状的权重(以便后续对重叠区域的像素求平均)
            out_patch_mask = torch.ones_like(out_patch)
            # 累加结果到对应位置
            E[:, :, h_idx * 4:(h_idx + tile) * 4, w_idx * 4:(w_idx + tile) * 4] += out_patch
            W[:, :, h_idx * 4:(h_idx + tile) * 4, w_idx * 4:(w_idx + tile) * 4] += out_patch_mask
            # 触发回调
            per_block_callback(block_count, current_block_index)
            current_block_index += 1

    # 计算最终结果，防止除零
    output = torch.where(W != 0, E / W, E)

    return output, block_count
