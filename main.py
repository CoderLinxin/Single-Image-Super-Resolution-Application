import random

import torch
from PIL import ImageFile
from experiments.hitsir_pro_experiment import hitsir_pro_experiment
from experiments.hitsir_pro_gan_experiment import hitsir_pro_gan_experiment


def main(model_name: str, is_test: bool, **kwargs):
    # 选择模型进行实验
    if model_name == 'hitsir_pro':
        hitsir_pro_experiment(is_test, **kwargs)
    if model_name == 'hitsir_pro_gan':
        hitsir_pro_gan_experiment(is_test, **kwargs)


if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    # 防止使用 pil 读取图像并进行相关处理(resize、convert)检测到图片数据出现截断而抛出异常
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # 应用模型
    main('hitsir_pro', is_test=False, is_augment=True, loss='l1',
         is_mult_size_conv_feat_extract=True, is_channel_spatial_attn=True, is_fusion=True,
         epochs=500, batch_size=2, test_model_name='best_psnr_ssim_lpips_model.pth',
         # 注意 embed_dim 必须是 num_heads[i] * 2 的整数倍
         embed_dim=180, base_win_size=[8, 8], depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
         mlp_ratio=2, upsampler='nearest+conv', hier_win_ratios=[0.5, 1, 2, 4, 6, 8, 10, 12],
         )

    # main('hitsir_pro_gan', is_test=False, is_augment=True, loss='l1',
    #      is_mult_size_conv_feat_extract=True, is_channel_spatial_attn=True, is_fusion=True,
    #      epochs=300, batch_size=2, test_model_name='best_psnr_ssim_lpips_model.pth',
    #      # 注意 embed_dim 必须是 num_heads[i] * 2 的整数倍
    #      embed_dim=180, base_win_size=[8, 8], depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
    #      mlp_ratio=2, upsampler='nearest+conv', hier_win_ratios=[0.5, 1, 2, 4, 6, 8, 10, 12],
    #      )

    # 训练 gan 时记得把生成器命名为 new_epoch_model.pth 然后放入 weights 文件夹中
