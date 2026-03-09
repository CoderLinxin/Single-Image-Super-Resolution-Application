import random

import torch
from PIL import ImageFile
from experiments.hitsir_pro_experiment import hitsir_pro_experiment
from experiments.hitsir_pro_gan_experiment import hitsir_pro_gan_experiment
from experiments.hitsir_pro测试浅层特征提取2_experiment import hitsir_pro测试浅层特征提取2_experiment
from experiments.hitsir_pro测试浅层特征提取3_experiment import hitsir_pro测试浅层特征提取3_experiment
from experiments.hitsir_pro测试浅层特征提取3_gan_experiment import hitsir_pro测试浅层特征提取3_gan_experiment
from experiments.hitsir_pro测试浅层特征提取3_测试经典超分_experiment import hitsir_pro测试浅层特征提取3_测试经典超分_experiment
from experiments.hitsir_pro_gan测试浅层特征提取2_experiment import hitsir_pro_gan测试浅层特征提取2_experiment
from experiments.hitsir_experiment import hitsir_experiment
from experiments.rrdb_experiment import rrdb_experiment
from experiments.swinir_experiment import swinir_experiment


def main(model_name: str, is_test: bool, **kwargs):
    # 选择模型进行实验
    if model_name == 'hitsir_pro':
        hitsir_pro_experiment(is_test, **kwargs)
    if model_name == 'hitsir_pro_gan':
        hitsir_pro_gan_experiment(is_test, **kwargs)
    if model_name == 'hitsir_pro测试浅层特征提取2':
        hitsir_pro测试浅层特征提取2_experiment(is_test, **kwargs)
    if model_name == 'hitsir_pro_gan测试浅层特征提取2':
        hitsir_pro_gan测试浅层特征提取2_experiment(is_test, **kwargs)
    if model_name == 'hitsir':
        hitsir_experiment(is_test, **kwargs)
    if model_name == 'hitsir_pro测试浅层特征提取3':
        hitsir_pro测试浅层特征提取3_experiment(is_test, **kwargs)
    if model_name == 'hitsir_pro测试浅层特征提取3_gan':
        hitsir_pro测试浅层特征提取3_gan_experiment(is_test, **kwargs)
    if model_name == 'hitsir_pro测试浅层特征提取3_测试经典超分':
        hitsir_pro测试浅层特征提取3_测试经典超分_experiment(is_test, **kwargs)
    if model_name == 'rrdb':
        rrdb_experiment(is_test, **kwargs)
    if model_name == 'swinir':
        swinir_experiment(is_test, **kwargs)


if __name__ == '__main__':
    # 设置随机种子(应用阶段不需要,增加多样性)
    # torch.manual_seed(123)
    # torch.cuda.manual_seed(123)

    # 防止使用 pil 读取图像并进行相关处理(resize、convert)检测到图片数据出现截断而抛出异常
    # ImageFile.LOAD_TRUNCATED_IMAGES = True

    # 应用模型
    # main('hitsir_pro', is_test=False, is_augment=True, loss='l1',
    #      is_mult_size_conv_feat_extract=True, is_channel_spatial_attn=True, is_fusion=True,
    #      epochs=400, batch_size=8, test_model_name='best_psnr_ssim_lpips_model.pth',
    #      # 注意 embed_dim 必须是 num_heads[i] * 2 的整数倍
    #      embed_dim=180, base_win_size=[8, 8], depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
    #      mlp_ratio=2, upsampler='nearest+conv', hier_win_ratios=[0.5, 1, 2, 4, 6, 8, 10, 12],
    #      )

    # main('hitsir_pro_gan', is_test=False, is_augment=True, loss='l1',
    #      is_mult_size_conv_feat_extract=True, is_channel_spatial_attn=True, is_fusion=True,
    #      epochs=200, batch_size=2, test_model_name='best_psnr_ssim_lpips_model.pth',
    #      # 注意 embed_dim 必须是 num_heads[i] * 2 的整数倍
    #      embed_dim=180, base_win_size=[8, 8], depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
    #      mlp_ratio=2, upsampler='nearest+conv', hier_win_ratios=[0.5, 1, 2, 4, 6, 8, 10, 12],
    #      )

    # 训练 gan 时记得把生成器命名为 new_epoch_model.pth 然后放入 weights 文件夹中

    # 验证 div_2k 训练的 psnr,注意 batch_size 增大不会影响验证时间,不是用 div_2k 验证集的原因是计算 lpips 太久太久
    # main('hitsir_pro测试浅层特征提取2', is_test=True, is_augment=True, loss='l1',
    #      is_mult_size_conv_feat_extract=True, is_channel_spatial_attn=True, is_fusion=True,
    #      epochs=1000, batch_size=16, test_model_name='best_psnr_model.pth',
    #      # 注意 embed_dim 必须是 num_heads[i] * 2 的整数倍
    #      embed_dim=60, base_win_size=[8, 8], depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
    #      mlp_ratio=2, upsampler='pixelshuffledirect', hier_win_ratios=[0.5, 1, 2, 4, 6, 8],
    #      )

    # main('hitsir', is_test=False, is_augment=True)

    # main('hitsir_pro测试浅层特征提取3_测试经典超分', is_test=True, is_augment=True, is_bsrgan_degrade=False, loss='l1',
    #      is_mult_size_conv_feat_extract=True, is_channel_spatial_attn=True, is_fusion=True,
    #      epochs=2000, batch_size=16, test_model_name='best_psnr_model.pth',
    #      # 注意 embed_dim 必须是 num_heads[i] * 2 的整数倍
    #      embed_dim=60, base_win_size=[8, 8], depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
    #      mlp_ratio=2, upsampler='pixelshuffledirect', hier_win_ratios=[0.5, 1, 2, 4, 6, 8],
    #      )

    # main('rrdb', is_test=False, is_augment=True)
    # main('swinir', is_test=True, is_augment=True)

    # 1~113: batch_size = 4
    # 114~: batch_size = 2
    # 272~300: 使用梯度累加
    # 340~: 使用其他数据集训练
    # main('hitsir_pro测试浅层特征提取3', is_test=False, is_augment=True, loss='l1',
    #      is_gradient_accurate=True, gradient_accurate_batch_size=8,
    #      is_mult_size_conv_feat_extract=True, is_channel_spatial_attn=True, is_fusion=True,
    #      epochs=400, batch_size=4, test_model_name='best_psnr_ssim_lpips_model.pth',
    #      # 注意 embed_dim 必须是 num_heads[i] * 2 的整数倍
    #      embed_dim=180, base_win_size=[8, 8], depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
    #      mlp_ratio=2, upsampler='nearest+conv', hier_win_ratios=[0.5, 1, 2, 4, 6, 8, 10, 12],
    #      )

    # 1~110: 动漫图像现实图像混杂训练
    # 111~150: 仅使用现实图像训练
    # 150~200: 动漫图像现实图像混杂训练
    main('hitsir_pro测试浅层特征提取3_gan', is_test=False, is_augment=True, loss='l1',
         is_gradient_accurate=False, gradient_accurate_batch_size=0,
         is_mult_size_conv_feat_extract=True, is_channel_spatial_attn=True, is_fusion=True,
         epochs=200, batch_size=2, test_model_name='best_psnr_ssim_lpips_model.pth',
         # 注意 embed_dim 必须是 num_heads[i] * 2 的整数倍
         embed_dim=180, base_win_size=[8, 8], depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
         mlp_ratio=2, upsampler='nearest+conv', hier_win_ratios=[0.5, 1, 2, 4, 6, 8, 10, 12],
         )
