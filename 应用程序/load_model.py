import PIL.Image
from torch import nn
from os import path
import torch
from utils.utils import convert_image
from PIL import Image
from torchvision import transforms
from models.hit_sir_pro测试浅层特征提取3 import HiT_SIR

# 定义模型
model = HiT_SIR(
    is_mult_size_conv_feat_extract=True, is_channel_spatial_attn=True, is_fusion=True,
    # 注意 embed_dim 必须是 num_heads[i] * 2 的整数倍
    embed_dim=180, base_win_size=[8, 8], depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2, upsampler='nearest+conv', hier_win_ratios=[0.5, 1, 2, 4, 6, 8, 10, 12],
).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')).eval()

# 加载模型参数
print('============ 加载模型权重 start ============')
pretrain_model_path = '../weights/hitsir_pro测试浅层特征提取3_gan_loss(l1)_mulsizeconvextract(True)_casa(True)_fusion_embed_dim(180)_len(depths)(6)_augment/new_epoch_model.pth'
# pretrain_model_path = '../weights/hitsir_pro测试浅层特征提取3_gan_loss(l1)_mulsizeconvextract(True)_casa(True)_fusion_embed_dim(180)_len(depths)(6)_augment/epoch=150(通用超分)/new_epoch_model.pth'
dic = torch.load(
    pretrain_model_path,
    map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), weights_only=True
)
model.load_state_dict(dic['model'])
start_epoch = dic['start_epoch'] + 1
print(f'模型权重路径: {pretrain_model_path}, 训练 epoch 数: {start_epoch - 1}')
print('============ 加载模型权重 end ============')


# 模型前向推理
def inference(lr_img: torch.Tensor):
    """
    :param lr_img: (c,h,w)
    :return:
    """

    with torch.no_grad():
        sr: torch.Tensor = model(lr_img.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')).unsqueeze(0)).clip(0, 1)
        return sr

