import PIL.Image
from torch import nn
from os import path
import torch
from utils.utils import convert_image
from PIL import Image
from torchvision import transforms
from models.hit_sir_pro import HiT_SIR


# 获取输出变换(Tensor -> PIL)
def get_sr_transform():
    return transforms.ToPILImage()


# 获取输入变换(PIL -> Tensor)
def get_lr_transform():
    def lr_transform(img):
        return convert_image(img, 'pil', '[0,1]', None, None, None, None)

    return lr_transform


# 创建模型
def create_model():
    return HiT_SIR(
        is_mult_size_conv_feat_extract=True, is_channel_spatial_attn=True, is_fusion=True,
        # 注意 embed_dim 必须是 num_heads[i] * 2 的整数倍
        embed_dim=180, base_win_size=[8, 8], depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2, upsampler='nearest+conv', hier_win_ratios=[0.5, 1, 2, 4, 6, 8, 10, 12],
    ).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))


# 加载模型权重
def load_model_weights(pretrain_model_path: str, model: nn.Module, device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """
    pretrain_model_path: 模型路径
    model: 模型
    """
    if path.exists(pretrain_model_path):
        print('============ 加载模型权重 start ============')

        dic = torch.load(pretrain_model_path, map_location=device, weights_only=True)
        model.load_state_dict(dic['model'])
        start_epoch = dic['start_epoch'] + 1

        print(f'模型权重路径: {pretrain_model_path}, 训练 epoch 数: {start_epoch - 1}')
        print('============ 加载模型权重 end ============')
    else:
        print('模型权重路径不存在')


# 主函数
def main(img_path: str, device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """
    img_path: 需要重建的低分辨率图像(lr)路径
    """

    with torch.no_grad():
        # 创建模型
        model = create_model().eval()
        # 加载模型权重参数
        load_model_weights('weights/hitsir_pro_loss(l1)_mulsizeconvextract(True)_casa(True)_fusion_embed_dim(180)_len(depths)(6)_augment/best_psnr_ssim_lpips_model.pth', model)
        # 获取 lr 变换
        lr_transform = get_lr_transform()
        # 获取 sr 变换
        sr_transform = get_sr_transform()
        # 打开 lr 图片
        lr = None
        with PIL.Image.open(img_path, mode='r') as img_open:
            lr = img_open.convert('RGB')
        # 对 lr 图片进行变换
        lr = lr_transform(lr).to(device)
        # 输入模型重建出 sr
        sr: torch.Tensor = model(lr.unsqueeze(0)).clip(0, 1)
        # sr 变换为 PIL Image
        sr: PIL.Image = sr_transform(sr.squeeze(0))
        # 展示重建的图片
        sr.show()


# 根据高分辨率图像生成对应的双三次插值低分辨率图像并保存到磁盘上
def get_bicubic_lr(hr_path: str):
    hr = None
    with PIL.Image.open(hr_path, mode='r') as img_open:
        hr = img_open.convert('RGB')
    lr = hr.resize(
        (hr.width // 4,
         hr.height // 4),
        Image.BICUBIC
    )
    return lr.save(hr_path.split('.')[-2] + '_lr.png')


# main('data/test/RealSRSet+5images/00003.png')
main('data/test/RealSRSet+5images/0014.jpg')
# main('data/test/RealSRSet+5images/0030.jpg')
# main('data/test/RealSRSet+5images/oldphoto2.png')
# main('data/test/RealSRSet+5images/Lincoln.png')
