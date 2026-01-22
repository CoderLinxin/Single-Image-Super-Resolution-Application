import glob
import os
import time
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from datasets.dataset import Dataset
from utils.utils import convert_image, AverageMeter
from configs.dataset_config import DatasetConfig
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

to_pil_transform = transforms.ToPILImage()
to_tensor_transform = transforms.ToTensor()

# 计算双三次插值图片与真实图片之间的 psnr 和 ssim
test_data_name = 'data/test/Set14_2'
hrs_path = glob.glob(test_data_name + '/hr' + '/*')
lrs_path = glob.glob(test_data_name + '/lr' + '/*')

test_set_psnr = AverageMeter()
test_set_ssim = AverageMeter()

for i, hr_path in enumerate(hrs_path):
    hr = None
    lr = None
    with Image.open(hrs_path[i], mode='r') as img_open1:
        hr = img_open1.convert('RGB')
    with Image.open(lrs_path[i], mode='r') as img_open2:
        lr = img_open2.convert('RGB')

    # 使用 pil 的双三次插值放大
    lr = lr.resize((hr.width, hr.height), Image.BICUBIC)
    # 转回 tensor
    lr = to_tensor_transform(lr).unsqueeze(0)
    hr = to_tensor_transform(hr).unsqueeze(0)

    # 转换为 ycbcr 中的 y 通道
    lr_y = convert_image(  # (H, W), in y-channel
        lr, source='[0,1]', target='y-channel', is_test=True, is_lr=False, is_lr_amplify=False, scaling_factor=4
    ).squeeze(0)

    hr_y = convert_image(  # (H, W), in y-channel
        hr, source='[0,1]', target='y-channel', is_test=True, is_lr=False, is_lr_amplify=False, scaling_factor=4
    ).squeeze(0)

    # 根据 y 通道计算 PSNR 和 SSIM
    psnr = peak_signal_noise_ratio(
        lr_y.cpu().numpy(), hr_y.cpu().numpy(),
        data_range=1.
    )
    ssim = structural_similarity(
        lr_y.cpu().numpy(), hr_y.cpu().numpy(),
        data_range=1.,
        gaussian_weights=True,
    )

    # 统计 PSNR 和 SSIM
    test_set_psnr.update(psnr, 1)
    test_set_ssim.update(ssim, 1)

print(f'psnr:{test_set_psnr.avg},ssim:{test_set_ssim.avg}')
