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
from utils.utils import ImageTransforms, get_augment_param, augment

to_pil_transform = transforms.ToPILImage()
to_tensor_transform = transforms.ToTensor()

test_data_name = 'data/test/Set14'
hrs_path = glob.glob(test_data_name + '/*')

test_set_psnr = AverageMeter()
test_set_ssim = AverageMeter()

hr_transform = ImageTransforms(
    split='eval|test',
    crop_size=64,
    scaling_factor=4,
    img_type='[0,1]',
    is_lr=False,
    is_lr_amplify=False
)

# 国际惯例测试实现: 从左上角开始裁剪能被 4 整除的最大的图像,接着裁剪四周边缘(上下左右) 4 个像素,最后才进行 psnr 的计算
for hr_path in hrs_path:
    hr = None
    sr = None
    with Image.open(hr_path, mode='r') as img_open1:
        hr = img_open1.convert('RGB')
        if hr.width % 4 != 0 or hr.height % 4 != 0:
            print(f'h:{hr.width},w:{hr.height}不能被4整除')
        hr, _ = hr_transform(hr)
        hr = hr.clip(0, 1)
    sr_path = hr_path.replace('Set14', 'Set14_swinir_sr')
    # names = os.path.basename(sr_path).split('.')[0].split('_')
    # sr_path = sr_path.replace(os.path.basename(sr_path), names[0] + names[1] + 'x4_SwinIR' + '.png')
    sr_path = sr_path.replace(os.path.basename(sr_path), os.path.basename(sr_path).split('.')[0] + 'x4_SwinIR' + '.png')
    with Image.open(sr_path, mode='r') as img_open2:
        sr = img_open2.convert('RGB')
        sr, _ = hr_transform(sr)
        sr = sr.clip(0, 1)

    sr = sr[..., 4:-4, 4:-4]
    hr = hr[..., 4:-4, 4:-4]

    # 转回 tensor
    sr = sr.unsqueeze(0)
    hr = hr.unsqueeze(0)

    # 转换为 ycbcr 中的 y 通道
    sr_y = convert_image(  # (H, W), in y-channel
        sr, source='[0,1]', target='y-channel', is_test=True, is_lr=False, is_lr_amplify=False, scaling_factor=4
    ).squeeze(0)

    hr_y = convert_image(  # (H, W), in y-channel
        hr, source='[0,1]', target='y-channel', is_test=True, is_lr=False, is_lr_amplify=False, scaling_factor=4
    ).squeeze(0)

    # 根据 y 通道计算 PSNR 和 SSIM
    psnr = peak_signal_noise_ratio(
        sr_y.cpu().numpy(), hr_y.cpu().numpy(),
        data_range=1.
    )
    ssim = structural_similarity(
        sr_y.cpu().numpy(), hr_y.cpu().numpy(),
        data_range=1.,
        gaussian_weights=True,
    )

    # 统计 PSNR 和 SSIM
    test_set_psnr.update(psnr, 1)
    test_set_ssim.update(ssim, 1)

print(f'psnr:{test_set_psnr.avg},ssim:{test_set_ssim.avg}')
