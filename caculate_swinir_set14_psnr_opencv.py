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
import cv2 as cv
import math

to_pil_transform = transforms.ToPILImage()
to_tensor_transform = transforms.ToTensor()

test_data_name = 'data/test/Set14_2'
hrs_path = glob.glob(test_data_name + '/*')

test_set_psnr = AverageMeter()
test_set_ssim = AverageMeter()

count = 0


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


for hr_path in hrs_path:
    hr = cv.imread(hr_path, cv.IMREAD_COLOR).astype(np.float32) / 255.
    h, w, _ = hr.shape
    hr = hr[:h - h % 4, :w - w % 4, :]
    hr = (hr * 255.0).round().astype(np.uint8)  # float32 to uint8
    hr = np.squeeze(hr)
    hr_y = bgr2ycbcr(hr.astype(np.float32) / 255.) * 255.

    sr_path = hr_path.replace('Set14_2', 'Set14_swinir_sr')
    # names = os.path.basename(sr_path).split('.')[0].split('_')
    # sr_path = sr_path.replace(os.path.basename(sr_path), names[0] + names[1] + 'x4_SwinIR' + '.png')
    sr_path = sr_path.replace(os.path.basename(sr_path), os.path.basename(sr_path).split('.')[0] + 'x4_SwinIR' + '.png')
    sr = cv.imread(sr_path, cv.IMREAD_COLOR).astype(np.float32) / 255.
    h, w, _ = sr.shape
    sr = sr[:h - h % 4, :w - w % 4, :]
    sr = (sr * 255.0).round().astype(np.uint8)  # float32 to uint8
    sr = np.squeeze(sr)
    sr_y = bgr2ycbcr(sr.astype(np.float32) / 255.) * 255.

    psnr_y = calculate_psnr(hr_y, sr_y, border=4)
    ssim_y = calculate_ssim(hr_y, sr_y, border=4)

    # 统计 PSNR 和 SSIM
    test_set_psnr.update(psnr_y, 1)
    test_set_ssim.update(ssim_y, 1)

    count += 1

print(f'数据集大小:{count}')
print(f'psnr:{test_set_psnr.avg},ssim:{test_set_ssim.avg}')
