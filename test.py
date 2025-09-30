import os
import torch
import lpips
from PIL import Image
from torchvision.transforms import ToTensor

# 初始化LPIPS模型
loss_fn = lpips.LPIPS(net='vgg')


def calculate_lpips(img_path1, img_path2):
    # 读取图像
    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)

    # 将图像转换为张量
    img1_tensor = ToTensor()(img1).unsqueeze(0)
    img2_tensor = ToTensor()(img2).unsqueeze(0)

    # 计算LPIPS
    lpips_value = loss_fn(img1_tensor, img2_tensor)

    return lpips_value.item()


lpips = calculate_lpips('data/train/DIV2K_train_HR/0001.png', 'data/train/DIV2K_train_HR/0001.png')
print(lpips)
