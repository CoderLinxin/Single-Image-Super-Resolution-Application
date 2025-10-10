import glob
import os

import torch
from torch.utils import data
from configs.dataset_config import DatasetConfig
from utils.utils import ImageTransforms, get_augment_param, augment
from abc import ABCMeta
from PIL import Image
from 参考资料.KAIR_master.utils import utils_blindsr
from torchvision import transforms
import numpy as np
import cv2 as cv


class Dataset(data.Dataset):
    __meta_class__ = ABCMeta

    def __init__(
            self,
            config: DatasetConfig,
            data_folder: str,
    ):
        """
        :param config: 数据集配置
        :param data_folder: 数据集路径
        """
        super(Dataset, self).__init__()
        self.config = config
        self.data_folder = data_folder
        self.images_path = glob.glob(data_folder + '/*')

        # 定义数据处理方式
        self.hr_transform = ImageTransforms(
            split=self.config.split,
            crop_size=self.config.image_size,
            scaling_factor=self.config.scaling_factor,
            img_type=self.config.hr_img_type,
            is_lr=False,
            is_lr_amplify=False
        )
        self.lr_transform = ImageTransforms(
            split=self.config.split,
            crop_size=self.config.image_size,
            scaling_factor=self.config.scaling_factor,
            img_type=self.config.lr_img_type,
            is_lr=True,
            is_lr_amplify=self.config.is_lr_amplify
        )

    def __getitem__(self, i):
        """
        为了使用 PyTorch 的 DataLoader，必须提供该方法.
        :参数 i: 图像检索号
        :返回: 返回第i个低分辨率和高分辨率的图像对
        """
        # 读取图像(不需要手动 close)
        img = None
        try:
            with Image.open(self.images_path[i], mode='r') as img_open:
                img = img_open.convert('RGB')
        except:
            print(f'错误文件路径:{self.images_path[i]}')

        # 获取高分辨率图像
        hr_imgs, box = self.hr_transform(img)
        # 获取低分辨率图像
        lr_imgs, _ = self.lr_transform(img, box=box)

        # 数据增强
        if self.config.split == "train" and self.config.is_augment:
            hflip, vflip, rot90 = get_augment_param()  # 获取数据增强的参数
            lr_imgs = augment(lr_imgs, hflip, vflip, rot90)
            hr_imgs = augment(hr_imgs, hflip, vflip, rot90)

        # 获取图片名称(filename.后缀)
        file_name_suffix = os.path.basename(self.images_path[i]).split('.')
        # 测试阶段需要用到图片名称和后缀名
        filename = file_name_suffix[0]  # 获取图片名称
        suffix = file_name_suffix[1]  # 获取图片后缀名

        # bsrgan 退化模型(lr_imgs 需要重新获取),hr_imgs无变化
        if self.config.split == "train":
            # (c,h,w) -> (h,w,c)
            hr_imgs = hr_imgs.permute(1, 2, 0)
            hr_imgs = hr_imgs.numpy()
            lr_imgs, hr_imgs = utils_blindsr.degradation_bsrgan(hr_imgs, self.config.scaling_factor, lq_patchsize=self.config.crop_size, isp_model=None)
            lr_imgs = torch.from_numpy(lr_imgs).permute(2, 0, 1)
            hr_imgs = torch.from_numpy(hr_imgs).permute(2, 0, 1)

        return lr_imgs, hr_imgs, (filename, suffix)

    def __len__(self):
        """
        为了使用PyTorch的DataLoader，必须提供该方法.
        :返回: 加载的图像总数
        """
        return len(self.images_path)
