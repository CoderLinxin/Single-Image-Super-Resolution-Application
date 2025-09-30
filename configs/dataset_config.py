class DatasetConfig:
    def __init__(
            self,
            split: str,
            crop_size: int = 64,
            scaling_factor: int = 4,
            lr_img_type: str = '[0,1]',
            hr_img_type: str = '[0,1]',
            is_lr_amplify: bool = False,
            is_augment: bool = False
    ):
        """
        :param split: 'train' 或者 'eval|test'
        :param crop_size: 对应高分辨率图像上截取的用于训练的图像块尺寸 / scaling_factor = 低分辨率上的图像块大小
        :param scaling_factor: 放大比例
        :param lr_img_type: 低分辨率图像预处理方式 [0,255], [0,1], [-1,1]
        :param hr_img_type: 高分辨率图像预处理方式 [0,255], [0,1], [-1,1]
        :param is_lr_amplify: 是否获取与 hr 图像一样大小的 lr 图像
        :param is_augment: 是否进行数据增强(训练阶段)
        """
        self.crop_size = crop_size
        self.split = split
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.is_lr_amplify = is_lr_amplify
        self.is_augment = is_augment

        # 高分辨率图像上截取的图像块大小
        self.image_size = self.crop_size * self.scaling_factor

        assert self.split.lower() in {'train', 'eval|test'}
        assert self.lr_img_type in {'[0,255]', '[0,1]', '[-1,1]'}, \
            f'lr_img_type should be one of "[0,255]" or "[0,1]" or "[-1,1]"'
        assert self.hr_img_type in {'[0,255]', '[0,1]', '[-1,1]'}, \
            f'hr_img_type should be one of "[0,255]" or "[0,1]" or "[-1,1]"'
