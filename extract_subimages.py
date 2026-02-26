import glob
import os.path
import timeit
import time

from PIL import Image

path = 'data/train'
train_data_set_path_list = [
    # 'blend',
    # 'RealSR(V3)',
    # 'DIV2K_train_HR',
    'Flickr2K_HR',
    'face',
    # '城市风景', '城市泊油路', '迪拜旅游城市', '日本庭院', '铁轨铁路', '乌克兰城市建筑', '自然风景'
]

save_path = 'data/train/subimages'

sub_image_size = 480
step = 480

# 遍历数据集
for dataset_path in train_data_set_path_list:
    save_data_set_path = os.path.join(save_path, dataset_path)
    data_set_path = os.path.join(path, dataset_path)
    images_path = glob.glob(data_set_path + '/*')

    if not os.path.exists(save_data_set_path):
        os.mkdir(save_data_set_path)

    print(f'正在处理:{data_set_path}')

    # 遍历单个数据集
    for image_path in images_path:
        img = None
        with Image.open(image_path, mode='r') as img_open:
            img = img_open.convert('RGB')

        img_name = os.path.basename(image_path).split('.')
        sub_image_count = 0

        # 截取子图
        left_count = img.width // sub_image_size
        top_count = img.height // sub_image_size

        for left in range(left_count):
            left = left * step
            right = left + sub_image_size
            for top in range(top_count):
                top = top * step
                bottom = top + sub_image_size

                box = (left, top, right, bottom)
                img_save_path = os.path.join(save_data_set_path, f'{img_name[0]}_{sub_image_count}.{img_name[1]}')
                img.crop(box).save(img_save_path)

                sub_image_count += 1

    print(f'{data_set_path}处理完毕')
