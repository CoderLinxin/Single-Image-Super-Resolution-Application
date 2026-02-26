import glob
import os.path
import timeit
import time

from PIL import Image

path = 'data/train'
train_data_set_path_list = [
    'subimages/RealSR(V3)',
    'OST_dataset/animal', 'OST_dataset/building',
    'RealSR(V3)',
    'OST_dataset/grass', 'OST_dataset/mountain',
    'OST_dataset/plant', 'OST_dataset/sky', 'OST_dataset/water',
    'DIV2K_train_HR'
]

# 遍历数据集
for data_set_path in train_data_set_path_list:
    data_set_path = os.path.join(path, data_set_path)
    images_path = glob.glob(data_set_path + '/*')

    start_time = time.time()

    # 遍历单个数据集
    for image_path in images_path:
        img = None
        with Image.open(image_path, mode='r') as img_open:
            img = img_open.convert('RGB')

    total_time = time.time() - start_time
    print(f'数据集:{data_set_path},图片数量;{len(images_path)},读取时间:{total_time}')

