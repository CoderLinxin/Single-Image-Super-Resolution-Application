import glob
import os.path
from PIL import Image

path = 'data/train'
train_data_set_path_list = [
    'OST_dataset/animal', 'OST_dataset/building',
    'OST_dataset/grass', 'OST_dataset/mountain',
    'OST_dataset/plant', 'OST_dataset/sky', 'OST_dataset/water'
]

delete_data_count = 0

# 遍历数据集
for data_set_path in train_data_set_path_list:
    data_set_path = os.path.join(path, data_set_path)
    images_path = glob.glob(data_set_path + '/*')

    # 遍历单个数据集
    for image_path in images_path:
        img = None
        with Image.open(image_path, mode='r') as img_open:
            img = img_open.convert('RGB')
        if img.height < 256 or img.width < 256:
            os.remove(image_path)
            print(f'{image_path} 宽高为: width={img.width},height={img.height} < 256x256,删除该图片')
            delete_data_count += 1

print(f"总共删除图片数: {delete_data_count}")
