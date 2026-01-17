import glob
import os.path
from PIL import Image

dest_path = 'E:/图像超分辨率应用/data/train/REDS2K'
source_path = 'F:/视频超分辨率相关代码复现/代码复现/data/REDS'

source_paths = glob.glob(source_path + '/*')

for i, file_path in enumerate(source_paths):
    file_path = os.path.join(file_path, '00000000.png')
    with Image.open(file_path, mode='r') as img_open:
        img = img_open.convert('RGB')
        img.save(os.path.join(dest_path, f'{i}.png'))
