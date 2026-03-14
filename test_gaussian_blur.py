import PIL.Image
from os import path
from PIL import Image, ImageFilter

hr_path = 'data/test/Set5/baby_GT.bmp'

with PIL.Image.open(hr_path, mode='r') as img_open:
    hr = img_open.convert('RGB')

lr = hr.filter(ImageFilter.GaussianBlur(radius=5))
# lr = hr.filter(ImageFilter.BoxBlur(radius=5))
lr.show()
