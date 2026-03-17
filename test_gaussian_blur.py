import PIL.Image
from os import path
from PIL import Image, ImageFilter
from torchvision.transforms import transforms

hr_path = 'data/test/Set5/baby_GT.bmp'

with PIL.Image.open(hr_path, mode='r') as img_open:
    hr = img_open.convert('RGB')

# lr = hr.filter(ImageFilter.GaussianBlur(radius=5))
# lr = hr.filter(ImageFilter.BoxBlur(radius=5))
# lr.show()


to_tensor_transform = transforms.ToTensor()
to_pil_transform = transforms.ToPILImage()

hazy = (to_tensor_transform(hr) * 0.7 + 80 / 255.).clip(0, 1)
hazy = to_pil_transform(hazy)
hazy.show()
