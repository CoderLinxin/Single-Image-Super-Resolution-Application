from PIL import Image
from torchvision import transforms

img_path = 'data/test/Set14/comic.png'

to_tensor_transform = transforms.ToTensor()

img = None
with Image.open(img_path, mode='r') as img_open:
    img = img_open.convert('RGB')
lr = img.resize((img.width // 4, img.height // 4), Image.BICUBIC)

# 从原图中尽可能大地中心裁剪出大小能被 scaling_factor 整除的图像块
x_remainder = img.width % 4
y_remainder = img.height % 4
left = x_remainder // 2
top = y_remainder // 2
right = img.width - (x_remainder - left)
bottom = img.height - (y_remainder - top)
box = (left, top, right, bottom)
img2 = img.crop(box)
lr2 = img2.resize((img2.width // 4, img2.height // 4), Image.BICUBIC)

lr = to_tensor_transform(lr)
lr2 = to_tensor_transform(lr2)

print((lr - lr2).abs().sum())
