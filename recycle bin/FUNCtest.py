from PIL import Image
import numpy as np
# 图像路径

hsi_gt = "Data/CLCD/train/label/00041.png"
# 使用Pillow打开图像

images_gt = Image.open(hsi_gt)

# 将图像转换为NumPy数组
image_array = np.array(images_gt)
image_array[image_array == 255] = 1
# 输出图像矩阵的形状
print("Image Sum:", image_array.sum())

