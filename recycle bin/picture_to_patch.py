#功能实现对HRSCD的裁切，变成512和512的patch，以对齐CLCD


import os
import re
from PIL import Image
import numpy as np

def crop_and_rename_images(input_folder, output_folder, patch_size=512):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    # 遍历输入文件夹
    for root, _, files in os.walk(input_folder):
        for file in files:
            # 获取图像文件路径
            input_path = os.path.join(root, file)
            # 使用正则表达式提取i和j的值
            match = re.match(r'image(\d+).jpg', file)

            if match:
                i_value = int(match.group(1))
                print(f"i: {i_value}")
            else:
                print("文件名不匹配指定的格式")

            # 打开图像文件
            image = Image.open(input_path)

            # 获取图像尺寸
            width, height = image.size

            # 计算水平和垂直方向上的patch数量
            horizontal_patches = width // patch_size
            vertical_patches = height // patch_size

            # 遍历所有patch
            for i in range(horizontal_patches):
                for j in range(vertical_patches):
                    # 计算patch的坐标
                    left = i * patch_size
                    upper = j * patch_size
                    right = left + patch_size
                    lower = upper + patch_size

                    # 裁剪patch
                    patch = image.crop((left, upper, right, lower))
                    
                    patch = patch.convert("RGB")
                    # 构造新文件名
                    new_filename = f"image{i_value}_{i}_{j}.jpg"

                    # 构造输出路径
                    output_path = os.path.join(output_folder, new_filename)

                    # 保存裁剪后的patch
                    patch.save(output_path)

                    print(f'Cropped and saved: {output_path}')

# 输入文件夹路径
input_folder = 'Data/HRSCD/test/msi_t2'

# 输出文件夹路径
output_folder = 'Data/HRSCD/test/msi_t2_patch'

# 调用函数进行裁剪和重命名
crop_and_rename_images(input_folder, output_folder)
