import os
from PIL import Image
import numpy as np

def list_files_in_directory(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list

def read_image(image_path):
    # 打开图像文件
    image = Image.open(image_path)
    return image

def image_to_matrix(image_path):
    # 打开图像文件
    image = Image.open(image_path)
    # 将图像转换为 NumPy 数组
    matrix = np.array(image)
    return matrix

# 指定路径
directory_path = "Data/HRSCD/val/label_path_copy"

# 调用函数获取文件列表
files = list_files_in_directory(directory_path)

# print(files)

# 调用函数读取图像
image = read_image(files[0])

# 显示图像信息
print(f"图像大小: {image.size}")
print(f"图像模式: {image.mode}")

for i in range(len(files)):
    matrix = image_to_matrix(files[i])
    if matrix.sum()==0:
        filename = os.path.basename(files[i])
        # print(filename)

        filepath=os.path.join("Data/HRSCD/val/label_patch",filename)
        os.remove(filepath)
        print(f"deleting:{filepath}")
