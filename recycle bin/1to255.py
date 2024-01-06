from PIL import Image
import numpy as np
import os

def process_label_image(input_folder, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹
    for root, _, files in os.walk(input_folder):
        for file in files:
            input_path = os.path.join(root, file)

            # 打开图像文件
            label_image = Image.open(input_path)

            # 将图像转换为NumPy数组
            label_array = np.array(label_image)

            # 将像素值大于0的设置为255
            label_array[label_array == 1] = 255

            # 将NumPy数组转换回图像
            processed_label_image = Image.fromarray(label_array.astype(np.uint8))
            # 构造输出路径
            output_path = os.path.join(output_folder, file)

            # 保存处理后的图像
            processed_label_image.save(output_path)

            print(f'Processed and saved: {output_path}')

# 输入文件夹路径
input_folder = 'Data/HRSCD/train/label'

# 输出文件夹路径
output_folder = 'Data/HRSCD/train/1to255_label'

# 调用函数进行处理
process_label_image(input_folder, output_folder)
