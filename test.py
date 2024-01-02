#调用一张图，输出二变化检测结果
import numpy as np
from model import PretrainedResNet50
from utils import generate_one_sample_of_one_picture
import torch
import time
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bands = 6  # 输入通道数
num_classes = 2  # 输出类别数，可根据你的任务调整
total_acc = 0
model = PretrainedResNet50(bands=bands, num_class=num_classes, freeze_bottom_layers=True).to(device)
torch.cuda.empty_cache()

save_path="Model/CLCD/model.pth"
model.load_state_dict(torch.load(save_path))
start = time.time()

hsi_t1_path = "Data/CLCD/train/time1/00000.png"
hsi_t2_path = "Data/CLCD/train/time2/00000.png"
hsi_gt = "Data/CLCD/train/label/00000.png"


sample = generate_one_sample_of_one_picture(hsi_t1_path, hsi_t2_path)

small_tensors = sample.reshape(-1, 128, 6, 3, 3)

small_tensors = torch.from_numpy(small_tensors)
small_tensors = small_tensors.to(device)
small_tensors = small_tensors.type(torch.cuda.FloatTensor)

result = np.zeros((small_tensors.shape[0],small_tensors.shape[1],2))

img=Image.open(hsi_gt) #hsi
img_array = np.array(img)

img_array[img_array == 255] = 1

print(img_array.shape)
time.sleep(100)

new_array = np.zeros((img_array.shape[0], 2))
new_array[img_array[:, 0] == 1, 0] = 1
new_array[img_array[:, 0] == 0, 1] = 1
print("Shape of merged_array:", new_array.shape) #512,2

for i in range(small_tensors.shape[0]):
    # for j in range(small_tensors.shape[1]):
    # print(small_tensors[i].shape)
    #128*6*3*3
    output = model(small_tensors[i])
    print(f"the output is{output}")
    output = output.cpu().detach().numpy()
    result[i] = output
    print(f"processing!number{i+1}")

result = result.reshape(-1,2)
print(result)
result = torch.from_numpy(result)
softmax_predictions = torch.argmax(torch.nn.functional.softmax(result, dim=1), dim=1)
print(softmax_predictions)

values_to_compare = new_array[:, 1]

print(values_to_compare)
values_to_compare = torch.tensor(values_to_compare)

print(softmax_predictions.shape)
print(values_to_compare.shape)

total_acc = total_acc + (softmax_predictions==values_to_compare).type(torch.float).sum().item()

print(total_acc)

end = time.time()
print('Inference time is: {}'.format(end-start))