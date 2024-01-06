#预训练CNN提取深度特征+CVA+CAD+Kmeans分割

# coding=utf-8
import time
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
#utils.py
from utils import *
from dataset import *
import numpy as np
import random
from train_options import parser
#train.py
from train import train, test
#model.py
from model import PretrainedResNet50

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# torch.cuda.set_per_process_memory_fraction(0.7, device=0)
torch.cuda.empty_cache()

# set seeds
def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed_torch(2022)


data_path = "Data/CLCD_samples"

#删除samples的代码
delete_all_samples(data_path)

is_exists = os.path.exists(os.path.join(data_path, 'training_samples.npy'))

if is_exists:
    training_samples,training_labels, val_samples, val_labels, test_samples, test_labels = load_sample_label_set(data_path)
else:
    training_samples, training_labels, val_samples, val_labels, test_samples, test_labels = generate_sample_label_set(args.hr1_train,args.hr1_val,args.hr1_test,args.hr2_train, args.hr2_val, args.hr2_test,args.lab_train,args.lab_val,args.lab_test)
    
# 定义数据集和数据加载器
# train_dataset = LoadDatasetFromFolder(hr1_path=args.hr1_train, hr2_path=args.hr2_train,lab_path=args.lab_train)
# val_dataset = LoadDatasetFromFolder(hr1_path=args.hr1_val, hr2_path=args.hr2_val,lab_path=args.lab_val)
# test_dataset = LoadDatasetFromFolder(hr1_path=args.hr1_test, hr2_path=args.hr2_test,lab_path=args.lab_test)



train_dataset, val_dataset, test_dataset = get_dataloader(training_samples, training_labels, val_samples, val_labels, test_samples, test_labels)
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)
# data_loader_test(train_dataloader)

# 选择和加载预训练的ResNet模型
# 使用模型
bands = 6  # 输入通道数
num_classes = 2  # 输出类别数，可根据你的任务调整
model = PretrainedResNet50(bands=bands, num_class=num_classes, freeze_bottom_layers=True).to(device)

# 输出每个参数及其 requires_grad 属性
# for name, param in model.named_parameters():
#     print(f"Parameter name: {name}, Requires grad: {param.requires_grad}")

# define the optimizer and the loss function
loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
stepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70, 90], gamma=0.1)

train_loss_list=[]
train_acc_list = []
val_loss_list = []
val_acc_list = []


for epoch in range(10):
    if epoch == 0:
        start_time = time.time()
    if epoch == 1:
        end_time = time.time()
        print(f"time for one epoch is:{end_time-start_time}")
    """ Training  """
    print(f"Epoch {epoch + 1}\n-------------------------------")
    print("learning rate is set as: {}".format(optimizer.param_groups[0]['lr']))
    train_loss, train_acc=train(train_dataloader, model, loss_fn, optimizer, device)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss, test_acc = test(val_dataloader, model, loss_fn, device)
    val_loss_list.append(test_loss)
    val_acc_list.append(test_acc)
    stepLR.step()
    draw_curve(train_loss_list, "Training Loss", "Epochs", "Loss", "b")
    draw_curve(val_loss_list,'Validation Loss', 'Epochs', 'Loss', 'r', args.result_path, 'loss.png',False)
    draw_curve(train_acc_list, 'Training accuracy', 'Epochs', 'Accuracy', 'b')
    draw_curve(val_acc_list, 'Validation accuracy', 'Epochs', 'Accuracy', 'r', args.result_path, 'accuracy.png',False)
print('Done!')
print('-------------------------------')

torch.save(model.state_dict(), args.save_path)    
print('Testing!')
print('-------------------------------')
test_loss, test_acc = test(test_dataloader, model, loss_fn, device)











