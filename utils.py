import time
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from os.path import join
import torch
import numpy as np
from torchvision.transforms import ToTensor
# 可视化GROUND TRUTH和PRE_MASK图像
# def pseudo_label_visualization(gt,pre_mask):
#     plt.figure()
#     ax = plt.subplot(1, 2, 1)
#     ax.set_title("GT")
#     plt.imshow(gt, cmap='gray')
#     ax = plt.subplot(1, 2, 2)
#     ax.set_title("CVA")
#     plt.imshow(pre_mask, cmap='gray')
#     plt.show()
def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.tif','.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def get_transform(convert=True, normalize=False):
    transform_list = []
    if convert:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class LoadDatasetFromFolder(Dataset):
    def __init__(self, hr1_path, hr2_path, lab_path):
        super(LoadDatasetFromFolder, self).__init__()

        # 获取PNG图片列表
        datalist = [name for name in os.listdir(hr1_path) if name.endswith('.png')]
        self.hr1_filenames = [join(hr1_path, x) for x in datalist]
        self.hr2_filenames = [join(hr2_path, x) for x in datalist]
        self.lab_filenames = [join(lab_path, x) for x in datalist]

        self.transform = ToTensor()
        self.label_transform = ToTensor()

    def __getitem__(self, index):
        hr1_img = self.transform(Image.open(self.hr1_filenames[index]).convert('RGB'))
        # print(hr1_img.shape) #3,512,512
        hr2_img = self.transform(Image.open(self.hr2_filenames[index]).convert('RGB'))
        label = self.label_transform(Image.open(self.lab_filenames[index]))
        # print(label.shape) #1,512,512

        return hr1_img, hr2_img, label

    def __len__(self):
        return len(self.hr1_filenames)




def data_loader_test(train_dataloader):
    # 获取第一个批次的数据
    data_iter = iter(train_dataloader)
    hr1_img, hr2_img, label = next(data_iter)

    # 打印一些信息
    print(f"HR1 Image Shape: {hr1_img.shape}, HR2 Image Shape: {hr2_img.shape}, Label Shape: {label.shape}")

    # 选择一个样本进行可视化
    sample_index = 0
    hr1_img_sample = hr1_img[sample_index]
    hr2_img_sample = hr2_img[sample_index]
    label_sample = label[sample_index]

    # 可视化图像和标签
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(hr1_img_sample.permute(1, 2, 0))
    plt.title('HR1 Image')

    plt.subplot(1, 3, 2)
    plt.imshow(hr2_img_sample.permute(1, 2, 0))
    plt.title('HR2 Image')

    plt.subplot(1, 3, 3)
    plt.imshow(label_sample.squeeze(), cmap='gray')
    plt.title('Label')

    plt.show()

def draw_curve(data, label, xlabel, ylabel, line, file_path=None, file_name=None, new_figure=True):
    epochs = range(len(data))
    if new_figure == True:
        plt.figure()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(epochs, data, line, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    if file_name != None:
        plt.savefig(os.path.join(file_path, file_name))


def generate_sample_label_set(hr1_train,hr1_val,hr1_test,hr2_train, hr2_val, hr2_test,lab_train,lab_val,lab_test):
    print("generating_sample_label_set")
    print('-------------------------------')

    files = os.listdir(hr1_train)
    image_files = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]
    print(len(image_files)) #360
    training_samples=np.zeros((len(image_files),262144, 3, 3, 6))
    training_labels=np.zeros((len(image_files),262144,1))
    print(training_samples.shape)
    k=0
    l=0
    for file_name in files:
        print(file_name)

        hr1_file_path = os.path.join(hr1_train, file_name)
        img=Image.open(hr1_file_path)
        img_array = np.array(img)
        print(img_array.shape)

        hr2_file_path = os.path.join(hr2_train, file_name)
        img_hr2=Image.open(hr2_file_path)
        img2_array = np.array(img_hr2)
        print(img2_array.shape)

        label_file_path= os.path.join(lab_train, file_name)
        img_label=Image.open(label_file_path)
        label_array = np.array(img_label)
        print(label_array.shape) #512x512x1
        reshaped_label = label_array.reshape(-1, 1)

        training_labels[l]=reshaped_label
        l=l+1

        concatenated_array = np.concatenate([img_array, img2_array], axis=-1)
        print("Shape of concatenated_array:", concatenated_array.shape) #512x512x6
        
        padded_array = np.pad(concatenated_array, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        print("Shape of padded_array:", padded_array.shape)
        print(padded_array[:,:,0]) #514x514x6
        
        window_size = (3, 3, 6)
        original_shape = padded_array.shape
        output_shape = ((original_shape[0] - window_size[0] + 1) *(original_shape[1] - window_size[1] + 1), *window_size)
        output_array = np.zeros(output_shape)
        index=0
        for i in range(concatenated_array.shape[0]):
            for j in range(concatenated_array.shape[1]):
                sub_array = padded_array[i:i + window_size[0], j:j + window_size[1], :]
                output_array[index, :, :, :] = sub_array
                index = index + 1
        print("Shape of output_array:", output_array.shape) #(262144, 3, 3, 6)
        training_samples[k]=output_array
        k = k+1
    print(training_samples.shape)
    merged_array = training_samples.reshape(-1, *training_samples.shape[2:])
    print("Shape of merged_array:", merged_array.shape) #[total_samples,3,3,6],得到最后的training_samples!

    print(training_labels.shape)
    merged_label_array = training_labels.reshape(-1, 1)
    print("Shape of merged_array:", merged_label_array.shape) #(94371840, 1)





    #training_samples的shape应该是[total_samples,6,3,3]
    #training_labels的shape应该是[total_samples,512*512]
    training_samples=[]
    training_labels=[]
    val_samples=[]
    val_labels=[]
    test_samples=[]
    test_labels=[]
    return training_samples, training_labels, val_samples, val_labels, test_samples, test_labels