import os
import numpy as np
import torch
from PIL import Image
from utils import LoadDataset
from imblearn.over_sampling import RandomOverSampler

def generate_sample_label_set(hr1_train,hr1_val,hr1_test,hr2_train, hr2_val, hr2_test,lab_train,lab_val,lab_test):
    print("generating_sample_label_set")
    print('-------------------------------')
    data_path = "Data/CLCD_samples"
    # (94371840, 3, 3, 6) (94371840, 1)
    training_samples, training_labels = generate_TVT_sample_label_set(hr1_train,hr2_train,lab_train)
    with open(os.path.join(data_path, 'training_samples.npy'),'bw') as outfile:
        np.save(outfile, training_samples)
    with open(os.path.join(data_path, 'training_labels.npy'),'bw') as outfile:
        np.save(outfile, training_labels)
    val_samples, val_labels = generate_TVT_sample_label_set(hr1_val,hr2_val,lab_val)
    with open(os.path.join(data_path, 'val_samples.npy'),'bw') as outfile:
        np.save(outfile, val_samples)
    with open(os.path.join(data_path, 'val_labels.npy'),'bw') as outfile:
        np.save(outfile, val_labels)
    test_samples, test_labels = generate_TVT_sample_label_set(hr1_test, hr2_test, lab_test)
    with open(os.path.join(data_path, 'test_samples.npy'),'bw') as outfile:
        np.save(outfile, test_samples)
    with open(os.path.join(data_path, 'test_labels.npy'),'bw') as outfile:
        np.save(outfile, test_labels)
    #training_samples的shape应该是[total_samples,6,3,3]
    #training_labels的shape应该是[total_samples,1]

    return training_samples, training_labels, val_samples, val_labels, test_samples, test_labels

#train_val_test
def generate_TVT_sample_label_set(hr1_train,hr2_train,lab_train):
    files = os.listdir(hr1_train)
    image_files = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]
    print(len(image_files)) #360
    set_samples_num = len(image_files)//10
    training_samples=np.zeros((set_samples_num,262144, 3, 3, 6)) 
    training_labels=np.zeros((set_samples_num,262144,1))
    print(training_samples.shape)
    k=0
    l=0
    filecnt = 0
    for file_name in files:
        filecnt = filecnt + 1
        if(filecnt == (set_samples_num) + 1):
            break
        print(file_name)

        hr1_file_path = os.path.join(hr1_train, file_name)
        img=Image.open(hr1_file_path)
        img_array = np.array(img)
        # print(img_array.shape) #512 512 3

        hr2_file_path = os.path.join(hr2_train, file_name)
        img_hr2=Image.open(hr2_file_path)
        img2_array = np.array(img_hr2)
        # print(img2_array.shape)

        label_file_path= os.path.join(lab_train, file_name)
        img_label=Image.open(label_file_path)
        label_array = np.array(img_label)
        # print(label_array.shape) #512 512 1
        reshaped_label = label_array.reshape(-1, 1)

        training_labels[l]=reshaped_label
        l=l+1

        concatenated_array = np.concatenate([img_array, img2_array], axis=-1)
        print("Shape of concatenated_array:", concatenated_array.shape) #512x512x6
        
        padded_array = np.pad(concatenated_array, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        print("Shape of padded_array:", padded_array.shape)
        print(padded_array[:,:,0]) #514 514 6
        
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
    merged_array = np.transpose(merged_array, (0,3,2,1)) #->6,3,3
    print("Shape of merged_array:", merged_array.shape) #[total_samples,6,3,3],得到最后的training_samples!

    print(training_labels.shape)
    merged_label_array = training_labels.reshape(-1, 1)
    merged_label_array[merged_label_array == 255] = 1
    new_array = np.zeros((merged_label_array.shape[0], 2))
    new_array[merged_label_array[:, 0] == 1, 0] = 1
    new_array[merged_label_array[:, 0] == 0, 1] = 1



    training_samples = merged_array 
    training_labels = new_array

    print("Shape of training_samples:", training_samples.shape) #(786432, 6, 3, 3)
    print(training_labels.shape) # (786432, 2)
    #0, 1代表未变化，1, 0代表变化！！！
    #对于image0来说 unchanged -- 782253, change -- 4179
    print('num of training samples: unchanged -- {}, change -- {}'.format(np.sum(np.all(training_labels == [0, 1], axis=1)), np.sum(np.all(training_labels == [1, 0], axis=1))))
    ros = RandomOverSampler(random_state=42)
    training_samples = training_samples.reshape([training_samples.shape[0], -1])
    training_samples, training_labels = ros.fit_resample(training_samples, training_labels)
    training_samples = training_samples.reshape([training_samples.shape[0], 6, 3, 3])
    # print(training_labels)
    # print(training_labels.shape) #(1564506, 1)
    # print(training_labels.sum()) #782253
    training_labels = np.where(training_labels == 1, [1., 0.], [0., 1.])
    # print(training_labels)
    print('num of training samples after ros: unchanged -- {}, change -- {}'.format(np.sum(np.all(training_labels == [0, 1], axis=1)), np.sum(np.all(training_labels == [1, 0], axis=1))))

    return training_samples, training_labels

def load_sample_label_set(datapath):
    print('loading sample_label_set')
    print('-------------------------------')
    training_samples = np.load(os.path.join(datapath, 'training_samples.npy'))
    training_labels = np.load(os.path.join(datapath, 'training_labels.npy'))
    val_samples = np.load(os.path.join(datapath, 'val_samples.npy'))
    val_labels = np.load(os.path.join(datapath, 'val_labels.npy'))
    test_samples = np.load(os.path.join(datapath, 'test_samples.npy'))
    test_labels = np.load(os.path.join(datapath, 'test_labels.npy'))
    return training_samples, training_labels, val_samples, val_labels, test_samples, test_labels


def get_dataloader(training_samples, training_labels, val_samples, val_labels, test_samples, test_labels):
    X_train = torch.from_numpy(training_samples)
    y_train = torch.from_numpy(training_labels)
    X_val = torch.from_numpy(val_samples)
    y_val = torch.from_numpy(val_labels)
    X_test = torch.from_numpy(test_samples)
    y_test = torch.from_numpy(test_labels)
    train_dataset = LoadDataset(X_train,y_train)
    val_dataset = LoadDataset(X_val,y_val)
    test_dataset = LoadDataset(X_test,y_test)

    return train_dataset, val_dataset, test_dataset
