import argparse

#training options
parser = argparse.ArgumentParser(description='Training Change Detection Network')

# training parameters
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--batchsize', default=2, type=int, help='batchsize')
parser.add_argument('--val_batchsize', default=16, type=int, help='batchsize for validation')
parser.add_argument('--num_workers', default=24, type=int, help='num of workers')
parser.add_argument('--n_class', default=2, type=int, help='number of class')
parser.add_argument('--gpu_id', default="0", type=str, help='which gpu to run.')
parser.add_argument('--suffix', default=['.png','.jpg','.tif'], type=list, help='the suffix of the image files.')
parser.add_argument('--img_size', default=512, type=int, help='imagesize')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')

# 结果保存路径
parser.add_argument('--result_path', type=str, default="Result/CLCD", help='result path')
parser.add_argument('--save_path', type=str, default="Model/CLCD", help='model save path')

# path for loading data from folder
parser.add_argument('--hr1_train', default='Data/CLCD/train/time1', type=str, help='image at t1 in training set')
parser.add_argument('--hr2_train', default='Data/CLCD/train/time2', type=str, help='image at t2 in training set')
parser.add_argument('--lab_train', default='Data/CLCD/train/label', type=str, help='label image in training set')

parser.add_argument('--hr1_val', default='Data/CLCD/val/time1', type=str, help='image at t1 in validation set')
parser.add_argument('--hr2_val', default='Data/CLCD/val/time2', type=str, help='image at t2 in validation set')
parser.add_argument('--lab_val', default='Data/CLCD/val/label', type=str, help='label image in validation set')

parser.add_argument('--hr1_test', default='Data/CLCD/test/time1', type=str, help='image at t1 in test set')
parser.add_argument('--hr2_test', default='Data/CLCD/test/time2', type=str, help='image at t2 in test set')
parser.add_argument('--lab_test', default='Data/CLCD/test/label', type=str, help='label image in test set')

# network saving
parser.add_argument('--model_dir', default='epochs/CLCD/MSCANet/', type=str, help='model save path')