import time
import numpy as np
import torch
from torch.nn import Softmax

#training process of the model
def train(dataloader, model, loss_fn, optimizer, device):
    size=len(dataloader.dataset) #360 for CLCD
    model.train()
    total_loss = 0
    total_acc = 0
    window_size = 3
    for batch,(X,y) in enumerate(dataloader):
        # print(X1.shape)
        # print(X2.shape)
        # X_con = torch.cat((X1,X2), dim=1)
        # print(X_con.shape) torch.Size([32, 6, 512, 512])
        X, y = X.to(device), y.to(device)
        # X_con=X_con.type(torch.cuda.FloatTensor)
        X = X.type(torch.cuda.FloatTensor)
        #X_con的shape是[batch_size,6,512,512]
        #先在X_con的外圈padding,X_con变成[batch_size, 6, 514,514]
        # X_con_padded = np.pad(X_con, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
        # print(X_con.shape)
        # print(X_con[0][1])
        #再把X_con切割成[batch_size,512*512,6,3,3](对514*514的图像做类似卷积的滑动窗口操作)
        # batch_size, num_channels, height, width = X_con_padded.shape
        # X_con_sliced = np.zeros((batch_size, height - window_size + 1, width - window_size + 1, num_channels, window_size, window_size))
        #滑动窗口操作
        # for i in range(height - window_size + 1):
        #     for j in range(width - window_size + 1):
        #         X_con_sliced[:, i, j, :, :, :] = X_con_padded[:, :, i:i+window_size, j:j+window_size]
        # X_con_sliced_reshaped = X_con_sliced.reshape(batch_size, -1, num_channels, window_size, window_size)
        # print(X_con_sliced_reshaped[0][0].shape) #(32, 262144, 6, 3, 3) #X_con_sliced_reshaped[0][0].shape=[6,3,3]
        # X_con_sliced_reshaped = torch.from_numpy(X_con_sliced_reshaped)
        # merged_tensor = X_con_sliced_reshaped.view(-1, 6, 3, 3)
        # pred_tensor=np.zeros((batch_size,height,width)) #这里要初始化为[batchsize,512*512]
        # for i in range(batch_size):
        #     pred_per_batch_size=np.
        #     for j in range(512*512):
        #         pred=model(X_con_sliced_reshaped[i][j])
        #     pred_tensor[i].append(pred) #pred_tensor存放的是预测图的一维tensor
        pred=model(X)
        print(pred.shape)

        #pred_tensor的shape应该是[batchsize,512*512]
        #y的shape会review成[batch_size,512*512]
        #loss=loss_fn(pred,y)

        #model的输入是6x3x3的tensor,输出二分类结果1或0，代表变化和不变

        loss=loss_fn(pred,y) #这里会对每个batch的样本算loss，再取平均，所以得到的是batch的平均loss

        #back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #这里total_loss代表的是一个epoch的average_loss，除以len(dataloader)可以得到!
        total_loss = total_loss + loss.item() / len(dataloader)
        # print(f"loss.item:{loss.item()}")
        # print(f"len:{len(dataloader)}") #180,因为360/2(总图片量除切的batchsize)

        # print(pred.shape)
        # print(y.shape)
        # print(torch.nn.functional.softmax(pred, dim=0)==y)
        total_acc = total_acc + (torch.nn.functional.softmax(pred, dim=0)==y).type(torch.float).sum().item()
        if batch % 100 == 0:
            loss, current =loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    total_acc /= size
    print(f"Train Error: \n Accuracy: {(100*total_acc):>0.1f}%, Avg loss: {total_loss:>8f} \n")
    return total_loss, total_acc


def test(dataloader,model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X1, y in dataloader:
            # X_con = torch.cat((X1,X2), dim=1)
            # print(X_con.shape) torch.Size([32, 6, 512, 512])
            X, y = X.to(device), y.to(device)
            # X_con=X_con.type(torch.cuda.FloatTensor)
            X = X.type(torch.cuda.FloatTensor)
            pred = model(X)

            test_loss += loss_fn(Softmax()(pred), y).item()
            correct += (Softmax()(pred).argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct