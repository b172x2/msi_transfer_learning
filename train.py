import time
import numpy as np
import torch
from torch.nn import Softmax

#training process of the model
def train(dataloader, model, loss_fn, optimizer, device):
    size=len(dataloader.dataset) 
    print(size) #786432
    model.train()
    total_loss = 0
    total_acc = 0
    for batch,(X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.type(torch.cuda.FloatTensor)
        pred=model(X)
        # print(pred.shape)

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
        softmax_predictions = torch.argmax(torch.nn.functional.softmax(pred, dim=1), dim=1)
        # pred_result = torch.zeros((softmax_predictions.shape[0],2))
        # pred_result[range(softmax_predictions.size(0)), softmax_predictions] = 1
        # pred_result = pred_result.type(torch.cuda.FloatTensor)
        values_to_compare = y[:, 1]
        total_acc = total_acc + (softmax_predictions==values_to_compare).type(torch.float).sum().item()
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
        for X, y in dataloader:
            # X_con = torch.cat((X1,X2), dim=1)
            # print(X_con.shape) torch.Size([32, 6, 512, 512])
            X, y = X.to(device), y.to(device)
            # X_con=X_con.type(torch.cuda.FloatTensor)
            X = X.type(torch.cuda.FloatTensor)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            softmax_predictions = torch.argmax(torch.nn.functional.softmax(pred, dim=1), dim=1)

            values_to_compare = y[:, 1]
            correct = correct + (softmax_predictions==values_to_compare).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct