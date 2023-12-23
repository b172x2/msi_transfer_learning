from torch.nn import Softmax
import torch.nn as nn
import torch
# pred=torch.tensor([[0.1,0.2,0.3],[0.2,0.6,0.8]])
# y=torch.tensor([[0.,1.,1.],[1.,1.,0.]])
pred=torch.tensor([[0.1,0.2],[0.2,0.6]])
y=torch.tensor([[0.,1.],[1.,0.]])

# pred=torch.tensor([0.,1.])
# y=torch.tensor([0.,1.])
print(pred)
print(y)
loss_fn = nn.CrossEntropyLoss()
loss=loss_fn(pred,y) #这里会做softmax处理的！
#这里loss的计算过程是对两个样本分别计算loss，再除以2取平均值

print(loss.item())

for i in range(1,10):
    print(i)