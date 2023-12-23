import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights

class PretrainedResNet50(nn.Module):
    def __init__(self, bands=3, num_class=1000, freeze_bottom_layers=False):
        super(PretrainedResNet50, self).__init__()
        net = models.resnet50(weights=ResNet50_Weights.DEFAULT) #预训练参数
        net.conv1 = nn.Conv2d(bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        net.fc = nn.Linear(2048, num_class)
        if freeze_bottom_layers:
        # 冻结底层卷积层
            for param in net.conv1.parameters():
                param.requires_grad = False
            for param in net.layer1.parameters():
                param.requires_grad = False
            for param in net.layer2.parameters():
                param.requires_grad = False
            # for param in net.layer3.parameters():
            #     param.requires_grad = False
            # for param in net.layer4.parameters():
            #     param.requires_grad = False
        self.net = net
    def forward(self, x):
        return self.net(x)


