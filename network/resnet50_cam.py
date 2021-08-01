import torch.nn as nn
import torch.nn.functional as F
from functions import torchutils
from net import resnet50
import torch


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage0 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage1 = nn.Sequential(self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, 200, 1, bias=False)
        self.backbone = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)####.detach()
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        cam0 = x4
        x = torchutils.gap2d(cam0, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, 200)
        cam0 = F.relu(cam0)

        return x, cam0

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return list(self.backbone.parameters()), list(self.newly_added.parameters())#, list(self.att.parameters())


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()
    def forward(self, x):

        h = x

        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x4 = F.conv2d(x4, self.classifier.weight)
        x4 = F.relu(x4)
        x4 = x4[0] + x4[1].flip(-1)    # the data-loader did clone the img and take a flip(-1) to train the image
                                    # maybe it can enhance the performance
        cls = super().forward(h)

        return cls[0], x4
