import torch.nn as nn
import torch.nn.functional as f
from functions import torchutils
from network import densenet


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.densenet169 = densenet.densenet169(pretrained=True)
        self.block0 = nn.Sequential(self.densenet169.features.block0)
        for p in self.block0.parameters():
            p.requires_grad = False
        self.block1 = nn.Sequential(self.densenet169.features.denseblock1)
        self.block2 = nn.Sequential(self.densenet169.features.transition1, self.densenet169.features.denseblock2)
        self.block3 = nn.Sequential(self.densenet169.features.transition2, self.densenet169.features.denseblock3)
        self.block4 = nn.Sequential(self.densenet169.features.transition3, self.densenet169.features.denseblock4)
        self.classifier = nn.Conv2d(1664, 200, (1, 1), bias=False)
        self.backbone = nn.ModuleList([self.block0, self.block1, self.block2, self.block3, self.block4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        x0 = self.block0(x)
        x1 = self.block1(x0).detach()
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x = torchutils.gap2d(x4, keepdims=True)
        x = self.classifier(x)

        x = x.view(-1, 200)
        cam = f.relu(f.conv2d(x4, self.classifier.weight))
        cam = cam[0] + cam[1].flip(-1)
        return x, cam

    def trainable_parameters(self):
        return list(self.backbone.parameters()), list(self.newly_added.parameters())
