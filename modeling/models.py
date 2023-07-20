import torch.nn as nn
import torchvision.models as models


class DenseNet(nn.Module):
    def __init__(self, type=161, num_classes=10, pretrained=False):
        super(DenseNet, self).__init__()
        if type == 121:
            self.backbone = models.densenet121(pretrained=pretrained)
        elif type == 161:
            self.backbone = models.densenet161(pretrained=pretrained)
        elif type == 169:
            self.backbone = models.densenet169(pretrained=pretrained)
        elif type == 201:
            self.backbone = models.densenet201(pretrained=pretrained)
        else:
            raise NotImplementedError
        self.linear = nn.Linear(1000, num_classes)
        self.penultimate_layer = None

    def get_penultimate_dim(self):
        return 1000

    def forward(self, x):
        bbone = self.backbone(x)
        self.penultimate_layer = bbone
        out = self.linear(bbone)
        return out


class VGG(nn.Module):
    def __init__(self, type=16, num_classes=10, pretrained=False):
        super(VGG, self).__init__()
        if type == 11:
            self.backbone = models.vgg11_bn(pretrained=pretrained)
        elif type == 13:
            self.backbone = models.vgg13_bn(pretrained=pretrained)
        elif type == 16:
            self.backbone = models.vgg16_bn(pretrained=pretrained)
        elif type == 19:
            self.backbone = models.vgg19_bn(pretrained=pretrained)
        else:
            raise NotImplementedError
        self.linear = nn.Linear(1000, num_classes)
        self.penultimate_layer = None

    def get_penultimate_dim(self):
        return 1000

    def forward(self, x):
        bbone = self.backbone(x)
        self.penultimate_layer = bbone
        out = self.linear(bbone)
        return out


class MLPclassifier(nn.Module):
    def __init__(self, dim, num_classes=10, embSize=256):
        super(MLPclassifier, self).__init__()
        self.embSize = embSize
        self.dim = dim
        self.lm1 = nn.Sequential(
            nn.Linear(self.dim, embSize),
            nn.ReLU()
        )
        self.lm2 = nn.Linear(embSize, num_classes)
        self.penultimate_layer = None

    def forward(self, x):
        x = x.view(-1, self.dim)
        emb = self.lm1(x)
        self.penultimate_layer = emb
        out = self.lm2(emb)
        return out

    def get_penultimate_dim(self):
        return self.embSize

