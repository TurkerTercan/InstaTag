import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


class ResnetObject(nn.Module):
    def __init__(self, num_classes):
        super(ResnetObject, self).__init__()
        self.model = models.resnet50(num_classes=1000, pretrained=True)
        layer4 = self.model.layer4
        self.model.layer4 = nn.Sequential(nn.Dropout(0.5),
                                           layer4)
        self.model.avgpool = AvgPool()
        self.model.fc = nn.Linear(2048, num_classes)

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
