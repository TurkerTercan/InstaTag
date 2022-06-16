import os

import torch
import torch.nn as nn
import torchvision.models as models
from source.models.resnet_object import AvgPool


class ResnetScene(nn.Module):
    def __init__(self,  num_classes):
        super(ResnetScene, self).__init__()
        model_file = 'resnet50_places365.pth.tar'
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)

        self.model = models.resnet50(num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)

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
