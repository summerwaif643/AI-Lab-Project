import torch
import torch.nn as nn 
import torch.nn.functional as functional
import torchvision.models as models
from torchvision import datasets, transforms

class ColorizationNet(nn.module):
    def __init__(self, input_size=128):
        resnet = models.resnet18(num_classes=365)

        ## What is the unsqueeze?
        # Change first convolutional layer to accept a single channel
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1))

        self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

        self.upsample = nn.Sequential(
            
        )