import numpy as np 
import matplotlib.pyplot as plt 

from skimage.color import lab2rgb, rgb2lab, rbg2gray
from skimage import io 

import torch 
import torch.nn as nn 
import torch.nn.functional as functional 

import torchvision.models as models
from torchvision import datasets, transform

import os

#Classes imports
from python.subproject.colorization import ColorizationNet

# Use GPU if available
use_gpu = torch.cuda.is_available()

model = ColorizationNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

