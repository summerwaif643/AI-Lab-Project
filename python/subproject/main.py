import numpy as np 
import matplotlib.pyplot as plt 

from skimage.color import lab2rgb, rgb2lab, rbg2gray
from skimage import io 

import torch 
import torch.nn
import torch.nn.functional as functional 

import torchvision.models as models
from torchvision import datasets, transform

import os

# Use GPU if available
use_gpu = torch.cuda.is_available()

