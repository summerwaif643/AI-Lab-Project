import os 
import glob 
import time 
import numpy as np 
import matplotlib.pyplot as plt 

import torch
from torch import device
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid 
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from pathlib import Path #rewrite this
from tqdm.notebook import tqdm #???
from skimage.color import rgb2lab, lab2rgb

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") #rewrite this as in ailab 

class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE), Image.BICUBIC),
                transforms.RandomHorizontalFlip() #create additional data by adding randomness
                ]) 
                
        else: #we're on any other split, so we dont have to create additional data
            self.transforms = transforms.Resize((SIZE, SIZE), 
                                                Image.BICUBIC)
                                                
