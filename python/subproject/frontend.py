from matplotlib import image
import streamlit as st 
import pandas as pd 

import torch
import torch.nn as nn 
from torchvision import datasets, transforms

from skimage.color import lab2rgb, rgb2lab, rgb2gray

from colorization import ColorizationNet 
from net import validate, to_rgb

import os

from grayscale import GrayscaleImageFolder


model = ColorizationNet()
model.load_state_dict(torch.load("/home/ddave/AI-Lab-Project/python/subproject/checkpoints/model-epoch-9-losses-0.003.pth"))
criterion = nn.MSELoss()

def colorize(dir):
    val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    val_imagefolder = GrayscaleImageFolder(dir, val_transforms)
    val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=64, shuffle=False)

    with torch.no_grad():
            #Colorize the image and save it in imagefolder
            validate(val_loader, model, criterion, True, 0)

# Import single class from main.py 
# Run the content of this file with: python -m streamlit run your_script.py

# File upload *Max 200mb File allowed
image = st.file_uploader("Drop your image here!")

# Columns 
col1, col2, col3 = st.columns(3)

if image is not None:
    ## Here call whatever you need for the frontend and display the resulting image
    ## Input relative path here;; 
    
    #Save image
    dir = "/home/ddave/AI-Lab-Project/python/subproject/images/streamlit_cache"
    filename = os.path.join(dir + '/class', image.name)
    with open(filename, "wb") as f:
        f.write((image).getbuffer())

    with col1: 
        st.write('Original image')
        st.image([image])

    with col2:
        st.write('Colorized image')
        colorize(dir)
        #Show colorized image ()
        st.image('/home/ddave/AI-Lab-Project/python/subproject/outputs/color/img-0-epoch-0.jpg')

#Once its done, delete all the contents in 
class_dir = "/home/ddave/AI-Lab-Project/python/subproject/images/streamlit_cache/class"
for filename in os.listdir(class_dir):
    path = os.path.join(class_dir, filename)
    try:
        if os.path.isfile(path):
            os.unlink(path)
    except Exception as e:
        st.write("Problem removing files")