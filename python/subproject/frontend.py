from matplotlib import image
import streamlit as st 
import pandas as pd 

import torch
import torch.nn as nn 
from skimage.color import lab2rgb, rgb2lab, rgb2gray

from colorization import ColorizationNet 
from main import validate, to_rgb

import os

'''
model = ColorizationNet()
print(1)
model.load_state_dict(torch.load("/home/ddave/AI-Lab-Project/python/subproject/checkpoints/model-epoch-23-losses-0.003.pth"))
print(2)
criterion = nn.MSELoss()
print(3)
val_imagefolder = ""
print(4)
val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=64, shuffle=False)


def colorize(image):
    with torch.no_grad():
            validate(val_loader, model, criterion, True, 0)

# Import single class from main.py 
# Run the content of this file with: python -m streamlit run your_script.py

# File upload *Max 200mb File allowed
image = st.file_uploader("Drop your image here!")

# Bring image to black and white
#bw_image = rgb2gray(image)

with open(os.path.join("python/subproject/images/streamlit_cache", image.name), "wb") as f:
    f.write(image.getbuffer()) 

# Columns 
col1, col2, col3 = st.columns(3)

if bw_image is not None:
    ## Here call whatever you need for the frontend and display the resulting image
    ## Input relative path here;; 
    with col1: 
        st.write('Original image')
        st.image([image])


    with col3:
        st.write("Ground truth")
        st.image([bw_image])

    with col2:
        st.write('Colorized image')
        colorize(image)
        st.image()

'''