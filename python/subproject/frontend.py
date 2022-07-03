from matplotlib import image
import streamlit as st 
import pandas as pd 

import torch
import torch.nn as nn 
from skimage.color import lab2rgb, rgb2lab, rgb2gray

from colorization import ColorizationNet 
from main import validate
img = "/home/ddave/AI-Lab-Project/python/68747470733a2f2f7261772e6769746875622e636f6d2f6d696b6f6c616c7973656e6b6f2f6c656e612f6d61737465722f6c656e612e706e67.png"

model = ColorizationNet()
model.load_state_dict(torch.load("/home/ddave/AI-Lab-Project/python/subproject/checkpoints/model-epoch-23-losses-0.003.pth"))
criterion = nn.MSELoss()
val_imagefolder = ""
val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=64, shuffle=False)

def colorize(image):
    with torch.no_grad():
            validate(val_loader, model, criterion, [image], 0)
# Import single class from main.py 
# Run the content of this file with: python -m streamlit run your_script.py

# File upload *Max 200mb File allowed
image = st.file_uploader("Drop your image here!")
bw_image = rgb2gray(image)

# Bring image to black and white 

# Columns 
col1, col2 = st.columns(2)

if bw_image is not None:
    ## Here call whatever you need for the frontend and display the resulting image
    ## Input relative path here;; 
    with col1: 
        st.write('Original image')
        st.image([bw_image])

    with col2:
        st.write('Colorized image')
        st.image()