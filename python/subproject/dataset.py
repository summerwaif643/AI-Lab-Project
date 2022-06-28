import os 
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

'''
We have two subdirectories
Train/ which contains a lot of images
70/30% of scheme for train test

'''

images = np.load("images/ab/ab/ab1.npy")
im = Image.fromarray(images[0])
#plt.imshow(images[0], cmap='gray')
plt.show()
im.show()
#print(images)