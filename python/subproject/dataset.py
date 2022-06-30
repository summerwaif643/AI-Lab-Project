from fileinput import filename
import os 
import numpy as np
import shutil 
from PIL import Image
from matplotlib import pyplot as plt

'''
We have two subdirectories
Train/ which contains a lot of images
70/30% of scheme for train test

'''
class Dataset():

    #TODO: Set up as working images dataset path
    #TODO: Maybe create a  
    __dataset_path = "/home/ddave/AI-Lab-Project/python/subproject/images/testSet_resize"

    def file_count(self, dir):
        """
        Returns total file count given directory.

            Parameters:
                dir (string): The directory

            Returns:
                n_files (int): The number of files in the directory
        """

        list = os.listdir(dir)
        n_files = len(list)
        return n_files

    def __init__(self, dataset_dir):
        self.__dataset_path = dataset_dir

        #TODO: Do some check on this image count (is 0, works correctly, pass as)
        #TODO: Check if it is 0 at the end of the class
        #TODO: Pass as *kwargs
        image_count = 0

        #TODO: Bring this to relative path
        # Count files to ensure 70/30 splitting with whatever dataset
        for i in os.listdir(self.__dataset_path):
            filename = os.path.join(self.__dataset_path, i)
            if os.path.isfile(filename):
                image_count += 1
                

        #Ensure 70%/30% splitting scheme
        #TODO: Is there a better way to do this? Missing one image
        train_files = int(image_count * 0.7)
        validation_files = int(image_count * 0.3)

        if not (os.path.exists("images/train_files/")):
            os.makedirs("images/train_files/")

        if not (os.path.exists("images/validation_files/")):
            os.makedirs("images/validation_files/")

        for i, file in enumerate(os.listdir(self.__dataset_path)):
            
            fullpath = os.path.join(self.__dataset_path, file)
            # First 30% files from dataset folder into validation folder
            if i < train_files:
                os.rename(fullpath, "/home/ddave/AI-Lab-Project/python/subproject/images/validation_files/" + file) 

            else:
                # 70% splitting scheme 
                os.rename(fullpath, "/home/ddave/AI-Lab-Project/python/subproject/images/train_files/" + file) 


        #TODO: Raise error if validation/train files are empty
        if self.file_count('images/validation_files/') == 0:
            raise Exception("No files were moved into validation")

        if self.file_count('images/train_files/') == 0:
            raise Exception("No files were moved into training")
