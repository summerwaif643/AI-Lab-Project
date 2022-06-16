import streamlit as st 
import pandas as pd 

# Run the content of this file with: python -m streamlit run your_script.py

# File upload *Max 200mb File allowed
bw_image = st.file_uploader("Drop your image here!")

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
        st.image([bw_image])