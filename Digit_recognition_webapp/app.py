
import streamlit as st
import json
import requests
import matplotlib.pyplot as plt
import numpy as np

URI = 'https://127.0.0.1:5000/'

st.title('Neural Network Visualizer')
st.sidebar.markdown('## Input Image')  # making input image as sidebar
if st.button('Get random predicition'):
    response = requests.post(URI, data={}) # no need to send data as we are not using in server
    response = json.loads(response.text)
    preds = response.get('prediciton')
    image = response.get('image')
    image = np.reshape(image,(28,28)) # doing reshape to get the original image size
    
    st.sidebar.image(image, width=150)
    
    for layer,p in enumerate(preds):
        numbers = np.squeeze(np.array(p)) # squeeze method removes additional dimensionality to the data
        
        plt.figure(figsize=(32,4))
        if layer == 2:
            row = 1
            col = 10
        else:
            row =2
            col = 16
        
        for i,number in enumerate(numbers):
            plt.subplot(row,col,i+1)
            plt.imshow(number * np.ones((8,8,3)).astype('float32'))
            plt.xticks([])
            plt.yticks([])
            if layer == 2:
                plt.xlabel(str(i), fontsize=40)
        plt.subplots_adjust(wspace=0.05, hspace=0.05) # by default these spaces are 0.2
        plt.tight_layout()
        st.text('Layer {}'.format(layer+1))
        st.pyplot()
