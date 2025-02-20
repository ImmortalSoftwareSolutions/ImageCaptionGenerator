# Streamlit UI for caption generator

import streamlit as st
from PIL import Image
import os
# from CaptionGenerator import generate_caption
from captiongenerator32k import generate_caption
from hashtag_generator import predict_hashtags
#from instagrapi import Client


#USERNAME = 'testid3004'
#PASSWORD = 'utkarsh30'

#cl = Client()
#cl.login(USERNAME, PASSWORD)


#def post():
#    cl.photo_upload(st.session_state['save_image'],st.session_state['caption'])
#    st.success("Uploaded on IG")
    

def add_hashtags(text):
    if not text:
        return False
    
    words = text.split()
    capitalized_words = [word.capitalize() for word in words]
    hashtag_string = "#" + "".join(capitalized_words)
    
    if len(hashtag_string) > 140:
        return False
    
    return hashtag_string


def run():
    st.title("Image Caption Generator")
    uploaded_file = st.file_uploader("Upload Your Image", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        save_image = './ImageUpload/'+uploaded_file.name
        with open(save_image,'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.image(image=save_image, width=300)
        st.divider()
        st.markdown("Caption Generated:")
        try:
            with st.spinner(text="Generating"):
                caption = generate_caption(save_image)
                st.markdown(caption)
                st.markdown("Hashtag")
                hashtags = predict_hashtags(save_image)
                st.markdown(hashtags)
        except Exception as e:
            st.markdown(e)
        
        
run()
