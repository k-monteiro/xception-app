from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model
from tensorflow import keras
from PIL import Image
import numpy as np
import pandas as pd
import os
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from gtts import gTTS
from io import BytesIO
import base64
from keras_preprocessing.image import ImageDataGenerator


def main():
    st.set_page_config(page_title="Sign Language Translator", layout="wide")
    tab1, tab2, tab3 = st.tabs(["Home", "Model", "Team"])
    with tab1:
        st.markdown("<h1 style='text-align: center'>Sign Language Recognition</h1>", unsafe_allow_html=True)
        
        st.markdown("<h2 style='text-align: center'>Deep Learning Model used to translate images of sign language into text and speech</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col2:
            file_ = open("home.gif", "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()

            st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="home gif">',unsafe_allow_html=True)
        
    with tab2:
        st.markdown("<h1 style='text-align: center'>Model</h1>", unsafe_allow_html=True)
        
        st.markdown("<h2 style='text-align: center'>Upload your images and see our model translate them into text and speech</h2>", unsafe_allow_html=True)

        
        def get_best_model():
            best_model = keras.models.load_model('model_Xception.h5')
            return best_model

        def preprocess_image(image, image_file, best_model):
            # image: numpy array

            # To display the uploaded image
            st.image(image_file, caption='Uploaded Image', width=300)

            #preprocessing
            image_width, image_height = 224,224
            # This path will point to the folder of the image.  The image must be stored in a sub folder within a folder.
            pred_data_dir= 'pred_images/images'
            test_datagen = ImageDataGenerator(rescale = 1./255.)
            pred_generator =  test_datagen.flow_from_directory(
                                pred_data_dir,
                                batch_size=32, 
                                class_mode='categorical',
                                color_mode='rgb',
                                target_size=(image_width, image_height))

            preds=best_model.predict(pred_generator)

            # create a list containing the class labels
            class_labels = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
            # find the index of the class with maximum score
            pred = np.argmax(preds)
            # print the label of the class with maximum score
            letter = class_labels[pred]

            return letter

        best_model = get_best_model()

        #t.markdown('You can find the Convolutional Neural Netowrk used [here](https://github.com/NJF0029/sign_language_translator)')
        
        col1, col2 = st.columns(2)

        with col1:
            
            st.subheader('Convert image to letter :fist: :arrow_right: 	:a: :arrow_right: :speaking_head_in_silhouette: :')
            image_file = st.file_uploader('Choose the Sign Language Image', ['jpeg','jpg', 'png'])

            if image_file is not None:
                with open(os.path.join("pred_images/images",image_file.name),"wb") as f:
                    f.write(image_file.getbuffer())
                image = Image.open(image_file)
                letter = preprocess_image(image, image_file, best_model)
                st.write(f'The image is predicted as {letter}')
                
                english_accent = st.selectbox(
                "Select your english accent",
                (
                    "Default",
                    "India",
                    "United Kingdom",
                    "United States",
                    "Canada",
                    "Australia",
                    "Ireland",
                    "South Africa",
                ),
                )

                if english_accent == "Default":
                    tld = "com"
                elif english_accent == "India":
                    tld = "co.in"

                elif english_accent == "United Kingdom":
                    tld = "co.uk"
                elif english_accent == "United States":
                    tld = "com"
                elif english_accent == "Canada":
                    tld = "ca"
                elif english_accent == "Australia":
                    tld = "com.au"
                elif english_accent == "Ireland":
                    tld = "ie"
                elif english_accent == "South Africa":
                    tld = "co.za"

                sound_file = BytesIO()
                tts = gTTS(text=letter, lang="en", tld=tld, slow=False)
                tts.write_to_fp(sound_file)
                st.audio(sound_file)

        with col2:
            
            st.subheader('Convert images to word/sentence :i_love_you_hand_sign: :arrow_right: :abc: :arrow_right: :speaking_head_in_silhouette: :')
            sentence_image_files = st.file_uploader('Choose the Sign Language Images', ['jpeg','jpg', 'png',], accept_multiple_files = True)

            if len(sentence_image_files) > 0:
                sentence = ''
                for image_file in sentence_image_files:
                    with open(os.path.join("pred_images/images",sentence_image_files.name),"wb") as f:
                        f.write(sentence_image_files.getbuffer())
                    image = Image.open(image_file)
                    letter = preprocess_image(image, image_file, best_model)
                    sentence += letter
                    st.write(f'The word/sentence is predicted as {sentence}')
                    
                english_accent = st.selectbox(
                "Select your english accent",
                (
                    "Default",
                    "India",
                    "United Kingdom",
                    "United States",
                    "Canada",
                    "Australia",
                    "Ireland",
                    "South Africa",
                ),
                )

                if english_accent == "Default":
                    tld = "com"
                elif english_accent == "India":
                    tld = "co.in"

                elif english_accent == "United Kingdom":
                    tld = "co.uk"
                elif english_accent == "United States":
                    tld = "com"
                elif english_accent == "Canada":
                    tld = "ca"
                elif english_accent == "Australia":
                    tld = "com.au"
                elif english_accent == "Ireland":
                    tld = "ie"
                elif english_accent == "South Africa":
                    tld = "co.za"
                    
                sound_file = BytesIO()
                tts = gTTS(text=sentence, lang="en", tld=tld, slow=False)
                tts.write_to_fp(sound_file)
                st.audio(sound_file)
    
    with tab3:
    
        st.title('Meet the Team')
        
        col1, col2 = st.columns(2)

        with col1:
            st.header("Nathan Fournillier :flag-gb:")
            image_nathan = Image.open('nathan.jpeg')
            st.image(image_nathan, width=250)
        
            st.header("Eleni Toelle :flag-de:")
            image_eleni = Image.open('eleni.jpeg')
            st.image(image_eleni, width=250)
        
        with col2:
            st.header("Rida Ruheen :flag-in:")
            image_rida = Image.open('rida.jpeg')
            st.image(image_rida, width=250)
            
            
            st.header("Kayan Monteiro :flag-br:")
            image_kayan = Image.open('kayan.jpeg')
            st.image(image_kayan, width=250)

if __name__ ==  "__main__":
    main()