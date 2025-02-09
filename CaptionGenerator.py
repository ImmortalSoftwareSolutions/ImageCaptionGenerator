import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from nltk.translate.bleu_score import corpus_bleu
from PIL import Image
import matplotlib.pyplot as plt

# Define paths
WORKING_DIR = ''  # Ensure this is the correct model directory

# Load the tokenizer
with open(os.path.join(WORKING_DIR, 'tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)

# Load the trained model
model = load_model(os.path.join(WORKING_DIR, 'model.keras'))

# Load VGG16 model for feature extraction
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Define max sequence length (Ensure this matches training)
max_length = 35

# Function to convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to extract image features using VGG16
def extract_features(img_path):
    image = load_img(img_path, target_size=(224, 224))  # Resize image
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)  # Preprocess for VGG16

    feature = vgg_model.predict(image, verbose=0)  # Extract features
    return feature  # Shape: (1, 4096)

# Function to predict caption for an image
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)

        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

# Function to generate caption for any image
def generate_caption(img_path):
    if not os.path.exists(img_path):
        print(f"Error: Image '{img_path}' not found.")
        return

    image = Image.open(img_path)
    
    # Extract features
    feature = extract_features(img_path)  # Extract VGG16 features
    
    # Generate predicted caption
    y_pred = predict_caption(model, feature, tokenizer, max_length)

    print('--------------------Predicted--------------------')
    #print(y_pred)
    
    
    #to remove "startseq" and "endseq" from results
    list_pred = list(y_pred)
    list_pred = list_pred[9:-7]
    text_pred = ''
    for i in range(len(list_pred)):
        text_pred+=list_pred[i]
    
    print(text_pred)
    return text_pred
    
    # Display image
    # plt.imshow(image)
    # plt.axis("off")
    # plt.show()

# Give path of image
generate_caption("Path_to_Image")
