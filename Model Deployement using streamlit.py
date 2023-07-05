# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 19:15:17 2023

@author: Xps
"""

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2 


# Définition de la fonction dice_coef
def dice_coef(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    smooth = 1e-5  # Pour éviter une division par zéro
    dice = (2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    return dice

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


# Chargement du modèle pré-entraîné
with tf.keras.utils.custom_object_scope({'dice_coef': dice_coef}):
    with tf.keras.utils.custom_object_scope({'iou': iou}):
        resid = tf.keras.models.load_model('C:/Users/User_ensias/Desktop/DATA/Final code/residual unet.h5')
        unet = tf.keras.models.load_model('C:/Users/User_ensias/Desktop/DATA/Final code/unet_augmented.h5')
        itern = tf.keras.models.load_model('C:/Users/User_ensias/Desktop/DATA/Final code/IterNet.h5')
# Page d'accueil
def home():
    # Titre de la page
    st.title("Welcome to our Application : 🚀  ")

    # Article
    st.header(" Deep Learning Approaches for Retinal Vessel Segmentation: A Comparative Study! ⚕️  ")
    st.header('')
    st.subheader('Prepared By : ')
    st.markdown("- Boussairi Hamza ")
    st.markdown("- ElYoussfi Rihab")
    st.markdown("- Ramiz Salma ")
    st.markdown("- Kabbaj Soufiane ")
    st.write("\n\n")

        # Header 1: Introduction
    st.header('The Retina and its Vasculature')
    # Paragraph 1: Introduction to the retina and its vasculature
    st.subheader('Introduction')
    st.write('The retina is a vital part of the visual system, responsible for capturing and '
             'transmitting visual information to the brain. It consists of several layers, '
             'including the innermost layer called the neurosensory retina. This layer contains '
             'various structures, including blood vessels that supply oxygen and nutrients to the '
             'retinal tissues.')
    
    # Header 2: Importance of Retinal Vessel Segmentation
    st.header('Importance of Retinal Vessel Segmentation in Clinical Practice ')
    
    # Paragraph 2: Explaining the importance of retinal vessel segmentation
    st.subheader('Clinical Significance')
    st.write('Accurate segmentation of retinal blood vessels plays a crucial role in various '
             'clinical applications. It provides valuable information about the health of the '
             'retina and can assist in the diagnosis, monitoring, and treatment of various eye '
             'diseases, such as diabetic retinopathy, macular degeneration, and glaucoma. By '
             'segmenting the vessels, clinicians can analyze their morphology, detect abnormalities, '
             'and track changes over time, leading to better patient care.')
    
    # Header 3: Challenges and Limitations
    st.header('Challenges and Limitations in Manual Segmentation Methods')
    
    # Paragraph 3: Discussing the challenges and limitations of manual segmentation
    st.subheader('Manual Segmentation Challenges')
    st.write('Manual segmentation of retinal blood vessels can be a time-consuming and labor-intensive '
             'task, requiring trained experts to annotate each vessel pixel. Moreover, it is prone '
             'to subjectivity and inter-observer variability, which can affect the reliability and '
             'consistency of the results. Additionally, manual segmentation may not scale well to '
             'large datasets, making it impractical for real-world clinical applications.')
    
    # Conclusion
    st.subheader('Conclusion')
    st.write('Advancements in deep learning techniques, such as UNET, ITERNET, and Residual UNET, '
             'have shown promise in automating the retinal vessel segmentation process. By leveraging '
             'these models and combining them with pre-processing techniques, such as applying filters '
             'from the OpenCV library, more accurate and efficient segmentation of retinal blood vessels '
             'can be achieved, leading to improved clinical outcomes.')
    # Bouton pour essayer le modèle
    st.write('\n\n')
    st.subheader(' 👋 To try our models, click the button below👋 ')

    # Using the cool button
    if st.button("Click me!", key="cool-button"):
        st.session_state.page = "run_model"
        
        
        
        
def run_model():
    st.title("Testez le fruit de notre projet! 🌟 ")
    st.write("Veuillez télécharger une image pour effectuer la segmentation.")

    # Uploader l'image
    uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Afficher l'image téléchargée
        image = Image.open(uploaded_file)
        
    
        # Prétraitement de l'image
        image = image.resize((256, 256))  # Redimensionner l'image si nécessaire
        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)  # Convertir de RGB à BGR
        image_array = image_array.astype('float32') / 255.0  # Normaliser les valeurs des pixels entre 0 et 1
        image_array = np.expand_dims(image_array, axis=0)  # Ajouter une dimension pour correspondre à la forme d'entrée du modèle
    
        # Prédiction de l'image en utilisant unet
        unet_image = unet.predict(image_array)
        # Conversion de l'image prédite en format image
        unet_image = unet_image[0] * 255.0  # Rétablir les valeurs des pixels à l'échelle originale
        unet_image = unet_image.astype(np.uint8)  # Convertir les valeurs des pixels en entiers non signés (0-255)
        unet_image = cv2.cvtColor(unet_image, cv2.COLOR_BGR2RGB)  # Convertir de BGR à RGB
        unet_image = Image.fromarray(unet_image)
        
        # Prédiction de l'image en utilisant residual unet
        resid_image = resid.predict(image_array)
        # Conversion de l'image prédite en format image
        resid_image = resid_image[0] * 255.0  # Rétablir les valeurs des pixels à l'échelle originale
        resid_image = resid_image.astype(np.uint8)  # Convertir les valeurs des pixels en entiers non signés (0-255)
        resid_image = cv2.cvtColor(resid_image, cv2.COLOR_BGR2RGB)  # Convertir de BGR à RGB
        resid_image = Image.fromarray(resid_image)
        
        # Prédiction de l'image en utilisant Iternet
        itern_image = itern.predict(image_array)
        # Conversion de l'image prédite en format image
        itern_image = itern_image[0] * 255.0  # Rétablir les valeurs des pixels à l'échelle originale
        itern_image = itern_image.astype(np.uint8)  # Convertir les valeurs des pixels en entiers non signés (0-255)
        itern_image = cv2.cvtColor(itern_image, cv2.COLOR_BGR2RGB)  # Convertir de BGR à RGB
        itern_image = Image.fromarray(itern_image)
        
        col1, col2, col3, col4 = st.columns(4)
        # Afficher la première image dans la première colonne
        with col1:
            st.image(image, caption='Image téléchargée', use_column_width=True)
        
        # Afficher la deuxième image dans la deuxième colonne
        with col2:
            st.image(unet_image, caption='Image prédite par Unet', use_column_width=True)
            
        with col3:
            st.image(resid_image, caption='Image prédite par Residual Unet', use_column_width=True)
        
        with col4:
            st.image(itern_image, caption='Image prédite par Iternet', use_column_width=True)
            
        if st.button("Revenir à l'accueil"):
            st.session_state.page = "home"

if "page" not in st.session_state:
    st.session_state.page = "home"
# Lancement de l'application Streamlit
#if __name__ == '__main__':
 #   home()

if __name__ == '__main__':
    if st.session_state.page == "home":
        home()
    elif st.session_state.page == "run_model":
        run_model()