import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from fpdf import FPDF
import datetime
import os
import gdown
import pytz

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Brain Tumor Classification AI", page_icon="🧠", layout="wide")

# --- 2. DESIGN "ULTRA-LUXURY GOLD" (FIDÈLE À L'IMAGE) ---
st.markdown("""
<style>
    /* Background avec effet de texture et particules */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at center, #f5eedc 0%, #d4c4a8 100%);
        background-image: url("https://www.transparenttextures.com/patterns/dust.png");
    }

    .main-header {
        font-family: 'Inter', sans-serif;
        color: #7d5a2d;
        font-size: 3.5em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
    }
    
    .sub-text {
        color: #8b6b43;
        text-align: center;
        font-size: 1.1em;
        margin-bottom: 5px;
    }

    .system-status {
        background: rgba(255, 255, 255, 0.5);
        border: 1px solid #7d5a2d;
        border-radius: 50px;
        padding: 5px 25px;
        color: #7d5a2d;
        text-align: center;
        width: fit-content;
        margin: 0 auto 30px auto;
        font-size: 0.8em;
        font-weight: bold;
    }

    /* Cartes Beige Soft */
    .gold-card {
        background: rgba(239, 230, 213, 0.9);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid #c4b596;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        height: 100%;
    }

    /* En-têtes de colonnes */
    .column-title {
        background: #8b6b43;
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
        font-size: 0.85em;
        font-weight: bold;
        margin-bottom: 20px;
        text-transform: uppercase;
    }

    /* Barres de probabilités personnalisées (Comme l'image) */
    .custom-progress-container {
        width: 100%;
        background-color: #d1c4ab;
        border-radius: 5px;
        margin: 10px 0;
        height: 12px;
    }
    .custom-progress-fill {
        height: 100%;
        border-radius: 5px;
        background: linear-gradient(90deg, #d4a373, #7d5a2d);
    }

    /* Bouton Analyze style Image */
    div.stButton > button {
        background: linear-gradient(180deg, #d4a373 0%, #8b6b43 100%) !important;
        color: white !important;
        border: 1px solid #7d5a2d !important;
        padding: 15px !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        font-size: 1.1em !important;
        box-shadow: 0 4px 15px rgba(139, 107, 67, 0.3) !important;
    }

    /* LinkedIn Button */
    .linkedin-btn {
        background: linear-gradient(180deg, #0077b5, #005a87);
        color: white !important;
        padding: 10px 20px;
        border-radius: 5px;
        text-decoration: none;
        font-weight: bold;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        float: right;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_neuro_model():
    model_path = 'brain_tumor_model_v6_final.keras'
    if not os.path.exists(model_path):
        file_id = '1QRVvhNHSx7qgw0GIDrRLsuX09uItsXM2' 
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
    
    # Re-construction de l'architecture pour charger les poids
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    model.load_weights(model_path)
    return model

# --- 4. NAVIGATION ---
algeria_tz = pytz.timezone('Africa/Algiers')
now = datetime.datetime.now(algeria_tz)

st.markdown('<p class="main-header">Brain Tumor Classification AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">MRI-Based Tumor Type Prediction</p>', unsafe_allow_html=True)
st.markdown(f'<div class="system-status">SYSTEM STATUS: ACTIVE | ALGERIA | {now.strftime("%d/%m/%Y - %H:%M:%S")}</div>', unsafe_allow_html=True)

# --- 5. INTERFACE ---
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="gold-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">👤 PATIENT INFORMATION</div>', unsafe_allow_html=True)
    nom = st.text_input("LAST NAME").upper()
    prenom = st.text_input("FIRST NAME").capitalize()
    age = st.number_input("AGE", min_value=0, value=30)
    gender = st.selectbox("GENDER", ["Male", "Female"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="gold-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">📤 UPLOAD MRI ACQUISITION</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Scan", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.markdown('<div style="border:3px solid #8b6b43; padding:5px; background:white;">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("<center>Current Scan</center>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="gold-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">🧬 PREDICTION RESULT</div>', unsafe_allow_html=True)
    if uploaded_file and st.button("Analyze"):
        model = load_neuro_model()
        img_array = np.array(image.resize((224, 224))) / 255.0
        preds = model.predict(np.expand_dims(img_array, axis=0))[0]
        
        classes = ['Non-Brain', 'Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        idx = np.argmax(preds)
        
        # --- CORRECTION DU BUG PROGRESS BAR ---
        confiance_val = float(preds[idx]) # Conversion explicite en float Python
        
        st.markdown(f"**Predicted Type:** {classes[idx]}")
        st.markdown(f"**Confidence:** {confiance_val*100:.1f}%")
        
        # Barre personnalisée style "Image"
        st.markdown(f"""
            <div class="custom-progress-container">
                <div class="custom-progress-fill" style="width: {confiance_val*100}%;"></div>
            </div>
        """, unsafe_allow_html=True)

        # Détails des autres probabilités
        for i, class_name in enumerate(classes):
            if i != idx:
                prob = float(preds[i]) * 100
                st.write(f"{class_name}: {prob:.1f}%")
    else:
        st.write("Awaiting data...")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank" class="linkedin-btn">
             For more information
        </a>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<p style="text-align:center; padding:20px; color:#8b6b43; font-weight:bold;">custom footer</p>', unsafe_allow_html=True)
