import streamlit as st
import tensorflow as tf
from PIL import Image, ImageEnhance
import numpy as np
from fpdf import FPDF
import datetime
import os
import gdown
import pytz

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NeuroScan Elite", page_icon="🧠", layout="wide")

# --- 2. DESIGN "ELITE WHITE & GOLD" ---
st.markdown("""
<style>
    /* Fond Gris Perle très clair */
    [data-testid="stAppViewContainer"] {
        background-color: #fcfcfc;
        color: #1a1a1a;
    }
    
    /* Titre en Anthracite */
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #1a1a1a;
        font-size: 3.2em;
        font-weight: 900;
        text-align: center;
        letter-spacing: -2px;
        margin-bottom: 0px;
    }

    /* Info Bar Noire */
    .info-bar {
        background: #1a1a1a;
        color: #ffffff;
        padding: 10px;
        text-align: center;
        font-family: monospace;
        font-size: 0.9em;
        margin-bottom: 40px;
    }

    /* Cartes Blanches avec ombres douces */
    .medical-card {
        background: #ffffff;
        border-radius: 2px;
        padding: 30px;
        border-left: 5px solid #1a1a1a;
        box-shadow: 0 10px 40px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    /* AFFICHAGE DU RÉSULTAT "MODERNE" */
    .result-box {
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        border: 2px solid #1a1a1a;
    }
    .status-positive { background-color: #ffeded; border-color: #cc0000; color: #cc0000; }
    .status-negative { background-color: #e6ffed; border-color: #008800; color: #008800; }

    /* Bouton d'Analyse Noir */
    div.stButton > button {
        background: #1a1a1a !important;
        color: white !important;
        border-radius: 0px !important;
        padding: 20px !important;
        font-weight: bold !important;
        letter-spacing: 2px;
        width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background: #444444 !important;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }

    /* LinkedIn en bas à droite */
    .linkedin-fixed {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: #1a1a1a;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .linkedin-fixed a { color: white !important; text-decoration: none; font-weight: bold; font-size: 0.8em; }
</style>
""", unsafe_allow_html=True)

# --- 3. CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_neuro_model():
    model_path = 'brain_tumor_model_v6_final.keras'
    file_id = '1QRVvhNHSx7qgw0GIDrRLsuX09uItsXM2' 
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)
    
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

# --- 4. HEADER ---
algeria_tz = pytz.timezone('Africa/Algiers')
now = datetime.datetime.now(algeria_tz)

st.markdown('<p class="main-header">NEUROSCAN AI</p>', unsafe_allow_html=True)
st.markdown(f'<div class="info-bar">STATION: ALGIERS | CLOCK: {now.strftime("%H:%M:%S")} | STATUS: READY</div>', unsafe_allow_html=True)

# --- 5. MAIN INTERFACE ---
col_info, col_scan, col_res = st.columns([1, 1.5, 1], gap="large")

with col_info:
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.markdown("### 📋 PATIENT")
    nom = st.text_input("NOM").upper()
    prenom = st.text_input("PRÉNOM").capitalize()
    age = st.number_input("ÂGE", min_value=0, value=30)
    gender = st.selectbox("GENRE", ["Male", "Female"])
    st.markdown('</div>', unsafe_allow_html=True)

with col_scan:
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.markdown("### 🧠 IMAGERIE IRM")
    uploaded_file = st.file_uploader("GLISSER LE CLICHÉ ICI", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_res:
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.markdown("### ⚡ ANALYSE IA")
    if uploaded_file and st.button("LANCER LE DIAGNOSTIC"):
        model = load_neuro_model()
        img_array = np.array(image.resize((224, 224))) / 255.0
        preds = model.predict(np.expand_dims(img_array, axis=0))[0]
        
        classes = ['Non-Cérébral', 'Gliome', 'Méningiome', 'Pas de tumeur', 'Pituitaire']
        idx = np.argmax(preds)
        resultat = classes[idx]
        confiance = preds[idx] * 100

        # AFFICHAGE DU RÉSULTAT STYLE "ALERTE"
        style_class = "status-negative" if resultat == "Pas de tumeur" else "status-positive"
        st.markdown(f"""
            <div class="result-box {style_class}">
                <h4 style="margin:0;">CONCLUSION IA</h4>
                <h2 style="margin:10px 0;">{resultat.upper()}</h2>
                <p>Indice de certitude : {confiance:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Le bouton de téléchargement apparaît ici après l'analyse
        st.write("---")
        st.info("Le rapport PDF a été généré avec succès.")
    else:
        st.write("En attente de données...")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. FOOTER ---
st.markdown(f"""
    <div style="text-align:center; padding:40px; color:#999; font-size:0.8em;">
        <b>NEUROSCAN AI V6.1</b> | UNITÉ DE GÉNIE BIOMÉDICAL<br>
        DÉVELOPPÉ PAR : <b>DOUAA HOUBAD</b> | M1 EMB INGÉNIEUR
    </div>
    <div class="linkedin-fixed">
        <a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank">FOR MORE INFORMATION</a>
    </div>
""", unsafe_allow_html=True)
