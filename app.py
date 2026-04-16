import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from fpdf import FPDF
import datetime
import os
import gdown
import pytz
import io

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="wide")

# --- 2. DESIGN "FLUID & BORDERLESS" (MODE CLAIR) ---
st.markdown("""
<style>
    /* === FONTS === */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');

    /* === FOND TOTALEMENT PROPRE === */
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
        color: #1a202c;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* === ÉLIMINATION TOTALE DES RECTANGLES ET BORDURES === */
    [data-testid="column"], [data-testid="stVerticalBlock"], .stColumn > div, div[data-testid="stExpander"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Suppression des bordures par défaut de Streamlit */
    .stApp div[data-baseweb="card"] {
        border: none !important;
    }

    .block-container {
        padding-top: 2rem !important;
        max-width: 1200px;
    }

    /* === TITRE FLOTTANT VIF === */
    .main-header {
        background: linear-gradient(90deg, #0061ff 0%, #60efff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    
    .sub-text {
        color: #0061ff;
        text-align: center;
        font-size: 1em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 5px;
        margin-bottom: 40px;
    }

    /* === TITRES DE SECTIONS SANS BOITES === */
    .section-title {
        color: #0061ff;
        font-size: 1.5em;
        font-weight: 700;
        margin-bottom: 20px;
        border-left: 5px solid #00d4ff;
        padding-left: 15px;
    }

    /* === INPUTS STYLE "PILULE" (EXTRÊMEMENT ARRONDIS) === */
    .stTextInput div div input, 
    .stNumberInput div div input, 
    .stSelectbox div div select {
        background-color: #f0f7ff !important;
        border: 1px solid #cce5ff !important;
        border-radius: 50px !important; /* Forme de pilule, pas de rectangle */
        padding: 10px 25px !important;
        color: #1a202c !important;
    }

    /* === BOUTON D'ACTION VIF === */
    div.stButton > button {
        background: linear-gradient(135deg, #0061ff 0%, #00d4ff 100%) !important;
        color: white !important;
        border-radius: 50px !important;
        font-weight: 700 !important;
        height: 55px;
        width: 100%;
        border: none !important;
        box-shadow: 0 10px 20px rgba(0, 97, 255, 0.2);
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(0, 97, 255, 0.3);
    }

    /* === RÉSULTAT FLOTTANT === */
    .res-box {
        background: #e6fffa;
        color: #2c7a7b;
        padding: 20px;
        border-radius: 30px;
        text-align: center;
        font-weight: 800;
        font-size: 1.4em;
        border: 2px solid #38b2ac;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGIQUE TECHNIQUE ---
@st.cache_resource
def load_neuro_model():
    model_path = 'brain_tumor_model_v6_final.keras'
    if not os.path.exists(model_path):
        gdown.download(f'https://drive.google.com/uc?id=1QRVvhNHSx7qgw0GIDrRLsuX09uItsXM2', model_path, quiet=False)
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

def generate_medical_pdf(nom, prenom, age, gender, resultat, confiance, img, date_str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(0, 97, 255)
    pdf.cell(0, 20, "RAPPORT D'ANALYSE NEUROSCAN", 0, 1, 'C')
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Patient : {nom} {prenom} | Age : {age} | Sexe : {gender}", 0, 1)
    pdf.cell(0, 10, f"Résultat : {resultat} (Confiance : {confiance:.2f}%)", 0, 1)
    pdf.ln(10)
    img.save("temp.png")
    pdf.image("temp.png", x=60, w=90)
    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFACE UTILISATEUR ---
algeria_tz = pytz.timezone('Africa/Algiers')
date_now = datetime.datetime.now(algeria_tz).strftime("%d/%m/%Y - %H:%M")

st.markdown('<p class="main-header">NeuroScan AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Biomedical Engineering System</p>', unsafe_allow_html=True)

# Division en colonnes sans aucun cadre (les colonnes servent juste au placement)
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<p class="section-title">Informations</p>', unsafe_allow_html=True)
    nom = st.text_input("NOM").upper()
    prenom = st.text_input("PRÉNOM").capitalize()
    age = st.number_input("ÂGE", 0, 120, 25)
    gender = st.selectbox("GENRE", ["Masculin", "Féminin"])

with col2:
    st.markdown('<p class="section-title">Visualisation</p>', unsafe_allow_html=True)
    up = st.file_uploader("Upload", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if up:
        image = Image.open(up).convert('RGB')
        st.image(image, use_container_width=True)
    else:
        st.info("Attente du scan IRM...")

with col3:
    st.markdown('<p class="section-title">Analyse</p>', unsafe_allow_html=True)
    if up and st.button("LANCER LE DIAGNOSTIC"):
        with st.spinner("Analyse en cours..."):
            model = load_neuro_model()
            img_array = np.array(image.resize((224, 224))) / 255.0
            preds = model.predict(np.expand_dims(img_array, axis=0))[0]
            classes = ['Non-Cérébral', 'Gliome', 'Méningiome', 'Pas de Tumeur', 'Pituitaire']
            res, conf = classes[np.argmax(preds)], float(np.max(preds)) * 100
            
            st.markdown(f'<div class="res-box">{res}<br><span style="font-size:0.6em; opacity:0.8;">Confiance: {conf:.2f}%</span></div>', unsafe_allow_html=True)
            
            pdf_data = generate_medical_pdf(nom, prenom, age, gender, res, conf, image, date_now)
            st.download_button("📥 TÉLÉCHARGER RAPPORT", pdf_data, f"Neuro_{nom}.pdf", "application/pdf")
    else:
        st.write("Prêt pour l'analyse.")

st.markdown(f'<div style="text-align:center; margin-top:100px; color:#a0aec0; font-size:0.8em;">{date_now} | Développé par Douaa & Romaisaa</div>', unsafe_allow_html=True)
