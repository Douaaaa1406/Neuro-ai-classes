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
st.set_page_config(page_title="NeuroScan AI | Diagnostic", page_icon="🧠", layout="wide")

# --- 2. DESIGN BIOMÉDICAL PRO (VIF & ARRONDIT) ---
st.markdown("""
<style>
    /* === FONTS === */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&family=Outfit:wght@400;700&display=swap');

    /* === FOND PRINCIPAL : DEEP MEDICAL BLUE === */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #001a2c 0%, #000a12 100%);
        color: #ffffff;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* === NETTOYAGE DES STRUCTURES RECTANGULAIRES === */
    [data-testid="column"], [data-testid="stVerticalBlock"], .stColumn > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        max-width: 1350px;
    }

    /* === EN-TÊTE ULTRA-MODERNE === */
    .main-header {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
        filter: drop-shadow(0 2px 10px rgba(0, 242, 254, 0.3));
    }
    
    .sub-text {
        color: #00d4ff;
        text-align: center;
        font-size: 1.1em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 6px;
        margin-top: 0px;
        opacity: 0.9;
    }

    .live-date {
        text-align: center;
        color: rgba(0, 212, 255, 0.6);
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.9em;
        margin-bottom: 40px;
    }

    /* === CARTES "BIO-CURVE" (FINI LES RECTANGLES) === */
    .art-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border-radius: 40px; /* Arrondi très prononcé */
        padding: 40px;
        border: 1px solid rgba(0, 242, 254, 0.15);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        min-height: 550px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .art-card:hover {
        transform: translateY(-10px) scale(1.02);
        border: 1px solid rgba(0, 242, 254, 0.4);
        background: rgba(255, 255, 255, 0.05);
    }

    .column-title {
        font-family: 'Outfit', sans-serif;
        color: #00f2fe;
        font-size: 1.6em;
        font-weight: 700;
        margin-bottom: 30px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* === INPUTS & WIDGETS ARRONDIS === */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(0, 242, 254, 0.05) !important;
        border: 1px solid rgba(0, 242, 254, 0.2) !important;
        border-radius: 20px !important; /* Cercles plutôt que carrés */
        color: #ffffff !important;
        padding: 12px 20px !important;
    }

    /* === BOUTON VIF (CYAN) === */
    div.stButton > button {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
        color: #001a2c !important;
        border-radius: 30px !important;
        font-weight: 700 !important;
        font-size: 1.1em !important;
        height: 60px;
        border: none !important;
        box-shadow: 0 10px 20px rgba(0, 242, 254, 0.3);
        transition: all 0.3s ease !important;
    }

    div.stButton > button:hover {
        box-shadow: 0 15px 30px rgba(0, 242, 254, 0.5);
        transform: scale(1.05);
    }

    /* === RÉSULTATS DIAGNOSTIC === */
    .diagnostic-result {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.1), rgba(79, 172, 254, 0.1));
        border-radius: 25px;
        padding: 25px;
        border: 1px solid #00f2fe;
        margin-top: 20px;
        text-align: center;
    }

    /* === FOOTER === */
    .footer-text {
        text-align: center;
        color: rgba(255, 255, 255, 0.5);
        margin-top: 50px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGIQUE TECHNIQUE (INCHANGÉE) ---
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
    pdf.set_text_color(0, 80, 120)
    pdf.cell(0, 20, "RAPPORT CLINIQUE - NEUROSCAN AI", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Date : {date_str}", 0, 1, 'R')
    pdf.ln(5)
    pdf.cell(0, 10, "1. INFORMATIONS PATIENT", 0, 1)
    pdf.set_font("Arial", '', 11)
    data = [["Nom/Prenom", f"{nom} {prenom}"], ["Age/Genre", f"{age}/{gender}"], ["Statut", "Analyse IA terminée"]]
    for row in data:
        pdf.cell(50, 10, row[0], 1); pdf.cell(100, 10, row[1], 1); pdf.ln()
    pdf.ln(10)
    img.save("temp.png")
    pdf.image("temp.png", x=65, w=80)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 15); pdf.set_fill_color(230, 245, 255)
    pdf.cell(0, 15, f"RESULTAT : {resultat.upper()} ({confiance:.1f}%)", 1, 1, 'C', True)
    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFACE UTILISATEUR ---
algeria_tz = pytz.timezone('Africa/Algiers')
date_str = datetime.datetime.now(algeria_tz).strftime("%d/%m/%Y | %H:%M:%S")

st.markdown('<p class="main-header">NEUROSCAN AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Biomedical Intelligence</p>', unsafe_allow_html=True)
st.markdown(f'<p class="live-date">SYSTEM READY • {date_str}</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">👤 Patient</div>', unsafe_allow_html=True)
    nom = st.text_input("NOM", key="nom").upper()
    prenom = st.text_input("PRÉNOM", key="prenom").capitalize()
    age = st.number_input("ÂGE", min_value=0, max_value=120, value=30)
    gender = st.selectbox("GENRE", ["Masculin", "Féminin"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">🔬 Imagerie IRM</div>', unsafe_allow_html=True)
    up = st.file_uploader("Upload Scan", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if up:
        image = Image.open(up).convert('RGB')
        st.image(image, use_container_width=True)
    else:
        st.info("Sélectionnez une image IRM")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">⚕️ Analyse IA</div>', unsafe_allow_html=True)
    
    if up and st.button("DÉMARRER LE SCAN", key="analyze"):
        with st.spinner("Analyse neuronale..."):
            model = load_neuro_model()
            img_array = np.array(image.resize((224, 224))) / 255.0
            preds = model.predict(np.expand_dims(img_array, axis=0))[0]
            classes = ['Non-Cérébral', 'Gliome', 'Méningiome', 'Pas de Tumeur', 'Pituitaire']
            res, conf = classes[np.argmax(preds)], float(np.max(preds)) * 100
            
            st.markdown(f'''
                <div class="diagnostic-result">
                    <h2 style="color:#00f2fe;margin:0;">{res}</h2>
                    <p style="color:white;opacity:0.8;">Indice de confiance : {conf:.2f}%</p>
                </div>
            ''', unsafe_allow_html=True)
            
            pdf = generate_medical_pdf(nom, prenom, age, gender, res, conf, image, date_str)
            st.download_button("📥 GÉNÉRER LE RAPPORT", pdf, f"Neuro_{nom}.pdf", "application/pdf")
    else:
        st.info("Système en attente de données...")
    
    st.markdown(f'''
        <div style="text-align:right; margin-top:80px;">
            <a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank" 
               style="color:#00f2fe; text-decoration:none; font-weight:600;">Expertise LinkedIn ↗</a>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<p class="footer-text">NeuroScan v2.0 | Algiers High-Tech | 2026</p>', unsafe_allow_html=True)
