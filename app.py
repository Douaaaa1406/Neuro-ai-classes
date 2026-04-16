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

# --- 2. DESIGN CYBER-BIOMÉDICAL (NOIR, VERT, BLEU) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;600;800&family=Space+Grotesk:wght@300;500;700&display=swap');

    /* === FOND PRINCIPAL : NOIR PROFOND === */
    [data-testid="stAppViewContainer"] {
        background-color: #000000;
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* === SUPPRESSION DES BOITES ET RECTANGLES INUTILES === */
    [data-testid="column"], [data-testid="stVerticalBlock"], .stColumn > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        max-width: 1350px;
    }

    /* === EN-TÊTE EN GRADIENT VIF === */
    .main-header {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #00d4ff 0%, #00ff87 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
        filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.4));
    }
    
    .sub-text {
        color: #00ff87;
        text-align: center;
        font-size: 1em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 6px;
        margin-top: 0px;
    }

    /* === TITRES DE SECTIONS === */
    .section-title {
        color: #00d4ff;
        font-size: 1.5em;
        font-weight: 700;
        margin-bottom: 20px;
        border-left: 4px solid #00ff87;
        padding-left: 15px;
    }

    /* === CORRECTION CRITIQUE : LISIBILITÉ DES CARREAUX === */
    /* On force un fond clair et un texte sombre pour que ce soit lisible à 100% */
    .stTextInput div div input, 
    .stNumberInput div div input, 
    .stSelectbox div div select {
        background-color: #ffffff !important; /* Fond Blanc */
        color: #001d3d !important; /* Texte Bleu Nuit très foncé */
        border: 2px solid #00d4ff !important;
        border-radius: 15px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
    }

    .stTextInput div div input:focus {
        border-color: #00ff87 !important;
        box-shadow: 0 0 10px rgba(0, 255, 135, 0.5) !important;
    }

    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #ffffff !important;
        font-weight: 500 !important;
        margin-bottom: 8px !important;
    }

    /* === BOUTON DE DIAGNOSTIC VIF === */
    div.stButton > button {
        background: linear-gradient(135deg, #00ff87 0%, #00d4ff 100%) !important;
        color: #000000 !important;
        border-radius: 50px !important;
        font-weight: 800 !important;
        font-size: 1.1em !important;
        height: 55px;
        width: 100%;
        border: none !important;
        box-shadow: 0 5px 20px rgba(0, 255, 135, 0.3);
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 30px rgba(0, 255, 135, 0.5);
        background: #ffffff !important;
    }

    /* === ZONE RÉSULTAT NÉON === */
    .res-card {
        background: rgba(0, 255, 135, 0.1);
        border: 2px solid #00ff87;
        border-radius: 30px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }

    /* === FOOTER === */
    .footer-text {
        text-align: center;
        color: rgba(255, 255, 255, 0.4);
        margin-top: 50px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGIQUE TECHNIQUE (MODÈLE & PDF) ---
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
    # Design du PDF plus sobre pour l'impression médicale
    pdf.set_font("Arial", 'B', 22)
    pdf.set_text_color(0, 50, 150)
    pdf.cell(0, 25, "RAPPORT CLINIQUE - NEUROSCAN AI", 0, 1, 'C')
    
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Généré le : {date_str}", 0, 1, 'R')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. INFORMATIONS PATIENT", 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Nom complet : {nom} {prenom}", 0, 1)
    pdf.cell(0, 10, f"Age / Sexe : {age} ans / {gender}", 0, 1)
    
    pdf.ln(10)
    img.save("temp_report.png")
    pdf.image("temp_report.png", x=60, w=90)
    pdf.ln(10)
    
    pdf.set_fill_color(230, 245, 255)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 20, f"CONCLUSION : {resultat.upper()} ({confiance:.2f}%)", 1, 1, 'C', True)
    
    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFACE ---
algeria_tz = pytz.timezone('Africa/Algiers')
date_str = datetime.datetime.now(algeria_tz).strftime("%d/%m/%Y | %H:%M")

st.markdown('<p class="main-header">NEUROSCAN AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Plateforme de Diagnostic Cérébral</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<p class="section-title">👤 Patient</p>', unsafe_allow_html=True)
    nom = st.text_input("NOM DE FAMILLE").upper()
    prenom = st.text_input("PRÉNOM").capitalize()
    age = st.number_input("ÂGE", 0, 120, 25)
    gender = st.selectbox("GENRE", ["Masculin", "Féminin"])

with col2:
    st.markdown('<p class="section-title">🔬 Acquisition IRM</p>', unsafe_allow_html=True)
    up = st.file_uploader("Upload", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if up:
        image = Image.open(up).convert('RGB')
        st.image(image, use_container_width=True)
    else:
        st.info("Attente du fichier IRM...")

with col3:
    st.markdown('<p class="section-title">⚡ Analyse IA</p>', unsafe_allow_html=True)
    if up and st.button("LANCER LE DIAGNOSTIC"):
        with st.spinner("Analyse des tissus..."):
            model = load_neuro_model()
            img_array = np.array(image.resize((224, 224))) / 255.0
            preds = model.predict(np.expand_dims(img_array, axis=0))[0]
            classes = ['Non-Cérébral', 'Gliome', 'Méningiome', 'Pas de Tumeur', 'Pituitaire']
            res, conf = classes[np.argmax(preds)], float(np.max(preds)) * 100
            
            st.markdown(f'''
                <div class="res-card">
                    <h2 style="color:#00ff87; margin:0;">{res}</h2>
                    <p style="color:white; opacity:0.8; font-size:0.9em;">Confiance : {conf:.2f}%</p>
                </div>
            ''', unsafe_allow_html=True)
            
            pdf_bytes = generate_medical_pdf(nom, prenom, age, gender, res, conf, image, date_str)
            st.download_button("📥 TÉLÉCHARGER LE RAPPORT PDF", pdf_bytes, f"NeuroScan_{nom}.pdf", "application/pdf")
    else:
        st.write("Prêt pour l'acquisition.")

    st.markdown(f'''
        <div style="text-align:right; margin-top:100px;">
            <a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank" 
               style="color:#00d4ff; text-decoration:none; font-weight:700; border-bottom: 2px solid #00ff87;">
               CONSULTER L'EXPERT ↗
            </a>
        </div>
    ''', unsafe_allow_html=True)

st.markdown(f'<p class="footer-text">NeuroScan v2.6 | Bahlouli Fatna Romaisaa & Houbad Douaa | M1 EMB | {date_str}</p>', unsafe_allow_html=True)
