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

# --- 2. DESIGN "DAYLIGHT BIOMEDICAL" (SANS MAUVE, SANS RECTANGLES) ---
st.markdown("""
<style>
    /* === FONTS === */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&family=Outfit:wght@400;700&display=swap');

    /* === FOND CLAIR DU JOUR === */
    [data-testid="stAppViewContainer"] {
        background-color: #f0f4f8; /* Gris très clair bleuté */
        background-image: radial-gradient(#d1d9e6 1px, transparent 1px);
        background-size: 30px 30px;
        color: #1a202c;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* === NETTOYAGE DES STRUCTURES === */
    [data-testid="column"], [data-testid="stVerticalBlock"], .stColumn > div {
        background-color: transparent !important;
        border: none !important;
    }

    .block-container {
        padding-top: 2rem !important;
        max-width: 1400px;
    }

    /* === EN-TÊTE PROFESSIONNEL VIF === */
    .main-header {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #0052cc 0%, #00a3ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4.2em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
        filter: drop-shadow(0 2px 4px rgba(0, 82, 204, 0.1));
    }
    
    .sub-text {
        color: #0052cc;
        text-align: center;
        font-size: 1.1em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 5px;
        margin-top: 5px;
        opacity: 0.8;
    }

    .live-date {
        text-align: center;
        color: #4a5568;
        font-size: 0.9em;
        margin-bottom: 40px;
        font-weight: 500;
    }

    /* === CARTES "ORGANIC WHITE" (COURBURES PRONONCÉES) === */
    .art-card {
        background: #ffffff;
        border-radius: 50px; /* Élimination des rectangles */
        padding: 40px;
        border: 1px solid rgba(0, 82, 204, 0.08);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.05);
        min-height: 560px;
        transition: all 0.4s ease;
    }

    .art-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 25px 50px rgba(0, 82, 204, 0.1);
    }

    .column-title {
        font-family: 'Outfit', sans-serif;
        color: #0052cc;
        font-size: 1.7em;
        font-weight: 700;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    /* === INPUTS ET CARREAUX (VISIBILITÉ MAXIMALE) === */
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #2d3748 !important;
        font-weight: 700 !important;
        font-size: 0.95em !important;
        margin-bottom: 8px !important;
    }

    .stTextInput div div input, 
    .stNumberInput div div input, 
    .stSelectbox div div select {
        background-color: #f7fafc !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 20px !important;
        color: #1a202c !important;
        padding: 12px 20px !important;
        font-weight: 500 !important;
    }

    .stTextInput div div input:focus {
        border-color: #0052cc !important;
        background-color: #ffffff !important;
        box-shadow: 0 0 0 3px rgba(0, 82, 204, 0.1) !important;
    }

    /* === BOUTON VIF (BLEU ÉLECTRIQUE) === */
    div.stButton > button {
        background: linear-gradient(135deg, #0052cc 0%, #00a3ff 100%) !important;
        color: #ffffff !important;
        border-radius: 30px !important;
        font-weight: 700 !important;
        font-size: 1.1em !important;
        height: 60px;
        width: 100%;
        border: none !important;
        box-shadow: 0 8px 20px rgba(0, 82, 204, 0.2);
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    div.stButton > button:hover {
        box-shadow: 0 12px 25px rgba(0, 82, 204, 0.3);
        transform: scale(1.02);
    }

    /* === DIAGNOSTIC RESULT (VERT VIF BIOMÉDICAL) === */
    .diagnostic-result {
        background: #ebfaf2;
        border-radius: 30px;
        padding: 25px;
        border: 2px solid #00c853;
        text-align: center;
        margin-top: 20px;
    }

    .result-text {
        color: #00c853;
        font-size: 1.8em;
        font-weight: 800;
        margin: 0;
    }

    /* === FOOTER === */
    .footer-text {
        text-align: center;
        color: #718096;
        margin-top: 50px;
        padding-bottom: 20px;
        font-weight: 500;
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
    pdf.set_font("Arial", 'B', 22)
    pdf.set_text_color(0, 82, 204)
    pdf.cell(0, 25, "RAPPORT CLINIQUE - NEUROSCAN AI", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Date d'analyse : {date_str}", 0, 1, 'R')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. IDENTIFICATION DU PATIENT", 0, 1)
    pdf.set_font("Arial", '', 12)
    for k, v in [["Nom complet", f"{nom} {prenom}"], ["Âge / Sexe", f"{age} / {gender}"], ["Système", "NeuroScan Intelligence V2.0"]]:
        pdf.cell(60, 10, k, 1); pdf.cell(100, 10, v, 1); pdf.ln()
    
    pdf.ln(10)
    img.save("temp_scan.png")
    pdf.image("temp_scan.png", x=60, w=90)
    pdf.ln(10)
    
    pdf.set_fill_color(235, 250, 242)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 20, f"CONCLUSION : {resultat.upper()} ({confiance:.2f}%)", 1, 1, 'C', True)
    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFACE ---
algeria_tz = pytz.timezone('Africa/Algiers')
now = datetime.datetime.now(algeria_tz)
date_str = now.strftime("%d/%m/%Y | %H:%M")

st.markdown('<p class="main-header">NeuroScan AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Plateforme de Diagnostic Cérébral</p>', unsafe_allow_html=True)
st.markdown(f'<p class="live-date">Session active • {date_str}</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">👤 Patient</div>', unsafe_allow_html=True)
    nom = st.text_input("NOM", placeholder="Ex: HOUBAD").upper()
    prenom = st.text_input("PRÉNOM", placeholder="Ex: Douaa").capitalize()
    age = st.number_input("ÂGE", 0, 120, 25)
    gender = st.selectbox("GENRE", ["Masculin", "Féminin", "Autre"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">🔬 Imagerie IRM</div>', unsafe_allow_html=True)
    up = st.file_uploader("Charger le scan", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if up:
        image = Image.open(up).convert('RGB')
        st.image(image, use_container_width=True)
    else:
        st.info("Veuillez insérer un scan IRM pour analyse.")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">⚕️ Diagnostic IA</div>', unsafe_allow_html=True)
    
    if up and st.button("LANCER L'ANALYSE"):
        with st.spinner("Analyse des tissus cérébraux..."):
            model = load_neuro_model()
            img_array = np.array(image.resize((224, 224))) / 255.0
            preds = model.predict(np.expand_dims(img_array, axis=0))[0]
            classes = ['Non-Cérébral', 'Gliome', 'Méningiome', 'Pas de Tumeur', 'Pituitaire']
            res, conf = classes[np.argmax(preds)], float(np.max(preds)) * 100
            
            st.markdown(f'''
                <div class="diagnostic-result">
                    <p style="color:#2d3748; margin:0; font-weight:600;">RÉSULTAT DÉTECTÉ</p>
                    <h2 class="result-text">{res}</h2>
                    <p style="color:#4a5568; margin:0; font-size:0.9em;">Indice de confiance : {conf:.2f}%</p>
                </div>
            ''', unsafe_allow_html=True)
            
            pdf_data = generate_medical_pdf(nom, prenom, age, gender, res, conf, image, date_str)
            st.download_button("📥 TÉLÉCHARGER LE RAPPORT", pdf_data, f"Rapport_Neuro_{nom}.pdf", "application/pdf")
    else:
        st.warning("En attente des données d'entrée...")
    
    st.markdown(f'''
        <div style="text-align:right; margin-top:80px;">
            <a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank" 
               style="color:#0052cc; text-decoration:none; font-weight:700; font-size:1em;">
               CONSULTER LE PROFIL SCIENTIFIQUE ↗
            </a>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'''
    <p class="footer-text">
        Développé par <strong>Bahlouli Fatna Romaisaa</strong> & <strong>Houbad Douaa</strong> | M1 EMB | 2026
    </p>
''', unsafe_allow_html=True)
