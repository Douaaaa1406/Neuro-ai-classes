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
st.set_page_config(page_title="NeuroScan AI | Pro", page_icon="🧠", layout="wide")

# --- 2. DESIGN "CYBER-BIOMEDICAL" (NOIR, VERT VIF, BLEU) ---
st.markdown("""
<style>
    /* === FONTS === */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;800&family=Space+Grotesk:wght@300;500;700&display=swap');

    /* === FOND NOIR PROFOND === */
    [data-testid="stAppViewContainer"] {
        background-color: #000000;
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* === SUPPRESSION TOTALE DES RECTANGLES === */
    [data-testid="column"], [data-testid="stVerticalBlock"], .stColumn > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        max-width: 1300px;
    }

    /* === TITRE EN GRADIENT BLEU & VERT === */
    .main-header {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #00d4ff 0%, #00ff87 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4.5em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
        filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.3));
    }
    
    .sub-text {
        color: #00ff87;
        text-align: center;
        font-size: 1.1em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 7px;
        margin-top: 0px;
        opacity: 0.9;
    }

    /* === TITRES DE SECTIONS (STYLE NÉON) === */
    .section-title {
        color: #00d4ff;
        font-size: 1.6em;
        font-weight: 700;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* === INPUTS "PILULE" (SANS ANGLES DROITS) === */
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #ffffff !important;
        font-weight: 500 !important;
        margin-left: 15px;
    }

    .stTextInput div div input, 
    .stNumberInput div div input, 
    .stSelectbox div div select {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 50px !important; /* Forme pilule */
        color: #ffffff !important;
        padding: 12px 25px !important;
    }

    .stTextInput div div input:focus {
        border-color: #00ff87 !important;
        box-shadow: 0 0 15px rgba(0, 255, 135, 0.2) !important;
    }

    /* === BOUTON VERT VIF (CYBER ACTION) === */
    div.stButton > button {
        background: linear-gradient(135deg, #00ff87 0%, #00d4ff 100%) !important;
        color: #000000 !important;
        border-radius: 50px !important;
        font-weight: 800 !important;
        font-size: 1.2em !important;
        height: 65px;
        width: 100%;
        border: none !important;
        box-shadow: 0 0 25px rgba(0, 255, 135, 0.4);
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        text-transform: uppercase;
    }

    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 0 40px rgba(0, 255, 135, 0.6);
        background: #ffffff !important;
    }

    /* === ZONE RÉSULTAT (VERT NÉON) === */
    .res-card {
        background: rgba(0, 255, 135, 0.1);
        border: 2px solid #00ff87;
        border-radius: 40px;
        padding: 30px;
        text-align: center;
        margin-top: 20px;
        animation: glow 2s infinite alternate;
    }

    @keyframes glow {
        from { box-shadow: 0 0 10px rgba(0, 255, 135, 0.2); }
        to { box-shadow: 0 0 25px rgba(0, 255, 135, 0.4); }
    }

    /* === FOOTER === */
    .footer {
        text-align: center;
        margin-top: 80px;
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGIQUE TECHNIQUE (TF & PDF) ---
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
    pdf.set_fill_color(0, 0, 0)
    pdf.rect(0, 0, 210, 297, 'F') # Fond noir pour le PDF pro
    pdf.set_font("Arial", 'B', 24)
    pdf.set_text_color(0, 212, 255)
    pdf.cell(0, 30, "NEUROSCAN AI - REPORT", 0, 1, 'C')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Date: {date_str}", 0, 1, 'R')
    pdf.ln(10)
    pdf.cell(0, 10, f"Patient: {nom} {prenom} | Age: {age} | Genre: {gender}", 0, 1)
    pdf.ln(10)
    img.save("temp_scan.png")
    pdf.image("temp_scan.png", x=55, w=100)
    pdf.ln(15)
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(0, 255, 135)
    pdf.cell(0, 20, f"RESULT: {resultat.upper()}", 1, 1, 'C')
    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFACE ---
algeria_tz = pytz.timezone('Africa/Algiers')
date_str = datetime.datetime.now(algeria_tz).strftime("%d/%m/%Y | %H:%M")

st.markdown('<p class="main-header">NEUROSCAN AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Cerebral Diagnostic System</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<p class="section-title">🔵 Identification</p>', unsafe_allow_html=True)
    nom = st.text_input("NOM DU PATIENT").upper()
    prenom = st.text_input("PRÉNOM DU PATIENT").capitalize()
    age = st.number_input("ÂGE", 0, 120, 30)
    gender = st.selectbox("GENRE", ["Masculin", "Féminin", "Autre"])

with col2:
    st.markdown('<p class="section-title">🟢 Acquisition IRM</p>', unsafe_allow_html=True)
    up = st.file_uploader("Scan", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if up:
        image = Image.open(up).convert('RGB')
        st.image(image, use_container_width=True)
    else:
        st.info("Système en attente de flux IRM...")

with col3:
    st.markdown('<p class="section-title">⚡ Analyse IA</p>', unsafe_allow_html=True)
    if up and st.button("Lancer le Scan"):
        with st.spinner("Analyse neuronale..."):
            model = load_neuro_model()
            img_array = np.array(image.resize((224, 224))) / 255.0
            preds = model.predict(np.expand_dims(img_array, axis=0))[0]
            classes = ['Non-Cérébral', 'Gliome', 'Méningiome', 'Pas de Tumeur', 'Pituitaire']
            res, conf = classes[np.argmax(preds)], float(np.max(preds)) * 100
            
            st.markdown(f'''
                <div class="res-card">
                    <div style="color:#00ff87; font-size:1.8em; font-weight:800;">{res}</div>
                    <div style="color:#ffffff; opacity:0.7; font-size:0.9em;">Indice de confiance : {conf:.2f}%</div>
                </div>
            ''', unsafe_allow_html=True)
            
            pdf_data = generate_medical_pdf(nom, prenom, age, gender, res, conf, image, date_str)
            st.download_button("📥 GÉNÉRER LE RAPPORT PDF", pdf_data, f"NeuroScan_{nom}.pdf", "application/pdf")
    else:
        st.write("Prêt pour l'acquisition.")

    # Lien LinkedIn stylisé
    st.markdown(f'''
        <div style="text-align:right; margin-top:100px;">
            <a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank" 
               style="color:#00d4ff; text-decoration:none; font-weight:700; border-bottom: 2px solid #00ff87;">
               CONSULTER L'EXPERT ↗
            </a>
        </div>
    ''', unsafe_allow_html=True)

st.markdown(f'<p class="footer">NeuroScan Engine v2.5 | Houbad Douaa & Bahlouli Fatna Romaisaa | {date_str}</p>', unsafe_allow_html=True)
