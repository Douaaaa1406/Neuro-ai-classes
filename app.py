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

# --- 2. DESIGN "NEON EMERALD" (BIO-TECH VIF) ---
st.markdown("""
<style>
    /* === FONTS === */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500;700&family=Outfit:wght@400;800&display=swap');

    /* === FOND : DEEP SLATE (PAS DE MAUVE) === */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, #002b2b 0%, #000d0d 100%);
        color: #e0f2f1;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* === SUPPRESSION DES STRUCTURES RIGIDES === */
    [data-testid="column"], [data-testid="stVerticalBlock"], .stColumn > div {
        background-color: transparent !important;
        border: none !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        max-width: 1350px;
    }

    /* === TITRE LUMINEUX (VERT VIF) === */
    .main-header {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #00ffa3 0%, #03dac5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4.5em;
        font-weight: 800;
        text-align: center;
        margin-bottom: -10px;
        filter: drop-shadow(0 0 15px rgba(0, 255, 163, 0.4));
    }
    
    .sub-text {
        color: #00ffa3;
        text-align: center;
        font-size: 1em;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 8px;
        opacity: 0.8;
    }

    /* === CARTES "ORGANIC" (SANS ANGLES DROITS) === */
    .art-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        border-radius: 50px; /* Courbure extrême pour éliminer les rectangles */
        padding: 45px;
        border: 1px solid rgba(0, 255, 163, 0.2);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
        min-height: 560px;
        transition: all 0.5s ease;
    }

    .art-card:hover {
        border: 1px solid #00ffa3;
        box-shadow: 0 0 30px rgba(0, 255, 163, 0.2);
        transform: scale(1.01);
    }

    .column-title {
        font-family: 'Outfit', sans-serif;
        color: #00ffa3;
        font-size: 1.8em;
        font-weight: 700;
        margin-bottom: 30px;
        letter-spacing: -0.5px;
    }

    /* === WIDGETS ARRONDIS & VIFS === */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(0, 255, 163, 0.05) !important;
        border: 1px solid rgba(0, 255, 163, 0.3) !important;
        border-radius: 25px !important;
        color: #ffffff !important;
        padding: 15px 25px !important;
    }

    /* === BOUTON D'ACTION "NEON" === */
    div.stButton > button {
        background: #00ffa3 !important;
        color: #001a1a !important;
        border-radius: 50px !important;
        font-weight: 800 !important;
        font-size: 1.2em !important;
        height: 65px;
        border: none !important;
        box-shadow: 0 0 20px rgba(0, 255, 163, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase;
    }

    div.stButton > button:hover {
        background: #ffffff !important;
        box-shadow: 0 0 40px rgba(0, 255, 163, 0.7);
        transform: translateY(-3px);
    }

    /* === ZONE RÉSULTAT === */
    .diagnostic-result {
        background: rgba(0, 255, 163, 0.1);
        border-radius: 35px;
        padding: 30px;
        border: 2px dashed #00ffa3;
        text-align: center;
        margin-top: 20px;
    }

    /* === FILE UPLOADER STYLE === */
    [data-testid="stFileUploader"] {
        border-radius: 35px;
        background: rgba(255, 255, 255, 0.02);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGIQUE TECHNIQUE (STABLE) ---
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
    pdf.set_text_color(0, 100, 80)
    pdf.cell(0, 25, "ANALYSIS REPORT - NEUROSCAN", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 10, f"Issued on: {date_str}", 0, 1, 'R')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 13)
    pdf.cell(0, 10, "PATIENT BIODATA", 0, 1)
    pdf.set_font("Arial", '', 11)
    for k, v in [["Full Name", f"{nom} {prenom}"], ["Age / Gender", f"{age} / {gender}"], ["System", "NeuroScan AI V2"]]:
        pdf.cell(50, 10, k, 1); pdf.cell(100, 10, v, 1); pdf.ln()
    
    pdf.ln(10)
    img.save("temp_scan.png")
    pdf.image("temp_scan.png", x=60, w=90)
    pdf.ln(10)
    
    pdf.set_fill_color(0, 255, 163)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 20, f"DIAGNOSIS: {resultat.upper()}", 1, 1, 'C', False)
    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFACE ---
algeria_tz = pytz.timezone('Africa/Algiers')
date_str = datetime.datetime.now(algeria_tz).strftime("%d/%m/%Y | %H:%M")

st.markdown('<p class="main-header">NEUROSCAN</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Advanced Neural Imaging</p>', unsafe_allow_html=True)
st.markdown(f'<div style="text-align:center; opacity:0.6; margin-bottom:40px;">{date_str}</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">01. Patient</div>', unsafe_allow_html=True)
    nom = st.text_input("NOM", value="DOE").upper()
    prenom = st.text_input("PRÉNOM", value="John").capitalize()
    age = st.number_input("ÂGE", 0, 120, 45)
    gender = st.selectbox("GENRE", ["Masculin", "Féminin", "Autre"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">02. MRI Data</div>', unsafe_allow_html=True)
    up = st.file_uploader("MRI", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if up:
        image = Image.open(up).convert('RGB')
        st.image(image, use_container_width=True)
    else:
        st.info("Upload MRI Scan to Begin")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">03. AI Insights</div>', unsafe_allow_html=True)
    
    if up and st.button("RUN DIAGNOSTIC"):
        with st.spinner("Processing..."):
            model = load_neuro_model()
            img_array = np.array(image.resize((224, 224))) / 255.0
            preds = model.predict(np.expand_dims(img_array, axis=0))[0]
            classes = ['Non-Cérébral', 'Gliome', 'Méningiome', 'Pas de Tumeur', 'Pituitaire']
            res, conf = classes[np.argmax(preds)], float(np.max(preds)) * 100
            
            st.markdown(f'''
                <div class="diagnostic-result">
                    <h2 style="color:#00ffa3; margin:0; font-size:2em;">{res}</h2>
                    <p style="color:white; margin:0; opacity:0.7;">Confidence: {conf:.2f}%</p>
                </div>
            ''', unsafe_allow_html=True)
            
            pdf_data = generate_medical_pdf(nom, prenom, age, gender, res, conf, image, date_str)
            st.download_button("📥 DOWNLOAD REPORT", pdf_data, f"NeuroScan_{nom}.pdf", "application/pdf")
    else:
        st.info("Awaiting Input Data")
    
    st.markdown(f'''
        <div style="text-align:right; margin-top:100px;">
            <a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank" 
               style="color:#00ffa3; text-decoration:none; font-weight:700; font-size:1.1em;">
               VIEW SCIENTIFIC PROFILE ↗
            </a>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<p style="text-align:center; margin-top:50px; opacity:0.4;">NeuroScan AI • Biomedical Innovation • 2026</p>', unsafe_allow_html=True)
