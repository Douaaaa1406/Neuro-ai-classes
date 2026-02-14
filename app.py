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
st.set_page_config(page_title="NeuroScan AI | Fine Art Edition", page_icon="🧠", layout="wide")

# --- 2. DESIGN "ARTISTIC ANTHRACITE & GRENAT" ---
st.markdown("""
<style>
    /* Fond Gris Galerie avec texture légère */
    [data-testid="stAppViewContainer"] {
        background-color: #2b2b2b;
        background-image: url("https://www.transparenttextures.com/patterns/dark-matter.png");
        color: #e0e0e0;
    }

    /* En-tête avec typographie élégante */
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #ffffff;
        font-size: 3.5em;
        font-weight: 200;
        text-align: center;
        letter-spacing: 4px;
        margin-top: -50px;
    }
    
    .sub-text {
        color: #800020; /* Grenat */
        text-align: center;
        font-size: 1.1em;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: -15px;
    }

    .system-status {
        border-top: 1px solid #444;
        border-bottom: 1px solid #444;
        padding: 10px;
        color: #888;
        text-align: center;
        width: 100%;
        margin: 20px 0;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.8em;
    }

    /* Cartes Gris Sombre avec bordure Grenat */
    .art-card {
        background: rgba(45, 45, 45, 0.95);
        border-radius: 5px;
        padding: 30px;
        border-left: 4px solid #800020;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        min-height: 600px;
        margin-bottom: 20px;
    }

    .column-title {
        color: #ffffff;
        border-bottom: 1px solid #800020;
        padding-bottom: 10px;
        font-size: 0.9em;
        font-weight: bold;
        margin-bottom: 25px;
        letter-spacing: 1px;
    }

    /* Cadre Image style "Exposition" */
    .scan-frame {
        border: 1px solid #555;
        padding: 10px;
        background: #1a1a1a;
        box-shadow: inset 0 0 15px rgba(0,0,0,0.5);
    }

    /* Bouton d'Analyse Grenat Profond */
    div.stButton > button {
        background: #800020 !important;
        color: white !important;
        border: none !important;
        padding: 15px !important;
        border-radius: 0px !important;
        font-weight: 300 !important;
        font-size: 1.2em !important;
        letter-spacing: 3px;
        width: 100%;
        transition: 0.4s !important;
    }
    div.stButton > button:hover {
        background: #a00028 !important;
        box-shadow: 0 0 20px rgba(128, 0, 32, 0.6) !important;
    }

    /* Inputs personnalisés */
    input, select, .stSelectbox {
        background-color: #333 !important;
        color: white !important;
        border: 1px solid #444 !important;
    }

    /* Progress bar Grenat */
    .p-bar-container {
        width: 100%; background: #1a1a1a; height: 10px; margin: 15px 0;
    }
    .p-bar-fill {
        height: 100%; background: #800020;
    }

    /* LinkedIn discret */
    .linkedin-link {
        color: #800020 !important;
        text-decoration: none;
        font-weight: bold;
        font-size: 0.9em;
        border: 1px solid #800020;
        padding: 5px 15px;
        transition: 0.3s;
    }
    .linkedin-link:hover {
        background: #800020;
        color: white !important;
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
    
    # Header élégant
    pdf.set_fill_color(128, 0, 32) # Grenat
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 15, "NEUROSCAN AI - CLINICAL DIAGNOSTIC", 0, 1, 'C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 5, "Advanced Deep Learning Medical Analysis", 0, 1, 'C')
    
    # Corps du rapport
    pdf.set_text_color(0, 0, 0)
    pdf.ln(15)
    
    # Tableau des informations Patient
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. PATIENT INFORMATION", 0, 1)
    pdf.set_font("Arial", '', 11)
    
    data = [
        ["Patient Name", f"{nom} {prenom}"],
        ["Age / Gender", f"{age} / {gender}"],
        ["Analysis Date", date_str],
        ["System Developer", "Douaa Houbad (M1 EMB)"],
        ["AI Model Architecture", "MobileNetV2-NeuroV6"]
    ]
    
    for row in data:
        pdf.cell(60, 8, row[0], 1)
        pdf.cell(100, 8, row[1], 1)
        pdf.ln()

    # Image Scan
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. ANALYZED MRI SCAN", 0, 1)
    img.save("temp_pdf_scan.png")
    pdf.image("temp_pdf_scan.png", x=60, w=90)
    
    # Résultat Final
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 12, f"CONCLUSION: {resultat.upper()}", 1, 1, 'C', True)
    pdf.set_font("Arial", 'I', 11)
    pdf.cell(0, 10, f"Confidence Level: {confiance:.2f}%", 0, 1, 'C')
    
    # Footer PDF
    pdf.set_y(-30)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5, "Disclaimer: This AI-generated report is for research purposes. It must be validated by a neuro-radiologist.", 0, 'C')

    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFACE ---
algeria_tz = pytz.timezone('Africa/Algiers')
now = datetime.datetime.now(algeria_tz)
date_str = now.strftime("%d/%m/%Y - %H:%M:%S")

st.markdown('<p class="main-header">NeuroScan Core AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Diagnostic Imagerie Médicale • M1 EMB</p>', unsafe_allow_html=True)
st.markdown(f'<div class="system-status">SECURE ACCESS | {date_str} | SYSTEM_V6_STABLE</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">DONNÉES PATIENT</div>', unsafe_allow_html=True)
    nom = st.text_input("NOM DE FAMILLE").upper()
    prenom = st.text_input("PRÉNOM").capitalize()
    age = st.number_input("ÂGE", min_value=0, value=30)
    gender = st.selectbox("GENRE", ["Male", "Female"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">SCAN ACQUISITION</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Importer IRM", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.markdown('<div class="scan-frame">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">RÉSULTATS NEURAUX</div>', unsafe_allow_html=True)
    if uploaded_file and st.button("LANCER L'ANALYSE"):
        model = load_neuro_model()
        img_prep = image.resize((224, 224))
        img_array = np.array(img_prep) / 255.0
        preds = model.predict(np.expand_dims(img_array, axis=0))[0]
        
        classes = ['Non-Cérébral', 'Gliome', 'Méningiome', 'Pas de Tumeur', 'Pituitaire']
        idx = np.argmax(preds)
        conf_val = float(preds[idx]) * 100
        res_text = classes[idx]

        st.markdown(f"**Diagnostic :** {res_text}")
        st.markdown(f"**Certitude :** {conf_val:.1f}%")
        st.markdown(f'<div class="p-bar-container"><div class="p-bar-fill" style="width:{conf_val}%;"></div></div>', unsafe_allow_html=True)

        # Bouton PDF avec toutes les infos
        pdf_data = generate_medical_pdf(nom, prenom, age, gender, res_text, conf_val, image, date_str)
        st.download_button("📥 GÉNÉRER LE RAPPORT CLINIQUE", pdf_data, f"NeuroScan_{nom}.pdf", "application/pdf")
    else:
        st.info("Système en attente d'imagerie...")

    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:right;"><a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank" class="linkedin-link">LinkedIn Profile</a></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
    <div style="text-align:center; padding:40px; color:#666; font-size:0.8em; letter-spacing:1px;">
        DESIGNED BY <b>DOUAA HOUBAD</b> | M1 EMB BIOMEDICAL ENGINEER<br>
        ALGIERS, ALGERIA • 2026
    </div>
""", unsafe_allow_html=True)
