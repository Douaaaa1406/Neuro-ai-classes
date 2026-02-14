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
        font-family: 'Times New Roman', serif;
        color: #ffffff;
        font-size: 3.5em;
        font-weight: bold;
        text-align: center;
        letter-spacing: 2px;
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

    /* Cartes Gris Sombre (Élimine les blocs blancs) */
    .art-card {
        background: rgba(45, 45, 45, 0.8);
        border-radius: 4px;
        padding: 25px;
        border-top: 3px solid #800020;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        min-height: 550px;
        margin-bottom: 20px;
    }

    .column-title {
        font-family: 'Times New Roman', serif;
        color: #ffffff;
        border-bottom: 1px solid #444;
        padding-bottom: 10px;
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 25px;
    }

    /* Bouton d'Analyse Grenat Profond */
    div.stButton > button {
        background: #800020 !important;
        color: white !important;
        border: none !important;
        border-radius: 2px !important;
        font-family: 'Times New Roman', serif;
        font-size: 1.2em !important;
        width: 100%;
        height: 50px;
        transition: 0.4s !important;
    }
    div.stButton > button:hover {
        background: #a00028 !important;
        transform: scale(1.02);
    }

    /* LinkedIn discret */
    .linkedin-link {
        color: #800020 !important;
        text-decoration: none;
        font-weight: bold;
        border-bottom: 1px solid #800020;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGIQUE TECHNIQUE (CORRIGÉE POUR LES TENSEURS) ---
@st.cache_resource
def load_neuro_model():
    model_path = 'brain_tumor_model_v6_final.keras'
    if not os.path.exists(model_path):
        gdown.download(f'https://drive.google.com/uc?id=1QRVvhNHSx7qgw0GIDrRLsuX09uItsXM2', model_path, quiet=False)
    
    # Reconstruction exacte incluant BatchNormalization pour éviter l'erreur de chargement
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
    
    # Header élégant en Times
    pdf.set_font("Times", 'B', 20)
    pdf.set_text_color(128, 0, 32)
    pdf.cell(0, 20, "NEUROSCAN AI - CLINICAL DIAGNOSTIC", 0, 1, 'C')
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Times", 'B', 12)
    pdf.cell(0, 10, "1. PATIENT INFORMATION", 0, 1)
    
    pdf.set_font("Times", '', 11)
    data = [
        ["Name", f"{nom} {prenom}"],
        ["Age / Gender", f"{age} / {gender}"],
        ["Developer", "Douaa Houbad (M1 EMB)"],
        ["Date", date_str]
    ]
    for row in data:
        pdf.cell(50, 8, row[0], 1)
        pdf.cell(100, 8, row[1], 1)
        pdf.ln()

    pdf.ln(10)
    img.save("temp_scan.png")
    pdf.image("temp_scan.png", x=60, w=90)
    
    pdf.ln(10)
    pdf.set_font("Times", 'B', 14)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 15, f"RESULT: {resultat.upper()}", 1, 1, 'C', True)
    
    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFACE ---
algeria_tz = pytz.timezone('Africa/Algiers')
date_str = datetime.datetime.now(algeria_tz).strftime("%d/%m/%Y - %H:%M")

st.markdown('<p class="main-header">NeuroScan AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Biomedical Engineering • Diagnostic V6</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">Patient Data</div>', unsafe_allow_html=True)
    nom = st.text_input("LAST NAME").upper()
    prenom = st.text_input("FIRST NAME").capitalize()
    age = st.number_input("AGE", min_value=0, value=30)
    gender = st.selectbox("GENDER", ["Male", "Female"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">MRI Scan</div>', unsafe_allow_html=True)
    up = st.file_uploader("Upload", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if up:
        image = Image.open(up).convert('RGB')
        st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">Neural Results</div>', unsafe_allow_html=True)
    if up and st.button("ANALYZE"):
        model = load_neuro_model()
        img_array = np.array(image.resize((224, 224))) / 255.0
        preds = model.predict(np.expand_dims(img_array, axis=0))[0]
        
        classes = ['Non-Brain', 'Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        res = classes[np.argmax(preds)]
        conf = float(np.max(preds)) * 100

        st.markdown(f"**Diagnostic:** {res}")
        st.markdown(f"**Confidence:** {conf:.2f}%")

        pdf_bytes = generate_medical_pdf(nom, prenom, age, gender, res, conf, image, date_str)
        st.download_button("📥 DOWNLOAD REPORT", pdf_bytes, f"Report_{nom}.pdf", "application/pdf")

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown(f'<a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank" class="linkedin-link">LinkedIn Profile</a>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<p style="text-align:center; color:#666; margin-top:50px;">Designed by Douaa Houbad | M1 EMB | 2026</p>', unsafe_allow_html=True)
