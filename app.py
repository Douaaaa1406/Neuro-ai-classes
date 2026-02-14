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

# --- 2. DESIGN "GRIS ACIER & GRENAT" (PLUS HAUT & PLUS CONTRASTÉ) ---
st.markdown("""
<style>
    /* Fond Gris Acier (plus soutenu que le précédent) */
    [data-testid="stAppViewContainer"] {
        background-color: #d1d5db;
        color: #111827;
    }

    /* Remonter tout le contenu vers le haut */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
    }

    /* En-tête compacte */
    .main-header {
        font-family: 'Times New Roman', serif;
        color: #111827;
        font-size: 3.2em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0px;
    }
    
    .sub-text {
        color: #800020; /* Grenat */
        text-align: center;
        font-size: 1.1em;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: -5px;
        margin-bottom: 10px;
    }

    .live-date {
        text-align: center;
        color: #374151;
        font-family: 'Courier New', monospace;
        font-size: 0.85em;
        margin-bottom: 25px;
    }

    /* Cartes Gris Moyen pour un contraste parfait */
    .art-card {
        background: #e5e7eb;
        border-radius: 8px;
        padding: 20px;
        border-top: 5px solid #800020;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        min-height: 520px;
        margin-bottom: 15px;
    }

    .column-title {
        font-family: 'Times New Roman', serif;
        color: #111827;
        border-bottom: 2px solid #800020;
        padding-bottom: 5px;
        font-size: 1.25em;
        font-weight: bold;
        margin-bottom: 20px;
    }

    /* Bouton d'Analyse Grenat */
    div.stButton > button {
        background: #800020 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        font-family: 'Times New Roman', serif;
        font-size: 1.1em !important;
        width: 100%;
        height: 45px;
        transition: 0.2s;
    }
    div.stButton > button:hover {
        background: #4d0013 !important;
    }

    /* Style des textes d'input */
    label {
        color: #111827 !important;
        font-weight: bold !important;
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
    
    # Header PDF
    pdf.set_font("Times", 'B', 20)
    pdf.set_text_color(128, 0, 32)
    pdf.cell(0, 20, "RAPPORT CLINIQUE - NEUROSCAN AI", 0, 1, 'C')
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Times", 'B', 10)
    pdf.cell(0, 10, f"Date de l'examen : {date_str}", 0, 1, 'R')
    
    pdf.ln(5)
    pdf.set_font("Times", 'B', 13)
    pdf.cell(0, 10, "1. INFORMATIONS PATIENT", 0, 1)
    
    pdf.set_font("Times", '', 11)
    data = [
        ["Nom / Prenom", f"{nom} {prenom}"],
        ["Age / Genre", f"{age} / {gender}"],
        ["Analyste", "Douaa Houbad (M1 EMB)"],
        ["Horodatage", date_str]
    ]
    for row in data:
        pdf.cell(60, 10, row[0], 1)
        pdf.cell(100, 10, row[1], 1)
        pdf.ln()

    pdf.ln(10)
    img.save("temp_pdf.png")
    pdf.image("temp_pdf.png", x=65, w=80)
    
    pdf.ln(10)
    pdf.set_font("Times", 'B', 15)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 15, f"RÉSULTAT : {resultat.upper()}", 1, 1, 'C', True)
    pdf.set_font("Times", 'I', 11)
    pdf.cell(0, 10, f"Indice de confiance : {confiance:.2f}%", 0, 1, 'C')
    
    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFACE ---
algeria_tz = pytz.timezone('Africa/Algiers')
now = datetime.datetime.now(algeria_tz)
date_str = now.strftime("%d/%m/%Y | %H:%M:%S")

st.markdown('<p class="main-header">NeuroScan AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Biomedical Engineering • Algiers</p>', unsafe_allow_html=True)
st.markdown(f'<p class="live-date">🕒 {date_str}</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">Données Patient</div>', unsafe_allow_html=True)
    nom = st.text_input("NOM").upper()
    prenom = st.text_input("PRÉNOM").capitalize()
    age = st.number_input("ÂGE", min_value=0, value=30)
    gender = st.selectbox("GENRE", ["Masculin", "Féminin"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">IRM Scan</div>', unsafe_allow_html=True)
    up = st.file_uploader("Scan", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if up:
        image = Image.open(up).convert('RGB')
        st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">Diagnostic</div>', unsafe_allow_html=True)
    if up and st.button("LANCER L'ANALYSE"):
        model = load_neuro_model()
        img_array = np.array(image.resize((224, 224))) / 255.0
        preds = model.predict(np.expand_dims(img_array, axis=0))[0]
        
        classes = ['Non-Cérébral', 'Gliome', 'Méningiome', 'Pas de Tumeur', 'Pituitaire']
        res = classes[np.argmax(preds)]
        conf = float(np.max(preds)) * 100

        st.markdown(f"**Diagnostic :** {res}")
        st.markdown(f"**Confiance :** {conf:.2f}%")

        pdf_bytes = generate_medical_pdf(nom, prenom, age, gender, res, conf, image, date_str)
        st.download_button("📥 TÉLÉCHARGER LE RAPPORT", pdf_bytes, f"NeuroScan_{nom}.pdf", "application/pdf")
    else:
        st.write("En attente d'acquisition...")

    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:right; border-top:1px solid #9ca3af; padding-top:100px;"><a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank" style="color:#800020; font-weight:bold; text-decoration:none;">LinkedIn</a></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<p style="text-align:center; color:#374151; font-family:serif; margin-top:30px;">Douaa Houbad | M1 EMB | 2026</p>', unsafe_allow_html=True)
