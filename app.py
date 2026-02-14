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

# --- 2. DESIGN "GRIS CLAIR GALERIE & GRENAT" ---
st.markdown("""
<style>
    /* Fond Gris Éclairci (Plus lumineux) */
    [data-testid="stAppViewContainer"] {
        background-color: #f0f2f6;
        color: #1f1f1f;
    }

    /* En-tête élégante */
    .main-header {
        font-family: 'Times New Roman', serif;
        color: #1a1a1a;
        font-size: 3.5em;
        font-weight: bold;
        text-align: center;
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

    .live-date {
        text-align: center;
        color: #555;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        margin-bottom: 20px;
    }

    /* Cartes Gris Doux (Élimine les blocs blancs agressifs) */
    .art-card {
        background: #e8eaf0;
        border-radius: 8px;
        padding: 25px;
        border-top: 4px solid #800020;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        min-height: 550px;
        margin-bottom: 20px;
    }

    .column-title {
        font-family: 'Times New Roman', serif;
        color: #1a1a1a;
        border-bottom: 2px solid #800020;
        padding-bottom: 8px;
        font-size: 1.3em;
        font-weight: bold;
        margin-bottom: 25px;
    }

    /* Bouton d'Analyse Grenat */
    div.stButton > button {
        background: #800020 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        font-family: 'Times New Roman', serif;
        font-size: 1.2em !important;
        width: 100%;
        height: 50px;
        transition: 0.3s !important;
    }
    div.stButton > button:hover {
        background: #4d0013 !important;
        box-shadow: 0 4px 12px rgba(128, 0, 32, 0.3) !important;
    }

    .linkedin-link {
        color: #800020 !important;
        text-decoration: none;
        font-weight: bold;
        border-bottom: 1px solid #800020;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGIQUE TECHNIQUE ---
@st.cache_resource
def load_neuro_model():
    model_path = 'brain_tumor_model_v6_final.keras'
    if not os.path.exists(model_path):
        gdown.download(f'https://drive.google.com/uc?id=1QRVvhNHSx7qgw0GIDrRLsuX09uItsXM2', model_path, quiet=False)
    
    # Reconstruction architecture
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
    pdf.ln(5)
    pdf.set_font("Times", 'B', 11)
    pdf.cell(0, 10, f"Généré le : {date_str}", 0, 1, 'R')
    
    pdf.ln(5)
    pdf.set_font("Times", 'B', 13)
    pdf.cell(0, 10, "1. INFORMATIONS PATIENT", 0, 1)
    
    pdf.set_font("Times", '', 11)
    # Tableau
    data = [
        ["Nom Complet", f"{nom} {prenom}"],
        ["Âge / Genre", f"{age} / {gender}"],
        ["Développeur", "Douaa Houbad (M1 EMB)"],
        ["Date & Heure d'analyse", date_str]
    ]
    for row in data:
        pdf.cell(60, 10, row[0], 1)
        pdf.cell(100, 10, row[1], 1)
        pdf.ln()

    pdf.ln(10)
    pdf.set_font("Times", 'B', 13)
    pdf.cell(0, 10, "2. SCAN IRM ANALYSÉ", 0, 1)
    img.save("temp_pdf_scan.png")
    pdf.image("temp_pdf_scan.png", x=60, w=90)
    
    pdf.ln(10)
    pdf.set_font("Times", 'B', 15)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 15, f"RÉSULTAT DU DIAGNOSTIC : {resultat.upper()}", 1, 1, 'C', True)
    pdf.set_font("Times", 'I', 11)
    pdf.cell(0, 10, f"Indice de confiance : {confiance:.2f}%", 0, 1, 'C')
    
    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFACE ---
# Gestion Date et Heure (Algérie)
algeria_tz = pytz.timezone('Africa/Algiers')
now = datetime.datetime.now(algeria_tz)
date_str_platform = now.strftime("%A %d %B %Y | %H:%M:%S")

st.markdown('<p class="main-header">NeuroScan AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Ingénierie Biomédicale • Diagnostic Avancé</p>', unsafe_allow_html=True)
st.markdown(f'<p class="live-date">📅 {date_str_platform}</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">Données Patient</div>', unsafe_allow_html=True)
    nom = st.text_input("NOM DE FAMILLE").upper()
    prenom = st.text_input("PRÉNOM").capitalize()
    age = st.number_input("ÂGE", min_value=0, value=30)
    gender = st.selectbox("GENRE", ["Masculin", "Féminin"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">Acquisition Scan</div>', unsafe_allow_html=True)
    up = st.file_uploader("Importer l'IRM", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if up:
        image = Image.open(up).convert('RGB')
        st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">Analyse Neurale</div>', unsafe_allow_html=True)
    if up and st.button("LANCER L'ANALYSE"):
        model = load_neuro_model()
        img_array = np.array(image.resize((224, 224))) / 255.0
        preds = model.predict(np.expand_dims(img_array, axis=0))[0]
        
        classes = ['Non-Cérébral', 'Gliome', 'Méningiome', 'Pas de Tumeur', 'Pituitaire']
        res = classes[np.argmax(preds)]
        conf = float(np.max(preds)) * 100

        st.markdown(f"**Diagnostic :** {res}")
        st.markdown(f"**Confiance :** {conf:.2f}%")

        # Génération PDF avec Date/Heure
        pdf_bytes = generate_medical_pdf(nom, prenom, age, gender, res, conf, image, date_str_platform)
        st.download_button("📥 TÉLÉCHARGER LE RAPPORT PDF", pdf_bytes, f"Diagnostic_{nom}.pdf", "application/pdf")
    else:
        st.info("En attente de l'imagerie...")

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:right;"><a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank" class="linkedin-link">Profil LinkedIn</a></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<p style="text-align:center; color:#888; margin-top:50px; font-family:serif;">Conçu par Douaa Houbad | Ingénieur Biomédical M1 EMB | 2026</p>', unsafe_allow_html=True)
