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

# --- 2. DESIGN ÉPURÉ (GRIS CLAIR & GRENAT) ---
st.markdown("""
<style>
    /* Fond Gris Perle Uni et Propre */
    [data-testid="stAppViewContainer"] {
        background-color: #f0f2f6;
        color: #1a1a1a;
    }

    /* Titre en Times New Roman */
    .main-header {
        font-family: 'Times New Roman', Times, serif;
        color: #1a1a1a;
        font-size: 3.5em;
        text-align: center;
        margin-bottom: 0px;
        font-weight: bold;
    }
    
    .sub-text {
        font-family: 'Times New Roman', Times, serif;
        color: #800020;
        text-align: center;
        font-size: 1.2em;
        letter-spacing: 2px;
        margin-top: -10px;
        margin-bottom: 40px;
    }

    /* Colonnes sans bannières, juste une séparation légère */
    .clean-section {
        background: white;
        padding: 30px;
        border-radius: 10px;
        border-top: 3px solid #800020;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        min-height: 500px;
    }

    .section-label {
        font-family: 'Times New Roman', Times, serif;
        color: #800020;
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 20px;
        border-bottom: 1px solid #eee;
    }

    /* Bouton d'Analyse Grenat Profond */
    div.stButton > button {
        background-color: #800020 !important;
        color: white !important;
        border-radius: 4px !important;
        border: none !important;
        height: 50px;
        width: 100%;
        font-family: 'Times New Roman', serif;
        font-size: 1.2em !important;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #5a0016 !important;
    }

    /* Texte LinkedIn */
    .linkedin-footer {
        text-align: right;
        margin-top: 20px;
    }
    .linkedin-footer a {
        color: #800020 !important;
        text-decoration: none;
        font-family: 'Times New Roman', serif;
        font-weight: bold;
        border: 1px solid #800020;
        padding: 5px 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. FONCTIONS TECHNIQUES ---
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

def create_medical_pdf(nom, prenom, age, gender, resultat, confiance, img, date_str):
    pdf = FPDF()
    pdf.add_page()
    # Utilisation de Times New Roman dans le PDF 
    pdf.set_font("Times", 'B', 20)
    pdf.set_text_color(128, 0, 32)
    pdf.cell(0, 20, "RAPPORT MEDICAL NEUROSCAN AI", 0, 1, 'C')
    
    pdf.ln(10)
    pdf.set_font("Times", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. INFORMATIONS PATIENT", 0, 1)
    
    pdf.set_font("Times", '', 11)
    # Données du patient 
    pdf.cell(50, 8, "Nom complet", 1)
    pdf.cell(100, 8, f"{nom} {prenom}", 1, 1)
    pdf.cell(50, 8, "Age / Genre", 1)
    pdf.cell(100, 8, f"{age} ans / {gender}", 1, 1)
    pdf.cell(50, 8, "Modele utilise", 1)
    pdf.cell(100, 8, "MobileNetV2-NeuroV6", 1, 1)
    pdf.cell(50, 8, "Date de l'examen", 1)
    pdf.cell(100, 8, date_str, 1, 1)

    pdf.ln(10)
    img.save("temp_scan.png")
    pdf.image("temp_scan.png", x=60, w=90)
    
    pdf.ln(10)
    pdf.set_font("Times", 'B', 16)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 15, f"RESULTAT : {resultat.upper()}", 1, 1, 'C', True)
    pdf.set_font("Times", 'I', 11)
    pdf.cell(0, 10, f"Indice de confiance : {confiance:.2f}%", 0, 1, 'C')

    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFACE PRINCIPALE ---
algeria_tz = pytz.timezone('Africa/Algiers')
now = datetime.datetime.now(algeria_tz)
date_str = now.strftime("%d/%m/%Y - %H:%M")

st.markdown('<p class="main-header">NeuroScan AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">ENGINEERING & DIAGNOSTICS</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown('<div class="clean-section">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Patient Data</p>', unsafe_allow_html=True)
    nom = st.text_input("NOM").upper()
    prenom = st.text_input("PRÉNOM").capitalize()
    age = st.number_input("ÂGE", min_value=0, value=30)
    gender = st.selectbox("GENRE", ["Male", "Female"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="clean-section">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">MRI Scan</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="clean-section">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Analysis Result</p>', unsafe_allow_html=True)
    if uploaded_file and st.button("RUN ANALYSIS"):
        model = load_neuro_model()
        img_array = np.array(image.resize((224, 224))) / 255.0
        preds = model.predict(np.expand_dims(img_array, axis=0))[0]
        
        classes = ['Non-Cérébral', 'Gliome', 'Méningiome', 'Pas de Tumeur', 'Pituitaire']
        idx = np.argmax(preds)
        res_text = classes[idx]
        conf_val = float(preds[idx]) * 100

        st.write(f"**Diagnostic :** {res_text}")
        st.write(f"**Confiance :** {conf_val:.2f}%")
        
        # PDF généré en Times Roman 
        pdf_bytes = create_medical_pdf(nom, prenom, age, gender, res_text, conf_val, image, date_str)
        st.download_button("📥 TELECHARGER LE RAPPORT PDF", pdf_bytes, f"NeuroScan_{nom}.pdf", "application/pdf")
    else:
        st.write("En attente d'analyse...")
    
    st.markdown(f'<div class="linkedin-footer"><a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank">LinkedIn</a></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<p style="text-align:center; color:#666; font-family:Times; padding:30px;">Designed by Douaa Houbad | Biomedical Engineer | {date_str}</p>', unsafe_allow_html=True)
