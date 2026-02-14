import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from fpdf import FPDF
import datetime
import os
import gdown
import pytz

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="wide")

# --- 2. STYLE ÉPURÉ & CONTRASTE (SANS RECTANGLES BLANCS) ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #e5e7eb; /* Gris clair pro */
        color: #111827;
    }

    .header-area {
        text-align: center;
        margin-top: -60px;
        margin-bottom: 30px;
    }

    .title-text {
        font-family: 'Times New Roman', serif;
        font-size: 3.5em;
        font-weight: bold;
        color: #111827;
    }

    .subtitle-text {
        font-family: 'Times New Roman', serif;
        font-size: 1.2em;
        color: #800020; /* Grenat */
        letter-spacing: 5px;
        text-transform: uppercase;
    }

    /* Suppression des bordures et fonds blancs des colonnes */
    [data-testid="column"] {
        background-color: transparent !important;
    }

    .section-header {
        font-family: 'Times New Roman', serif;
        color: #800020;
        font-size: 1.4em;
        font-weight: bold;
        border-bottom: 2px solid #800020;
        margin-bottom: 20px;
        padding-bottom: 5px;
    }

    div.stButton > button {
        background-color: #800020 !important;
        color: white !important;
        border: none !important;
        font-family: 'Times New Roman', serif;
        font-size: 1.2em !important;
        width: 100%;
        height: 50px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. FONCTIONS TECHNIQUES (CORRECTION DU LOAD_WEIGHTS) ---
@st.cache_resource
def load_neuro_model():
    model_path = 'brain_tumor_model_v6_final.keras'
    if not os.path.exists(model_path):
        gdown.download(f'https://drive.google.com/uc?id=1QRVvhNHSx7qgw0GIDrRLsuX09uItsXM2', model_path, quiet=False)
    
    try:
        # TENTATIVE 1 : Charger comme un modèle complet (Recommandé pour .keras)
        model = tf.keras.models.load_model(model_path)
    except Exception:
        # TENTATIVE 2 : Si c'est uniquement des poids, reconstruire l'architecture
        base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        model.load_weights(model_path)
    return model

def create_medical_pdf(nom, prenom, age, gender, result, confidence, img, date_str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Times", 'B', 20)
    pdf.set_text_color(128, 0, 32)
    pdf.cell(0, 20, "DIAGNOSTIC REPORT: NEUROSCAN AI", 0, 1, 'C')
    
    pdf.ln(10)
    pdf.set_font("Times", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. PATIENT INFORMATION", 0, 1)
    
    pdf.set_font("Times", '', 11)
    # Tableau Patient 
    data = [
        ["Name", f"{nom} {prenom}"],
        ["Age / Gender", f"{age} / {gender}"],
        ["Prediction Model", "MobileNetV2-NeuroV6"],
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
    pdf.cell(0, 15, f"RESULT: {result.upper()}", 1, 1, 'C', True)
    pdf.set_font("Times", 'I', 11)
    pdf.cell(0, 10, f"Algorithm Confidence: {confidence:.2f}%", 0, 1, 'C')
    
    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFACE ---
algeria_tz = pytz.timezone('Africa/Algiers')
date_display = datetime.datetime.now(algeria_tz).strftime("%d/%m/%Y - %H:%M")

st.markdown(f'<div class="header-area"><p class="title-text">NeuroScan AI</p><p class="subtitle-text">Engineering & Diagnostics</p></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<p class="section-header">Patient Information</p>', unsafe_allow_html=True)
    nom = st.text_input("LAST NAME").upper()
    prenom = st.text_input("FIRST NAME").capitalize()
    age = st.number_input("AGE", min_value=0, value=30)
    gender = st.selectbox("GENDER", ["Male", "Female"])

with col2:
    st.markdown('<p class="section-header">MRI Acquisition</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Scan", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)

with col3:
    st.markdown('<p class="section-header">Neural Analysis</p>', unsafe_allow_html=True)
    if uploaded_file and st.button("START ANALYSIS"):
        model = load_neuro_model()
        img_prep = np.array(image.resize((224, 224))) / 255.0
        preds = model.predict(np.expand_dims(img_prep, axis=0))[0]
        
        classes = ['Non-Brain', 'Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        idx = np.argmax(preds)
        res_text = classes[idx]
        conf = float(preds[idx]) * 100

        st.write(f"**Diagnostic :** {res_text}")
        st.write(f"**Confiance :** {conf:.2f}%")
        
        pdf_bytes = create_medical_pdf(nom, prenom, age, gender, res_text, conf, image, date_display)
        st.download_button("📥 DOWNLOAD REPORT", pdf_bytes, f"Report_{nom}.pdf", "application/pdf")

st.markdown(f'<p style="text-align:center; color:#4b5563; font-family:Times; margin-top:100px;">Designed by Douaa Houbad | M1 EMB | {date_display}</p>', unsafe_allow_html=True)
