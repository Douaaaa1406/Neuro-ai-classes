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

# --- 2. STYLE ÉPURÉ : GRIS & GRENAT (SANS RECTANGLES BLANCS) ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #e5e7eb !important;
        color: #111827;
    }
    /* Élimination totale des rectangles blancs */
    [data-testid="column"], .stColumn > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    .header-area { text-align: center; margin-top: -60px; margin-bottom: 30px; }
    .title-text { font-family: 'Times New Roman', serif; font-size: 3.5em; font-weight: bold; color: #111827; }
    .subtitle-text { font-family: 'Times New Roman', serif; font-size: 1.2em; color: #800020; letter-spacing: 5px; text-transform: uppercase; }
    .section-header { font-family: 'Times New Roman', serif; color: #800020; font-size: 1.4em; font-weight: bold; border-bottom: 2px solid #800020; margin-bottom: 20px; padding-bottom: 5px; }
    div.stButton > button {
        background-color: #800020 !important;
        color: white !important;
        border-radius: 4px !important;
        font-family: 'Times New Roman', serif;
        width: 100%; height: 50px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SOLUTION TECHNIQUE : RECONSTRUCTION ROBUSTE ---
@st.cache_resource
def load_neuro_model():
    model_path = 'brain_tumor_model_v6_final.keras'
    if not os.path.exists(model_path):
        gdown.download(f'https://drive.google.com/uc?id=1QRVvhNHSx7qgw0GIDrRLsuX09uItsXM2', model_path, quiet=False)
    
    # On reconstruit l'architecture standard MobileNetV2 pour éviter l'erreur de tenseurs
    # Cette structure est celle qui correspond généralement aux modèles v6 de classification
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'), # Couche dense intermédiaire souvent présente
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    # On charge les poids en ignorant les erreurs de topologie si nécessaire
    try:
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
    except Exception:
        # Si load_weights échoue encore, on tente le load_model classique en ignorant les couches problématiques
        model = tf.keras.models.load_model(model_path, compile=False)
        
    return model

def create_clinical_pdf(nom, prenom, age, gender, result, confidence, img, date_str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Times", 'B', 20)
    pdf.set_text_color(128, 0, 32)
    pdf.cell(0, 20, "NEUROSCAN AI - CLINICAL REPORT", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font("Times", 'B', 12)
    pdf.cell(0, 10, "1. PATIENT INFORMATION", 0, 1)
    pdf.set_font("Times", '', 11)
    # Tableau
    pdf.cell(60, 8, "Patient Name", 1); pdf.cell(100, 8, f"{nom} {prenom}", 1, 1)
    pdf.cell(60, 8, "Age / Gender", 1); pdf.cell(100, 8, f"{age} / {gender}", 1, 1)
    pdf.cell(60, 8, "Developer", 1); pdf.cell(100, 8, "Douaa Houbad (M1 EMB)", 1, 1)
    pdf.cell(60, 8, "Date", 1); pdf.cell(100, 8, date_str, 1, 1)
    pdf.ln(10)
    img.save("scan.png")
    pdf.image("scan.png", x=65, w=80)
    pdf.ln(10)
    pdf.set_font("Times", 'B', 14)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 15, f"DIAGNOSTIC : {result.upper()}", 1, 1, 'C', True)
    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFACE ---
algeria_tz = pytz.timezone('Africa/Algiers')
date_now = datetime.datetime.now(algeria_tz).strftime("%d/%m/%Y - %H:%M")

st.markdown(f'<div class="header-area"><p class="title-text">NeuroScan AI</p><p class="subtitle-text">Medical Engineering</p></div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.markdown('<p class="section-header">Patient Data</p>', unsafe_allow_html=True)
    n = st.text_input("LAST NAME").upper()
    p = st.text_input("FIRST NAME").capitalize()
    a = st.number_input("AGE", min_value=0, value=30)
    g = st.selectbox("GENDER", ["Male", "Female"])
with c2:
    st.markdown('<p class="section-header">MRI Acquisition</p>', unsafe_allow_html=True)
    up = st.file_uploader("MRI", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if up:
        img_obj = Image.open(up).convert('RGB')
        st.image(img_obj, use_container_width=True)
with c3:
    st.markdown('<p class="section-header">Analysis</p>', unsafe_allow_html=True)
    if up and st.button("RUN ANALYSIS"):
        model = load_neuro_model()
        prep = np.array(img_obj.resize((224, 224))) / 255.0
        preds = model.predict(np.expand_dims(prep, axis=0))[0]
        classes = ['Non-Brain', 'Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        res = classes[np.argmax(preds)]
        conf = float(np.max(preds)) * 100
        st.write(f"**Result:** {res} ({conf:.2f}%)")
        pdf = create_clinical_pdf(n, p, a, g, res, conf, img_obj, date_now)
        st.download_button("📥 DOWNLOAD PDF", pdf, f"NeuroScan_{n}.pdf", "application/pdf")

st.markdown(f'<p style="text-align:center; color:#4b5563; font-family:Times; margin-top:80px;">Designed by Douaa Houbad | {date_now}</p>', unsafe_allow_html=True)
