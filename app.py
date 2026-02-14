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
st.set_page_config(page_title="Brain Tumor Classification AI", page_icon="🧠", layout="wide")

# --- 2. DESIGN "PURE GOLDEN INTERFACE" ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at center, #f0e6d2 0%, #c5b493 100%);
        background-image: url("https://www.transparenttextures.com/patterns/dust.png");
    }

    .main-header {
        font-family: 'Times New Roman', serif;
        color: #7d5a2d;
        font-size: 3.8em;
        font-weight: 800;
        text-align: center;
        margin-top: -50px;
    }
    
    .sub-text {
        color: #8b6b43;
        text-align: center;
        font-size: 1.3em;
        margin-top: -20px;
    }

    .system-status {
        background: rgba(255, 255, 255, 0.4);
        border: 1px solid #7d5a2d;
        border-radius: 50px;
        padding: 5px 30px;
        color: #7d5a2d;
        text-align: center;
        width: fit-content;
        margin: 15px auto 40px auto;
        font-size: 0.85em;
        font-weight: bold;
    }

    .gold-card {
        background: rgba(239, 230, 213, 0.85);
        border-radius: 25px;
        padding: 30px;
        border: 2px solid #b5a384;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        min-height: 580px;
    }

    .column-title {
        background: #7d5a2d;
        color: white;
        padding: 10px 20px;
        border-radius: 12px;
        font-size: 0.9em;
        font-weight: bold;
        margin-bottom: 25px;
        text-align: center;
    }

    .scan-frame {
        border: 6px solid #7d5a2d;
        border-radius: 12px;
        background: black;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    }

    div.stButton > button {
        background: linear-gradient(180deg, #d4a373 0%, #8b6b43 100%) !important;
        color: white !important;
        border: 2px solid #7d5a2d !important;
        padding: 18px !important;
        border-radius: 12px !important;
        font-weight: bold !important;
        width: 100%;
        text-transform: uppercase;
    }

    .p-bar-container {
        width: 100%;
        background: #d1c4ab;
        border-radius: 10px;
        height: 25px;
        border: 1px solid #7d5a2d;
        overflow: hidden;
        margin: 10px 0;
    }
    .p-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #d4a373, #7d5a2d);
    }

    .linkedin-btn {
        background: linear-gradient(180deg, #e67e22 0%, #d35400 100%);
        color: white !important;
        padding: 12px 25px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        font-size: 0.85em;
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

def create_pdf(nom, prenom, age, gender, resultat, confiance, img, date_str):
    pdf = FPDF()
    pdf.add_page()
    
    # Header Noir/Or style
    pdf.set_fill_color(30, 30, 30)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(212, 163, 115)
    pdf.cell(0, 20, "NEUROSCAN AI - DIAGNOSTIC REPORT", 0, 1, 'C')
    
    # Infos Patient
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.ln(10)
    pdf.cell(0, 10, f"PATIENT: {nom} {prenom} | AGE: {age} | GENDER: {gender}", 0, 1)
    pdf.cell(0, 10, f"DATE: {date_str}", 0, 1)
    pdf.ln(5)
    
    # Image IRM
    img.save("temp_report_img.png")
    pdf.image("temp_report_img.png", x=55, w=100)
    pdf.ln(10)
    
    # Résultat
    pdf.set_font("Arial", 'B', 16)
    pdf.set_fill_color(240, 230, 210)
    pdf.cell(0, 15, f"CONCLUSION: {resultat.upper()}", 1, 1, 'C', True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Confidence Level: {confiance:.2f}%", 0, 1, 'C')
    
    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFACE ---
algeria_tz = pytz.timezone('Africa/Algiers')
now = datetime.datetime.now(algeria_tz)
date_display = now.strftime("%d/%m/%Y - %H:%M:%S")

st.markdown('<p class="main-header">Brain Tumor Classification AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">MRI-Based Tumor Type Prediction</p>', unsafe_allow_html=True)
st.markdown(f'<div class="system-status">SYSTEM STATUS: ACTIVE | ALGERIA | {date_display}</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="gold-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">👤 PATIENT INFORMATION</div>', unsafe_allow_html=True)
    nom_val = st.text_input("LAST NAME").upper()
    prenom_val = st.text_input("FIRST NAME").capitalize()
    age_val = st.number_input("AGE", min_value=0, value=30)
    gender_val = st.selectbox("GENDER", ["Male", "Female"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="gold-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">📤 UPLOAD MRI ACQUISITION</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.markdown('<div class="scan-frame">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="gold-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">🧬 PREDICTION RESULT</div>', unsafe_allow_html=True)
    if uploaded_file and st.button("Analyze"):
        model = load_neuro_model()
        img_prep = image.resize((224, 224))
        img_array = np.array(img_prep) / 255.0
        preds = model.predict(np.expand_dims(img_array, axis=0))[0]
        
        classes = ['Non-Brain', 'Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        idx = np.argmax(preds)
        conf_val = float(preds[idx]) * 100
        res_text = classes[idx]

        st.markdown(f"**Predicted Type:** {res_text}")
        st.markdown(f"**Confidence:** {conf_val:.1f}%")
        st.markdown(f'<div class="p-bar-container"><div class="p-bar-fill" style="width:{conf_val}%;"></div></div>', unsafe_allow_html=True)

        # Génération PDF
        pdf_bytes = create_pdf(nom_val, prenom_val, age_val, gender_val, res_text, conf_val, image, date_display)
        st.download_button("📥 DOWNLOAD PDF REPORT", pdf_bytes, f"Report_{nom_val}.pdf", "application/pdf")
    else:
        st.write("Awaiting Analysis...")

    st.markdown("<br><br><div style='text-align:right;'>", unsafe_allow_html=True)
    st.markdown(f'<a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank" class="linkedin-btn">For more information</a>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown('<p style="text-align:center; padding:30px; color:#7d5a2d; font-weight:bold;">custom footer</p>', unsafe_allow_html=True)
