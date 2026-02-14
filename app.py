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

# --- 2. STYLE ÉPURÉ "HIGH-CONTRAST SOFT" ---
st.markdown("""
<style>
    /* Fond Gris Clair Uniforme (Contraste amélioré) */
    [data-testid="stAppViewContainer"] {
        background-color: #e8ebf0;
        color: #2c3e50;
    }

    /* En-tête sans bloc */
    .header-container {
        text-align: center;
        margin-top: -50px;
        margin-bottom: 40px;
    }

    .main-title {
        font-family: 'Times New Roman', serif;
        color: #1a1a1a;
        font-size: 3.2em;
        margin-bottom: 0px;
    }

    .sub-title {
        font-family: 'Times New Roman', serif;
        color: #800020; /* Grenat */
        font-size: 1.1em;
        letter-spacing: 4px;
        font-weight: bold;
    }

    /* Suppression des rectangles blancs : On utilise des conteneurs invisibles */
    .column-content {
        padding: 10px;
        border-top: 2px solid #800020; /* Ligne de séparation élégante */
        margin-top: 10px;
    }

    .label-style {
        font-family: 'Times New Roman', serif;
        color: #800020;
        font-size: 1.3em;
        font-weight: bold;
        margin-bottom: 15px;
    }

    /* Bouton d'Analyse - Style Minimaliste Grenat */
    div.stButton > button {
        background-color: #800020 !important;
        color: white !important;
        border: none !important;
        border-radius: 2px !important;
        height: 45px;
        width: 100%;
        font-family: 'Times New Roman', serif;
        font-size: 1.1em !important;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #4d0013 !important;
    }

    /* Footer LinkedIn discret */
    .footer-link {
        text-align: right;
        margin-top: 50px;
    }
    .footer-link a {
        color: #800020 !important;
        text-decoration: none;
        font-family: 'Times New Roman', serif;
        font-weight: bold;
        border-bottom: 1px solid #800020;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. FONCTIONS TECHNIQUES ---
@st.cache_resource
def load_neuro_model():
    model_path = 'brain_tumor_model_v6_final.keras'
    if not os.path.exists(model_path):
        # [cite_start]Utilisation du modèle spécifié dans le rapport [cite: 3]
        gdown.download(f'https://drive.google.com/uc?id=1QRVvhNHSx7qgw0GIDrRLsuX09uItsXM2', model_path, quiet=False)
    
    # Architecture simplifiée pour l'inférence
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    model.load_weights(model_path)
    return model

def create_pdf_report(nom, prenom, age, gender, result, confidence, img, date_str):
    pdf = FPDF()
    pdf.add_page()
    # Police Times New Roman pour le prestige
    pdf.set_font("Times", 'B', 20)
    pdf.set_text_color(128, 0, 32)
    pdf.cell(0, 20, "DIAGNOSTIC REPORT: NEUROSCAN AI", 0, 1, 'C')
    
    pdf.ln(10)
    pdf.set_font("Times", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    [cite_start]pdf.cell(0, 10, "1. PATIENT INFORMATION", 0, 1) [cite: 2]
    
    pdf.set_font("Times", '', 11)
    # [cite_start]Tableau de données [cite: 3]
    pdf.cell(50, 8, "Name", 1)
    pdf.cell(100, 8, f"{nom} {prenom}", 1, 1)
    pdf.cell(50, 8, "Age / Gender", 1)
    pdf.cell(100, 8, f"{age} / {gender}", 1, 1)
    pdf.cell(50, 8, "Prediction Model", 1)
    pdf.cell(100, 8, "MobileNetV2-NeuroV6", 1, 1)
    pdf.cell(50, 8, "Date", 1)
    pdf.cell(100, 8, date_str, 1, 1)

    pdf.ln(10)
    img.save("scan_temp.png")
    pdf.image("scan_temp.png", x=60, w=90)
    
    pdf.ln(10)
    pdf.set_font("Times", 'B', 14)
    pdf.set_fill_color(240, 240, 240)
    [cite_start]pdf.cell(0, 12, f"RESULT: {result.upper()}", 1, 1, 'C', True) [cite: 6]
    pdf.set_font("Times", 'I', 11)
    [cite_start]pdf.cell(0, 10, f"Confidence: {confidence:.2f}%", 0, 1, 'C') [cite: 7]

    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFACE ---
algeria_tz = pytz.timezone('Africa/Algiers')
now = datetime.datetime.now(algeria_tz)
date_full = now.strftime("%d/%m/%Y - %H:%M")

st.markdown(f"""
<div class="header-container">
    <p class="main-title">NeuroScan AI</p>
    <p class="sub-title">ENGINEERING & DIAGNOSTICS</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<p class="label-style">Patient Data</p>', unsafe_allow_html=True)
    st.markdown('<div class="column-content">', unsafe_allow_html=True)
    nom = st.text_input("LAST NAME").upper()
    prenom = st.text_input("FIRST NAME").capitalize()
    [cite_start]age = st.number_input("AGE", min_value=0, value=30) [cite: 3]
    [cite_start]gender = st.selectbox("GENDER", ["Male", "Female"]) [cite: 3]
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<p class="label-style">MRI Scan</p>', unsafe_allow_html=True)
    st.markdown('<div class="column-content">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload MRI", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<p class="label-style">Analysis</p>', unsafe_allow_html=True)
    st.markdown('<div class="column-content">', unsafe_allow_html=True)
    if uploaded_file and st.button("ANALYZE SCAN"):
        model = load_neuro_model()
        img_array = np.array(image.resize((224, 224))) / 255.0
        preds = model.predict(np.expand_dims(img_array, axis=0))[0]
        
        # Classes définies dans le système
        classes = ['Non-Brain', 'Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        idx = np.argmax(preds)
        res_text = classes[idx]
        conf_val = float(preds[idx]) * 100

        st.markdown(f"**Status:** {res_text}")
        st.markdown(f"**Accuracy:** {conf_val:.2f}%")
        
        # Rapport PDF en Times New Roman
        pdf_out = create_pdf_report(nom, prenom, age, gender, res_text, conf_val, image, date_full)
        st.download_button("📥 DOWNLOAD PDF REPORT", pdf_out, f"Diagnostic_{nom}.pdf", "application/pdf")
    else:
        st.write("Awaiting MRI acquisition...")
    
    st.markdown(f'<div class="footer-link"><a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank">LinkedIn Profile</a></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<p style="text-align:center; color:#555; font-family:Times; padding-top:60px;">Designed by Douaa Houbad | M1 EMB Biomedical Engineer | 2026</p>', unsafe_allow_html=True)
