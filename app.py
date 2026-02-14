import streamlit as st
import tensorflow as tf
from PIL import Image, ImageEnhance
import numpy as np
from fpdf import FPDF
import datetime
import os
import gdown
import pytz

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="NeuroScan AI | Clinical Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --- 2. DESIGN "CLINICAL LIGHT" (NOIR & BLANC) ---
st.markdown("""
<style>
    /* Fond principal Clair */
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
        color: #000000;
    }
    
    /* En-tête en Noir Profond */
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #000000;
        font-size: 3em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }

    /* Barre d'état stylisée */
    .info-bar {
        background: #000000;
        padding: 10px;
        border-radius: 0px;
        color: #ffffff;
        text-align: center;
        font-family: monospace;
        font-size: 0.85em;
        margin-bottom: 30px;
        text-transform: uppercase;
    }

    /* Cartes Médicales (Contraste Noir/Blanc) */
    .medical-card {
        background: #ffffff;
        border: 2px solid #000000;
        border-radius: 0px; /* Style minimaliste angulaire */
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Bouton d'Analyse Noir */
    div.stButton > button {
        background: #000000 !important;
        color: #ffffff !important;
        border: none !important;
        padding: 15px 0px !important;
        width: 100%;
        font-weight: bold !important;
        font-size: 1.1em !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: 0.3s;
        border-radius: 0px !important;
    }
    div.stButton > button:hover {
        background: #333333 !important;
        transform: translateY(-2px);
    }

    /* Inputs personnalisés */
    input, select, .stSelectbox {
        border: 1px solid #000000 !important;
        border-radius: 0px !important;
    }

    /* Affichage de l'image (Cadre Noir) */
    .scan-display {
        display: flex;
        justify-content: center;
        border: 5px solid #000000;
        background: #f0f0f0;
        padding: 5px;
    }

    /* LinkedIn bouton discret */
    .linkedin-fixed {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: #000000;
        padding: 10px 15px;
        z-index: 1000;
    }
    .linkedin-fixed a {
        color: #ffffff !important;
        text-decoration: none !important;
        font-size: 0.8em;
        font-weight: bold;
    }

    /* Footer */
    .custom-footer {
        margin-top: 50px;
        text-align: center;
        border-top: 2px solid #000000;
        padding: 20px;
        font-family: 'Helvetica', sans-serif;
        color: #000000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_neuro_model():
    model_path = 'brain_tumor_model_v6_final.keras'
    file_id = '1QRVvhNHSx7qgw0GIDrRLsuX09uItsXM2' 
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(model_path):
        try: gdown.download(url, model_path, quiet=False)
        except: st.error("Error accessing AI Weights.")
    
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

# --- 4. NAVIGATION ---
algeria_tz = pytz.timezone('Africa/Algiers')
now = datetime.datetime.now(algeria_tz)

st.markdown('<p class="main-header">NEUROSCAN CORE AI</p>', unsafe_allow_html=True)
st.markdown(f'<div class="info-bar">ALGIERS STATION | SYSTEM TIME: {now.strftime("%H:%M:%S")} | MODE: CLINICAL_LIGHT</div>', unsafe_allow_html=True)

# --- 5. INTERFACE PRINCIPALE ---
col_left, col_mid, col_right = st.columns([1, 1.5, 1], gap="medium")

with col_left:
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.markdown("### 📋 PATIENT DATA")
    nom = st.text_input("LAST NAME").upper()
    prenom = st.text_input("FIRST NAME").capitalize()
    age = st.number_input("AGE", min_value=0, value=30)
    gender = st.selectbox("GENDER", ["MALE", "FEMALE", "OTHER"])
    st.markdown('</div>', unsafe_allow_html=True)

with col_mid:
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.markdown("### 🧠 MRI SCAN")
    uploaded_file = st.file_uploader("UPLOAD SOURCE FILE", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.markdown('<div class="scan-display">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.markdown("### ⚡ DIAGNOSTIC")
    if uploaded_file and st.button("RUN ANALYSIS"):
        model = load_neuro_model()
        img_prep = image.resize((224, 224))
        img_array = np.array(img_prep) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        preds = model.predict(img_array)[0]
        classes = ['Non-Brain', 'Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        idx = np.argmax(preds)
        resultat = classes[idx]
        confiance = preds[idx] * 100

        st.markdown(f"**RESULT:** `{resultat}`")
        st.markdown(f"**CONFIDENCE:** `{confiance:.2f}%`")
        
        # --- PDF GÉNÉRATION (Version Professionnelle Stable) ---
        class PDF(FPDF):
            def header(self):
                self.set_fill_color(0, 0, 0)
                self.rect(0, 0, 210, 35, 'F')
                self.set_font('Arial', 'B', 18)
                self.set_text_color(255, 255, 255)
                self.cell(0, 15, 'NEUROSCAN AI - CLINICAL REPORT', 0, 1, 'C')
                self.ln(10)

        pdf = PDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "1. PATIENT AND TEST DETAILS", 0, 1)
        
        pdf.set_font('Arial', '', 10)
        details = [
            ["Name", f"{nom} {prenom}"],
            ["Age / Gender", f"{age} / {gender}"],
            ["Date / Time", f"{now.strftime('%d/%m/%Y')} at {now.strftime('%H:%M')}"],
            ["AI Model", "MobileNetV2-NeuroV6"]
        ]
        for item in details:
            pdf.cell(50, 8, item[0], 1)
            pdf.cell(100, 8, item[1], 1)
            pdf.ln()

        pdf.ln(5)
        pdf.cell(0, 10, "2. SCAN ANALYSIS", 0, 1)
        image.save("temp_scan.jpg")
        pdf.image("temp_scan.jpg", x=65, w=80)
        pdf.set_y(pdf.get_y() + 85)

        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 12, f"CONCLUSION: {resultat.upper()}", 1, 1, 'C')
        
        pdf.set_y(-30)
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 5, "Disclaimer: AI-generated research tool. Not a substitute for professional medical advice.", 0, 'C')

        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button("📥 DOWNLOAD REPORT", pdf_bytes, f"Report_{nom}.pdf", "application/pdf")
    else:
        st.write("IDLE - System ready.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. FOOTER & LINKEDIN ---
st.markdown(f"""
    <div class="custom-footer">
        NEUROSCAN AI V6.1.0 | MOBILE-NET-V2 ENGINE<br>
        <b>DOUAA HOUBAD</b> | M1 EMB | BIOMEDICAL ENGINEER
    </div>
    <div class="linkedin-fixed">
        <a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank">
            FOR MORE INFORMATION
        </a>
    </div>
""", unsafe_allow_html=True)
