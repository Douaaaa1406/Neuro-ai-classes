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
    page_title="NeuroScan AI | High-Tech Diagnostic",
    page_icon="🧠",
    layout="wide"
)

# --- 2. DESIGN CSS "PRO-DARK" (INSPIRÉ DE TON IMAGE) ---
st.markdown("""
<style>
    /* Fond principal sombre et technologique */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle, #101e33 0%, #060b13 100%);
        color: #e0e0e0;
    }
    
    /* En-tête */
    .main-header {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
        letter-spacing: 3px;
    }

    /* Barre d'info High-Tech */
    .info-bar {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(79, 172, 254, 0.3);
        color: #4facfe;
        text-align: center;
        font-family: monospace;
        margin-bottom: 25px;
    }

    /* Bouton ANALYZE (Style Image) */
    div.stButton > button {
        background: linear-gradient(180deg, #1976D2 0%, #0D47A1 100%) !important;
        color: white !important;
        border: 1px solid #4facfe !important;
        padding: 10px 40px !important;
        border-radius: 5px !important;
        font-weight: bold !important;
        text-transform: uppercase;
        width: 100%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    
    /* Centrage de l'image */
    .stImage {
        display: flex;
        justify-content: center;
    }

    /* Cartes de résultats */
    .result-card {
        background: rgba(16, 30, 51, 0.8);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #1976D2;
        box-shadow: 0 0 20px rgba(79, 172, 254, 0.2);
    }

    /* LinkedIn en bas à droite */
    .linkedin-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 100;
    }
    .linkedin-btn {
        background: #0077b5;
        color: white !important;
        padding: 10px 20px;
        border-radius: 5px;
        text-decoration: none;
        font-weight: bold;
        font-size: 14px;
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
        with st.spinner("Initialisation des serveurs de prédiction..."):
            try:
                gdown.download(url, model_path, quiet=False)
            except: st.error("Lien modèle invalide.")
    
    # Architecture (V6)
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

# --- 4. NAVIGATION & TEMPS RÉEL ---
algeria_tz = pytz.timezone('Africa/Algiers')
now = datetime.datetime.now(algeria_tz)

st.markdown('<p class="main-header">BRAIN TUMOR CLASSIFICATION AI</p>', unsafe_allow_html=True)
st.markdown(f'<div class="info-bar">SYSTEM STATUS: ACTIVE | LOCATION: ALGERIA | {now.strftime("%d/%m/%Y - %H:%M:%S")}</div>', unsafe_allow_html=True)

# --- 5. INTERFACE UTILISATEUR ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📝 Patient Information")
    with st.container():
        nom = st.text_input("Last Name").upper()
        prenom = st.text_input("First Name").capitalize()
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])

with col_right:
    st.markdown("### 🧬 MRI Acquisition")
    uploaded_file = st.file_uploader("Upload MRI Scan (DICOM/JPG/PNG)", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        # Centrage automatique via colonnes internes
        c1, c2, c3 = st.columns([1, 2, 1])
        c2.image(image, use_container_width=True, caption="Current Scan")

# --- 6. ANALYSE ET RAPPORT ---
if uploaded_file and st.button("ANALYZE SCAN"):
    with st.spinner("Running Neural Analysis..."):
        model = load_neuro_model()
        
        # Preprocessing
        img_input = image.resize((224, 224))
        img_array = np.array(img_input) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        preds = model.predict(img_array)[0]
        classes = ['Non-Brain Image', 'Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']
        idx = np.argmax(preds)
        resultat = classes[idx]
        confiance = preds[idx] * 100

    if idx == 0 or confiance < 65:
        st.error("Validation Failed: The uploaded image is not a valid brain MRI.")
    else:
        st.markdown(f"""
            <div class="result-card">
                <h2 style="color:#4facfe;">PREDICTION RESULT</h2>
                <p style="font-size:1.2em;">Predicted Type: <b>{resultat}</b></p>
                <div style="background:#060b13; border-radius:5px; width:100%; height:25px;">
                    <div style="background:#4facfe; width:{confiance}%; height:100%; border-radius:5px; text-align:center; color:white; font-size:0.8em;">
                        {confiance:.1f}%
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # --- 7. GÉNÉRATION DU PDF PROFESSIONNEL ---
        class PDF(FPDF):
            def header(self):
                self.set_fill_color(16, 30, 51)
                self.rect(0, 0, 210, 40, 'F')
                self.set_font('Arial', 'B', 20)
                self.set_text_color(255, 255, 255)
                self.cell(0, 20, 'DIAGNOSTIC REPORT: NEUROSCAN AI', 0, 1, 'C')
                self.ln(10)

        pdf = PDF()
        pdf.add_page()
        
        # Section Infos (Tableau)
        pdf.set_font('Arial', 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "1. PATIENT INFORMATION", 0, 1)
        
        pdf.set_font('Arial', '', 11)
        data = [
            ["Name", f"{nom} {prenom}"],
            ["Age / Gender", f"{age} / {gender}"],
            ["Test Date", now.strftime("%d/%m/%Y")],
            ["Test Time", now.strftime("%H:%M")],
            ["Prediction Model", "MobileNetV2-NeuroV6"]
        ]
        
        for row in data:
            pdf.cell(50, 10, row[0], 1)
            pdf.cell(140, 10, row[1], 1)
            pdf.ln()

        # Image MRI
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "2. ANALYZED MRI SCAN", 0, 1)
        image.save("temp_report.jpg")
        pdf.image("temp_report.jpg", x=60, w=90)
        
        # Decision
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "3. CLINICAL DECISION", 0, 1)
        pdf.set_fill_color(240, 240, 240)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 15, f"RESULT: {resultat.upper()}", 1, 1, 'C', True)
        pdf.set_font('Arial', 'I', 11)
        pdf.cell(0, 10, f"Algorithm Confidence: {confiance:.2f}%", 0, 1, 'C')

        # Footer Clause
        pdf.set_y(-30)
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 5, "Disclaimer: This report is generated by a deep learning model. It is intended for research assistance only and MUST NOT replace the professional judgment of a qualified neuro-radiologist or medical doctor.", 0, 'C')

        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button("📥 DOWNLOAD PROFESSIONAL REPORT", pdf_bytes, f"Report_{nom}.pdf", "application/pdf")

# --- 7. FOOTER & LINKEDIN ---
st.markdown("""<br><br><br>""", unsafe_allow_html=True)
st.markdown(f"""
    <div style="text-align: center; color: #4facfe; font-family: monospace; border-top: 1px solid rgba(79,172,254,0.2); padding-top: 20px;">
        <p>Version 6.1.0 | Model: MobileNetV2-Custom-Neuro</p>
        <p>Developed by: <b>Douaa Houbad</b> | M1 EMB | Biomedical Engineer</p>
    </div>w
    <div class="linkedin-container">
        <a href="ww.linkedin.com/in/douaa-houbad-006b6a305" target="_blank" class="linkedin-btn">www.linkedin.com/in/douaa-houbad-006b6a305
            LinkedIn Profile
        </a>
    </div>
""", unsafe_allow_html=True)
