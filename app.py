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
    page_title="NeuroScan AI | Système Expert",
    page_icon="🧠",
    layout="wide"
)

# --- 2. DESIGN CSS "CINEMATIC DARK" ---
st.markdown("""
<style>
    /* Fond global */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle, #0d1b2a 0%, #03070b 100%);
        color: #e0e0e0;
    }

    /* Centrage du contenu principal */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }

    /* En-tête High-Tech */
    .main-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
        letter-spacing: 2px;
    }

    .info-bar {
        background: rgba(79, 172, 254, 0.1);
        padding: 10px;
        border-radius: 5px;
        border: 1px solid rgba(79, 172, 254, 0.3);
        color: #4facfe;
        text-align: center;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.9em;
        margin-bottom: 30px;
    }

    /* Style des Inputs (Sombre) */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div {
        background-color: #16213e !important;
        color: white !important;
        border: 1px solid #4facfe !important;
    }

    /* Centrage de l'image */
    .stImage > img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        border: 2px solid #4facfe;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(79, 172, 254, 0.4);
    }

    /* Bouton d'analyse style image */
    div.stButton > button {
        background: linear-gradient(180deg, #1976D2 0%, #0D47A1 100%) !important;
        color: white !important;
        border: 1px solid #4facfe !important;
        padding: 12px 0px !important;
        width: 100%;
        font-weight: bold !important;
        border-radius: 4px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5) !important;
    }

    /* Footer fixe et bouton LinkedIn */
    .footer-custom {
        text-align: center;
        padding: 20px;
        font-size: 0.8em;
        color: #4facfe;
        border-top: 1px solid rgba(79, 172, 254, 0.2);
        margin-top: 50px;
    }

    .linkedin-fixed {
        position: fixed;
        bottom: 15px;
        right: 15px;
        background-color: #0077b5;
        padding: 8px 15px;
        border-radius: 5px;
        z-index: 1000;
    }
    .linkedin-fixed a {
        color: white !important;
        text-decoration: none !important;
        font-weight: bold;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. CHARGEMENT DU MODÈLE (Inchangé) ---
@st.cache_resource
def load_neuro_model():
    model_path = 'brain_tumor_model_v6_final.keras'
    file_id = '1QRVvhNHSx7qgw0GIDrRLsuX09uItsXM2' 
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(model_path):
        try: gdown.download(url, model_path, quiet=False)
        except: st.error("Model download error.")
    
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

# --- 4. ENTÊTE ---
algeria_tz = pytz.timezone('Africa/Algiers')
now = datetime.datetime.now(algeria_tz)

st.markdown('<p class="main-header">BRAIN TUMOR CLASSIFICATION AI</p>', unsafe_allow_html=True)
st.markdown(f'<div class="info-bar">SYSTEM STATUS: ACTIVE | LOCATION: ALGERIA | {now.strftime("%d/%m/%Y - %H:%M:%S")}</div>', unsafe_allow_html=True)

# --- 5. LAYOUT PRINCIPAL (CENTRÉ) ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 👤 Patient Information")
    nom = st.text_input("Last Name").upper()
    prenom = st.text_input("First Name").capitalize()
    age = st.number_input("Age", min_value=0, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

with col_right:
    st.markdown("### 🧬 MRI Acquisition")
    uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        # Centrage forcé
        st.image(image, width=320)

# --- 6. ANALYSE ET GÉNÉRATION PDF ---
if uploaded_file and st.button("ANALYZE SCAN"):
    with st.spinner("Processing Neural Networks..."):
        model = load_neuro_model()
        img_prep = image.resize((224, 224))
        img_array = np.array(img_prep) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        preds = model.predict(img_array)[0]
        classes = ['Non-Brain', 'Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        idx = np.argmax(preds)
        resultat = classes[idx]
        confiance = preds[idx] * 100

    if idx == 0 or confiance < 60:
        st.error("Invalid Scan: The image does not meet neurological criteria.")
    else:
        st.success(f"Analysis Complete: {resultat} ({confiance:.2f}%)")

        # --- RAPPORT PDF CORRIGÉ (SANS CHEVAUCHEMENT) ---
        class PDF(FPDF):
            def header(self):
                self.set_fill_color(13, 27, 42)
                self.rect(0, 0, 210, 35, 'F')
                self.set_font('Arial', 'B', 18)
                self.set_text_color(255, 255, 255)
                self.cell(0, 15, 'NEUROSCAN AI - DIAGNOSTIC REPORT', 0, 1, 'C')
                self.ln(10)

        pdf = PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # 1. Infos Patient (Tableau)
        pdf.set_font('Arial', 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "1. PATIENT AND TEST DETAILS", 0, 1)
        
        pdf.set_font('Arial', '', 10)
        col_width = 45
        row_height = 8
        
        details = [
            ["Last Name", nom], ["First Name", prenom],
            ["Age / Gender", f"{age} / {gender}"],
            ["Test Date/Time", f"{now.strftime('%d/%m/%Y')} at {now.strftime('%H:%M')}"],
            ["Model Version", "MobileNetV2-NeuroV6"]
        ]
        
        for item in details:
            pdf.cell(col_width, row_height, item[0], 1)
            pdf.cell(100, row_height, item[1], 1)
            pdf.ln()

        # 2. Image (Positionnement fixe pour éviter chevauchement)
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "2. ANALYZED MRI SCAN", 0, 1)
        image.save("temp_pdf_img.jpg")
        # On place l'image et on ajoute un saut de ligne après sa hauteur
        pdf.image("temp_pdf_img.jpg", x=65, y=pdf.get_y(), w=80)
        pdf.set_y(pdf.get_y() + 85) # Décale le curseur après l'image (80mm + marge)

        # 3. Decision
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "3. CLINICAL CONCLUSION", 0, 1)
        pdf.set_fill_color(230, 242, 255)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 12, f"RESULT: {resultat.upper()}", 1, 1, 'C', True)
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 8, f"Confidence Level: {confiance:.2f}%", 0, 1, 'C')

        # 4. Disclaimer (Bas de page)
        pdf.ln(5)
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(100, 100, 100)
        disclaimer = "This report is generated by a deep learning AI model. It is intended for research and preliminary screening assistance only. It MUST NOT replace the definitive diagnosis or professional judgment of a qualified neuro-radiologist or medical doctor."
        pdf.multi_cell(0, 5, disclaimer, 0, 'C')

        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button("📥 DOWNLOAD PROFESSIONAL REPORT", pdf_bytes, f"NeuroScan_{nom}.pdf", "application/pdf")

# --- 7. FOOTER & LINKEDIN FIXE ---
st.markdown(f"""
    <div class="footer-custom">
        <b>NeuroScan AI Version 6.1.0</b> | Architecture: MobileNetV2-Custom-Neuro<br>
        Developed by: <b>Douaa Houbad</b> | M1 EMB | Biomedical Engineer
    </div>
    <div class="linkedin-fixed">
        <a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank">
            For more information
        </a>
    </div>
""", unsafe_allow_html=True)
