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
st.set_page_config(page_title="NeuroScan AI | Studio Edition", page_icon="🧠", layout="wide")

# --- 2. DESIGN "CLINICAL STUDIO" (GRIS CLAIR & GRENAT) ---
st.markdown("""
<style>
    /* Fond Gris Clair avec Image IRM en filigrane */
    [data-testid="stAppViewContainer"] {
        background-color: #f4f4f4;
        background-image: linear-gradient(rgba(244, 244, 244, 0.92), rgba(244, 244, 244, 0.92)), 
                        url('https://img.freepik.com/vecteurs-libre/fond-radiographie-du-cerveau_23-2148065360.jpg');
        background-size: cover;
        background-attachment: fixed;
        color: #333333;
    }

    /* Titre Minimaliste */
    .main-header {
        font-family: 'Times New Roman', Times, serif;
        color: #1a1a1a;
        font-size: 4em;
        font-weight: 300;
        text-align: center;
        margin-top: -40px;
    }
    
    .sub-text {
        color: #800020;
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
        letter-spacing: 3px;
        margin-top: -20px;
    }

    /* Suppression des caisses (bordures et fonds lourds) */
    .clean-column {
        background: rgba(255, 255, 255, 0.6); /* Transparence douce */
        backdrop-filter: blur(5px);
        padding: 20px;
        border-radius: 15px;
        min-height: 600px;
        border-top: 2px solid #800020; /* Juste une ligne fine grenat en haut */
    }

    .section-title {
        color: #800020;
        font-family: 'Times New Roman', serif;
        font-size: 1.4em;
        font-weight: bold;
        margin-bottom: 20px;
        border-bottom: 1px solid #ddd;
    }

    /* Cadre Image IRM épuré */
    .scan-display {
        border: 1px solid #ccc;
        background: white;
        padding: 5px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    }

    /* Bouton d'Analyse Artistique */
    div.stButton > button {
        background: #800020 !important;
        color: white !important;
        border: none !important;
        padding: 15px !important;
        border-radius: 4px !important;
        font-family: 'Times New Roman', serif;
        font-size: 1.3em !important;
        width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background: #5a0016 !important;
        box-shadow: 0 5px 15px rgba(128, 0, 32, 0.3);
    }

    /* Custom Footer */
    .footer-text {
        text-align: center;
        padding: 50px;
        color: #666;
        font-family: 'Times New Roman', serif;
        font-style: italic;
    }

    /* LinkedIn */
    .linkedin-box {
        text-align: right;
        margin-top: 30px;
    }
    .linkedin-box a {
        color: #800020 !important;
        text-decoration: none;
        font-weight: bold;
        border: 1px solid #800020;
        padding: 5px 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. FONCTIONS ---
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
    
    # Configuration Police Times (standard dans FPDF)
    pdf.set_font("Times", 'B', 18)
    pdf.set_text_color(128, 0, 32) # Grenat
    pdf.cell(0, 20, "RAPPORT DE DIAGNOSTIC NEUROSCAN AI", 0, 1, 'C')
    
    pdf.set_draw_color(128, 0, 32)
    pdf.line(20, 32, 190, 32)
    
    pdf.ln(10)
    pdf.set_font("Times", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. INFORMATIONS PATIENT", 0, 1)
    
    # Tableau style classique Times Roman
    pdf.set_font("Times", '', 11)
    pdf.cell(50, 8, "Nom complet", 1)
    pdf.cell(100, 8, f"{nom} {prenom}", 1, 1)
    pdf.cell(50, 8, "Age / Genre", 1)
    pdf.cell(100, 8, f"{age} ans / {gender}", 1, 1)
    pdf.cell(50, 8, "Date de l'examen", 1)
    pdf.cell(100, 8, date_str, 1, 1)
    pdf.cell(50, 8, "Développeur", 1)
    pdf.cell(100, 8, "Douaa Houbad (M1 EMB)", 1, 1)
    pdf.cell(50, 8, "Modèle IA", 1)
    pdf.cell(100, 8, "MobileNetV2-NeuroV6", 1, 1)

    # Image
    pdf.ln(10)
    pdf.set_font("Times", 'B', 12)
    pdf.cell(0, 10, "2. CLICHÉ IRM ANALYSÉ", 0, 1)
    img.save("pdf_temp_img.png")
    pdf.image("pdf_temp_img.png", x=65, w=80)
    
    # Conclusion
    pdf.ln(10)
    pdf.set_font("Times", 'B', 14)
    pdf.set_fill_color(245, 245, 245)
    pdf.cell(0, 15, f"CONCLUSION : {resultat.upper()}", 1, 1, 'C', True)
    pdf.set_font("Times", 'I', 11)
    pdf.cell(0, 10, f"Indice de confiance : {confiance:.2f}%", 0, 1, 'C')

    pdf.set_y(-30)
    pdf.set_font("Times", 'I', 8)
    pdf.cell(0, 10, "Document confidentiel généré par NeuroScan AI System.", 0, 0, 'C')

    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFACE ---
algeria_tz = pytz.timezone('Africa/Algiers')
now = datetime.datetime.now(algeria_tz)
date_str = now.strftime("%d/%m/%Y - %H:%M")

st.markdown('<p class="main-header">NeuroScan AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">ENGINEERING & DIAGNOSTICS</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="clean-column">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Patient Data</p>', unsafe_allow_html=True)
    nom = st.text_input("NOM").upper()
    prenom = st.text_input("PRÉNOM").capitalize()
    age = st.number_input("ÂGE", min_value=0, value=30)
    gender = st.selectbox("GENRE", ["Male", "Female"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="clean-column">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">MRI Scan</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Cliché", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.markdown('<div class="scan-display">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="clean-column">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Analysis</p>', unsafe_allow_html=True)
    if uploaded_file and st.button("RUN DIAGNOSTIC"):
        model = load_neuro_model()
        img_prep = image.resize((224, 224))
        img_array = np.array(img_prep) / 255.0
        preds = model.predict(np.expand_dims(img_array, axis=0))[0]
        
        classes = ['Non-Cérébral', 'Gliome', 'Méningiome', 'Pas de Tumeur', 'Pituitaire']
        idx = np.argmax(preds)
        conf_val = float(preds[idx]) * 100
        res_text = classes[idx]

        st.markdown(f"**Result:** {res_text}")
        st.markdown(f"**Confidence:** {conf_val:.1f}%")
        
        # Barre de progression Grenat
        st.markdown(f'<div class="p-bar-container"><div class="p-bar-fill" style="width:{conf_val}%;"></div></div>', unsafe_allow_html=True)

        # PDF avec police Times
        pdf_data = create_medical_pdf(nom, prenom, age, gender, res_text, conf_val, image, date_str)
        st.download_button("📥 DOWNLOAD REPORT (Times Roman)", pdf_data, f"NeuroScan_{nom}.pdf", "application/pdf")
    else:
        st.write("Ready for analysis.")

    st.markdown(f'<div class="linkedin-box"><a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank">LinkedIn</a></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<div class="footer-text">Designed by Douaa Houbad | M1 EMB Biomedical Engineer | {date_str}</div>', unsafe_allow_html=True)
