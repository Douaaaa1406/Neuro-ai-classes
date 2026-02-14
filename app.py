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

# --- 2. STYLE ÉPURÉ : GRIS CLAIR & GRENAT (SANS RECTANGLES) ---
st.markdown("""
<style>
    /* Fond Gris Clair Uniforme */
    [data-testid="stAppViewContainer"] {
        background-color: #e5e7eb !important;
        color: #111827;
    }

    /* Suppression des fonds de colonnes (élimine les rectangles blancs) */
    [data-testid="column"], .stColumn > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
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
        color: #800020;
        letter-spacing: 5px;
        text-transform: uppercase;
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

    /* Bouton d'Analyse Grenat */
    div.stButton > button {
        background-color: #800020 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        font-family: 'Times New Roman', serif;
        font-size: 1.2em !important;
        width: 100%;
        height: 50px;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #4d0013 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGIQUE TECHNIQUE CORRIGÉE ---
@st.cache_resource
def load_neuro_model():
    model_path = 'brain_tumor_model_v6_final.keras'
    if not os.path.exists(model_path):
        # Lien de votre modèle sur Google Drive
        gdown.download(f'https://drive.google.com/uc?id=1QRVvhNHSx7qgw0GIDrRLsuX09uItsXM2', model_path, quiet=False)
    
    # Correction de la ValueError : Chargement du modèle complet
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception:
        # Solution de secours si c'est un fichier de poids
        base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        model.load_weights(model_path)
    return model

def create_clinical_pdf(nom, prenom, age, gender, result, confidence, img, date_str):
    pdf = FPDF()
    pdf.add_page()
    
    # Police Times New Roman
    pdf.set_font("Times", 'B', 20)
    pdf.set_text_color(128, 0, 32)
    pdf.cell(0, 20, "NEUROSCAN AI - CLINICAL REPORT", 0, 1, 'C')
    
    pdf.ln(10)
    pdf.set_font("Times", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. PATIENT & SYSTEM INFORMATION", 0, 1)
    
    pdf.set_font("Times", '', 11)
    # Tableau incluant les détails du développeur
    info_data = [
        ["Patient Name", f"{nom} {prenom}"],
        ["Age / Gender", f"{age} / {gender}"],
        ["Lead Developer", "Douaa Houbad (M1 EMB)"],
        ["AI Architecture", "MobileNetV2-NeuroV6"],
        ["Analysis Date", date_str]
    ]
    
    for row in info_data:
        pdf.cell(60, 8, row[0], 1)
        pdf.cell(100, 8, row[1], 1)
        pdf.ln()

    pdf.ln(10)
    img.save("temp_report.png")
    pdf.image("temp_report.png", x=60, w=90)
    
    pdf.ln(10)
    pdf.set_font("Times", 'B', 14)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 15, f"DIAGNOSTIC : {result.upper()}", 1, 1, 'C', True)
    pdf.set_font("Times", 'I', 11)
    pdf.cell(0, 10, f"Confidence Index : {confidence:.2f}%", 0, 1, 'C')
    
    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFACE ---
algeria_tz = pytz.timezone('Africa/Algiers')
date_now = datetime.datetime.now(algeria_tz).strftime("%d/%m/%Y - %H:%M")

st.markdown(f"""
    <div class="header-area">
        <p class="title-text">NeuroScan AI</p>
        <p class="subtitle-text">Medical Engineering Solutions</p>
    </div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown('<p class="section-header">Patient Data</p>', unsafe_allow_html=True)
    n = st.text_input("LAST NAME").upper()
    p = st.text_input("FIRST NAME").capitalize()
    a = st.number_input("AGE", min_value=0, value=30)
    g = st.selectbox("GENDER", ["Male", "Female"])

with c2:
    st.markdown('<p class="section-header">MRI Acquisition</p>', unsafe_allow_html=True)
    up = st.file_uploader("Upload MRI", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if up:
        img_obj = Image.open(up).convert('RGB')
        st.image(img_obj, use_container_width=True)

with c3:
    st.markdown('<p class="section-header">Neural Diagnostic</p>', unsafe_allow_html=True)
    if up and st.button("RUN ANALYSIS"):
        net = load_neuro_model()
        prep = np.array(img_obj.resize((224, 224))) / 255.0
        prediction = net.predict(np.expand_dims(prep, axis=0))[0]
        
        classes = ['Non-Brain', 'Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        best_idx = np.argmax(prediction)
        res_str = classes[best_idx]
        conf_score = float(prediction[best_idx]) * 100

        st.write(f"**Result:** {res_str}")
        st.write(f"**Confidence:** {conf_score:.2f}%")
        
        pdf_out = create_clinical_pdf(n, p, a, g, res_str, conf_score, img_obj, date_now)
        st.download_button("📥 DOWNLOAD CLINICAL PDF", pdf_out, f"NeuroScan_{n}.pdf", "application/pdf")
    else:
        st.write("System ready for scan...")

st.markdown(f'<p style="text-align:center; color:#4b5563; font-family:Times; margin-top:100px;">Designed by Douaa Houbad | Biomedical Engineer | {date_now}</p>', unsafe_allow_html=True)
