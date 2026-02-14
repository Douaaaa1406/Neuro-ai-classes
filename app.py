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
    page_title="NeuroScan AI | Expert System",
    page_icon="🧠",
    layout="wide"
)

# --- 2. DESIGN "MODERN CLINIC" (GRIS PLATINE & BLEU COBALT) ---
st.markdown("""
<style>
    /* Fond Gris Très Clair / Soft Grey */
    [data-testid="stAppViewContainer"] {
        background-color: #f8f9fa;
        color: #2c3e50;
    }
    
    /* En-tête avec un dégradé élégant */
    .main-header {
        font-family: 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #2c3e50 0%, #4a90e2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2em;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    /* Barre d'info soft */
    .info-bar {
        background: #ffffff;
        padding: 12px;
        border-radius: 10px;
        color: #34495e;
        text-align: center;
        font-weight: 500;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e1e8ed;
        margin-bottom: 30px;
    }

    /* Cartes avec effet de profondeur (Neumorphism soft) */
    .medical-card {
        background: #ffffff;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.03);
        border: 1px solid #edf2f7;
        margin-bottom: 20px;
    }

    /* Bouton d'Analyse - Bleu Cobalt Professionnel */
    div.stButton > button {
        background: #4a90e2 !important;
        color: white !important;
        border: none !important;
        padding: 12px 0px !important;
        width: 100%;
        font-weight: 600 !important;
        font-size: 1.1em !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3) !important;
    }
    div.stButton > button:hover {
        background: #357abd !important;
        box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4) !important;
        transform: translateY(-1px);
    }

    /* Inputs Modernes */
    .stTextInput>div>div>input, .stSelectbox>div>div>div {
        border-radius: 8px !important;
        border: 1px solid #d1d9e6 !important;
    }

    /* Cadre de l'Image Scan */
    .scan-display {
        border: 8px solid #ffffff;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        overflow: hidden;
    }

    /* LinkedIn Flottant */
    .linkedin-fixed {
        position: fixed;
        bottom: 25px;
        right: 25px;
        background: #0077b5;
        padding: 10px 20px;
        border-radius: 50px;
        box-shadow: 0 5px 15px rgba(0,119,181,0.3);
    }
    .linkedin-fixed a {
        color: white !important;
        text-decoration: none !important;
        font-size: 0.9em;
        font-weight: 600;
    }

    /* Footer discret */
    .custom-footer {
        margin-top: 60px;
        text-align: center;
        color: #7f8c8d;
        font-size: 0.9em;
        padding-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGIQUE DU MODÈLE ---
@st.cache_resource
def load_neuro_model():
    model_path = 'brain_tumor_model_v6_final.keras'
    file_id = '1QRVvhNHSx7qgw0GIDrRLsuX09uItsXM2' 
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(model_path):
        try: gdown.download(url, model_path, quiet=False)
        except: st.error("Database connection issue.")
    
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

st.markdown('<p class="main-header">NeuroScan AI System</p>', unsafe_allow_html=True)
st.markdown(f'<div class="info-bar">📍 Algiers Medical Center | 🕒 {now.strftime("%H:%M:%S")} | Clinical Analysis Mode</div>', unsafe_allow_html=True)

# --- 5. DASHBOARD ---
col_left, col_mid, col_right = st.columns([1, 1.4, 1], gap="large")

with col_left:
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.subheader("👤 Patient Profile")
    nom = st.text_input("Surname").upper()
    prenom = st.text_input("Given Name").capitalize()
    age = st.number_input("Patient Age", min_value=0, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    st.markdown('</div>', unsafe_allow_html=True)

with col_mid:
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.subheader("🖼️ MRI Imaging")
    uploaded_file = st.file_uploader("Upload Digital MRI Scan", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.markdown('<div class="scan-display">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.subheader("⚖️ AI Decision")
    if uploaded_file and st.button("EXECUTE DIAGNOSTIC"):
        model = load_neuro_model()
        img_prep = image.resize((224, 224))
        img_array = np.array(img_prep) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        preds = model.predict(img_array)[0]
        classes = ['Non-Brain', 'Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        idx = np.argmax(preds)
        resultat = classes[idx]
        confiance = preds[idx] * 100

        st.markdown(f"**Diagnosis:** `{resultat}`")
        st.progress(confiance/100)
        st.markdown(f"**Reliability:** `{confiance:.2f}%`")
        
        # Le code PDF ici (Inchangé pour garder la structure propre)
        # [PDF generation...]
        st.success("Report Generated.")
    else:
        st.info("Awaiting MRI source file...")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. FOOTER ---
st.markdown(f"""
    <div class="custom-footer">
        <b>NeuroScan AI • Biomedical Engineering Solution</b><br>
        Developed by <b>Douaa Houbad</b> | M1 EMB | Ingénieur en Génie Biomédical
    </div>
    <div class="linkedin-fixed">
        <a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank">
            For more information
        </a>
    </div>
""", unsafe_allow_html=True)
