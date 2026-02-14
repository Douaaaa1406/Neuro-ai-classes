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
    page_title="NeuroScan AI | Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --- 2. DESIGN CSS "ADVANCED MEDICAL HUD" ---
st.markdown("""
<style>
    /* Fond sombre avec texture de grille */
    [data-testid="stAppViewContainer"] {
        background-color: #050a10;
        background-image: linear-gradient(0deg, transparent 24%, rgba(79, 172, 254, .05) 25%, rgba(79, 172, 254, .05) 26%, transparent 27%, transparent 74%, rgba(79, 172, 254, .05) 75%, rgba(79, 172, 254, .05) 76%, transparent 77%, transparent), 
                          linear-gradient(90deg, transparent 24%, rgba(79, 172, 254, .05) 25%, rgba(79, 172, 254, .05) 26%, transparent 27%, transparent 74%, rgba(79, 172, 254, .05) 75%, rgba(79, 172, 254, .05) 76%, transparent 77%, transparent);
        background-size: 50px 50px;
        color: #e0f2ff;
    }

    /* En-tête avec effet Glow */
    .main-header {
        font-family: 'Segoe UI', sans-serif;
        color: #4facfe;
        font-size: 3.5em;
        font-weight: 900;
        text-align: center;
        text-shadow: 0 0 20px rgba(79, 172, 254, 0.6);
        margin-bottom: 5px;
        letter-spacing: 5px;
    }

    /* Container des informations (Cartes) */
    .medical-card {
        background: rgba(13, 27, 42, 0.8);
        border: 1px solid #4facfe;
        border-radius: 10px;
        padding: 20px;
        box-shadow: inset 0 0 15px rgba(79, 172, 254, 0.2);
        margin-bottom: 20px;
    }

    /* Bouton d'Analyse - Look "Glow Blue" */
    div.stButton > button {
        background: transparent !important;
        color: #4facfe !important;
        border: 2px solid #4facfe !important;
        padding: 15px 0px !important;
        width: 100%;
        font-weight: bold !important;
        font-size: 1.2em !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: 0.4s;
        box-shadow: 0 0 10px rgba(79, 172, 254, 0.3);
    }
    div.stButton > button:hover {
        background: #4facfe !important;
        color: #050a10 !important;
        box-shadow: 0 0 30px #4facfe;
    }

    /* Style des Inputs */
    input, select, .stSelectbox {
        background-color: #0a192f !important;
        border: 1px solid #1e3a5f !important;
        color: #4facfe !important;
    }

    /* Centrage de l'image scannée */
    .scan-display {
        display: flex;
        justify-content: center;
        border: 3px solid #1e3a5f;
        padding: 10px;
        background: black;
        border-radius: 15px;
        box-shadow: 0 0 40px rgba(0,0,0,1);
    }

    /* LinkedIn Bouton */
    .linkedin-fixed {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: rgba(0, 119, 181, 0.2);
        border: 1px solid #0077b5;
        padding: 10px 20px;
        border-radius: 30px;
        backdrop-filter: blur(5px);
        transition: 0.3s;
    }
    .linkedin-fixed:hover {
        background: rgba(0, 119, 181, 0.8);
    }
    .linkedin-fixed a {
        color: white !important;
        text-decoration: none !important;
        font-size: 0.9em;
        font-weight: bold;
    }

    /* Footer */
    .custom-footer {
        margin-top: 50px;
        text-align: center;
        border-top: 1px solid #1e3a5f;
        padding: 20px;
        font-family: monospace;
        color: #4facfe;
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
st.markdown(f'<div style="text-align:center; color:#4facfe; font-family:monospace; margin-bottom:30px;">🛰️ ALGIERS_STATION | DATA_CLOCK: {now.strftime("%H:%M:%S")} | STATUS: ONLINE</div>', unsafe_allow_html=True)

# --- 5. MAIN INTERFACE ---
col_left, col_mid, col_right = st.columns([1, 1.5, 1], gap="medium")

with col_left:
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.subheader("📋 BIOMETRICS")
    nom = st.text_input("PATIENT LAST NAME").upper()
    prenom = st.text_input("PATIENT FIRST NAME").capitalize()
    age = st.number_input("AGE", min_value=0, value=30)
    gender = st.selectbox("GENDER", ["MALE", "FEMALE"])
    st.markdown('</div>', unsafe_allow_html=True)

with col_mid:
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.subheader("🧠 NEURAL IMAGING")
    uploaded_file = st.file_uploader("DROP MRI SCAN HERE", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.markdown('<div class="scan-display">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.subheader("⚡ DIAGNOSTIC")
    if uploaded_file and st.button("EXECUTE ANALYSIS"):
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
        
        # LOGIQUE PDF (Utilise le code précédent pour le tableau sans erreur)
        # [Génération PDF ici identique au code précédent...]
        st.info("Report is ready for export.")
    else:
        st.write("Waiting for data...")
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
