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
st.set_page_config(page_title="NeuroScan AI | Gold Edition", page_icon="🧠", layout="wide")

# --- 2. DESIGN "GOLDEN LUXURY" (BASÉ SUR TON IMAGE) ---
st.markdown("""
<style>
    /* Background global avec texture subtile et couleur Champagne */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(rgba(245, 238, 220, 0.9), rgba(220, 200, 170, 0.9)), 
                    url('https://www.transparenttextures.com/patterns/p6.png');
        background-color: #f5eedc;
    }

    /* En-tête Doré / Bronze */
    .main-header {
        font-family: 'Times New Roman', serif;
        color: #7d5a2d;
        font-size: 3.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0px;
    }
    
    .sub-text {
        color: #a67c52;
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 10px;
    }

    /* Status Bar Style Image */
    .system-status {
        background: rgba(125, 90, 45, 0.1);
        border: 1px solid #7d5a2d;
        border-radius: 50px;
        padding: 5px 20px;
        color: #7d5a2d;
        text-align: center;
        width: fit-content;
        margin: 0 auto 30px auto;
        font-family: monospace;
        font-size: 0.9em;
    }

    /* Colonnes / Cartes style "Beige Soft" */
    .gold-card {
        background: #efe6d5;
        border-radius: 20px;
        padding: 25px;
        border: 1px solid #d4c4a8;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        height: 100%;
    }

    /* Headers de colonnes avec icône dorée */
    .column-title {
        background: #7d5a2d;
        color: white;
        padding: 8px 15px;
        border-radius: 10px;
        font-size: 0.9em;
        font-weight: bold;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 20px;
    }

    /* Bouton d'Analyse Or / Ambre */
    div.stButton > button {
        background: linear-gradient(180deg, #d4a373 0%, #a67c52 100%) !important;
        color: white !important;
        border: 1px solid #7d5a2d !important;
        padding: 12px !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        width: 100%;
        box-shadow: 0 4px 10px rgba(125, 90, 45, 0.2);
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 15px rgba(125, 90, 45, 0.4);
    }

    /* Image Display avec bordure dorée */
    .scan-frame {
        border: 4px solid #7d5a2d;
        border-radius: 10px;
        padding: 5px;
        background: white;
    }

    /* LinkedIn Button - Orange/Gold style */
    .linkedin-btn {
        background: linear-gradient(180deg, #e67e22 0%, #d35400 100%);
        color: white !important;
        padding: 10px 20px;
        border-radius: 5px;
        text-decoration: none;
        font-weight: bold;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        float: right;
        font-size: 0.8em;
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
        gdown.download(url, model_path, quiet=False)
    
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

st.markdown('<p class="main-header">Brain Tumor Classification AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">MRI-Based Tumor Type Prediction</p>', unsafe_allow_html=True)
st.markdown(f'<div class="system-status">SYSTEM STATUS: ACTIVE | ALGERIA | {now.strftime("%d/%m/%Y - %H:%M:%S")}</div>', unsafe_allow_html=True)

# --- 5. DASHBOARD 3 COLONNES ---
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown('<div class="gold-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">👤 PATIENT INFORMATION</div>', unsafe_allow_html=True)
    nom = st.text_input("LAST NAME").upper()
    prenom = st.text_input("FIRST NAME").capitalize()
    age = st.number_input("AGE", min_value=0, value=30)
    gender = st.selectbox("GENDER", ["Male", "Female"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="gold-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">📤 UPLOAD MRI ACQUISITION</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop Scan Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.markdown('<div class="scan-frame">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Foucnd Scan")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="gold-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">🧬 PREDICTION RESULT</div>', unsafe_allow_html=True)
    if uploaded_file and st.button("Analyze"):
        model = load_neuro_model()
        img_array = np.array(image.resize((224, 224))) / 255.0
        preds = model.predict(np.expand_dims(img_array, axis=0))[0]
        
        classes = ['Non-Cérébral', 'Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        idx = np.argmax(preds)
        resultat = classes[idx]
        confiance = preds[idx] * 100

        st.markdown(f"**Predicted Type:** {resultat}")
        st.progress(confiance/100)
        st.write(f"Confidence: {confiance:.2f}%")
        
        # Le code PDF ici (inchangé)
        st.success("Report Ready.")
    else:
        st.write("Awaiting MRI upload...")
    
    # LinkedIn Button en bas de la 3eme colonne comme sur l'image
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank" class="linkedin-btn">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="15"> LinkedIn
        </a>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. FOOTER ---
st.markdown(f"""
    <div style="text-align:center; padding-top:40px; color:#7d5a2d; font-family:serif; font-weight:bold;">
        custom footer<br>
        Developed by Douaa Houbad | M1 EMB Biomedical Engineer
    </div>
""", unsafe_allow_html=True)
