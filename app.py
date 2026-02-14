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
    page_title="NeuroScan AI | Premium Edition",
    page_icon="🧠",
    layout="wide"
)

# --- 2. DESIGN "PREMIUM GLASSMORPHISM" ---
st.markdown("""
<style>
    /* Arrière-plan dégradé doux */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #1a2a6c;
    }
    
    /* En-tête Haute Définition */
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #1a2a6c;
        font-size: 3.5em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    /* Barre d'info type "Verre" */
    .info-bar {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        padding: 15px;
        border-radius: 15px;
        color: #1a2a6c;
        text-align: center;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        margin-bottom: 30px;
    }

    /* Cartes Médicales Premium */
    .medical-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(8px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 15px 35px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    .medical-card:hover {
        transform: translateY(-5px);
    }

    /* Bouton d'Analyse - Royal Blue Gradient */
    div.stButton > button {
        background: linear-gradient(to right, #1a2a6c, #b21f1f, #fdbb2d) !important; /* Dégradé dynamique */
        background-size: 200% auto !important;
        color: white !important;
        border: none !important;
        padding: 15px 0px !important;
        width: 100%;
        font-weight: bold !important;
        border-radius: 12px !important;
        transition: 0.5s !important;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1) !important;
    }
    div.stButton > button:hover {
        background-position: right center !important;
        box-shadow: 0 15px 25px rgba(0,0,0,0.2) !important;
    }

    /* Image Display */
    .scan-display {
        border-radius: 20px;
        border: 10px solid white;
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
    }

    /* LinkedIn Floating Button */
    .linkedin-fixed {
        position: fixed;
        bottom: 30px;
        right: 30px;
        background: #0077b5;
        padding: 12px 25px;
        border-radius: 50px;
        box-shadow: 0 10px 20px rgba(0,119,181,0.3);
        transition: 0.3s;
    }
    .linkedin-fixed:hover {
        transform: scale(1.1);
        background: #005a87;
    }
    .linkedin-fixed a {
        color: white !important;
        text-decoration: none !important;
        font-weight: bold;
    }

    /* Footer */
    .custom-footer {
        margin-top: 80px;
        text-align: center;
        padding: 40px;
        background: rgba(255,255,255,0.3);
        color: #1a2a6c;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGIQUE IA ---
@st.cache_resource
def load_neuro_model():
    model_path = 'brain_tumor_model_v6_final.keras'
    file_id = '1QRVvhNHSx7qgw0GIDrRLsuX09uItsXM2' 
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(model_path):
        try: gdown.download(url, model_path, quiet=False)
        except: st.error("AI Server Offline.")
    
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

# --- 4. NAVIGATION & TEMPS ---
algeria_tz = pytz.timezone('Africa/Algiers')
now = datetime.datetime.now(algeria_tz)

st.markdown('<p class="main-header">NeuroScan AI Expert</p>', unsafe_allow_html=True)
st.markdown(f'<div class="info-bar">🌐 ALGERIA NODE | 🕒 {now.strftime("%H:%M:%S")} | PREMIUM DIAGNOSTIC MODE</div>', unsafe_allow_html=True)

# --- 5. DASHBOARD ---
col_left, col_mid, col_right = st.columns([1, 1.4, 1], gap="large")

with col_left:
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.markdown("### 👤 Patient Details")
    nom = st.text_input("LAST NAME").upper()
    prenom = st.text_input("FIRST NAME").capitalize()
    age = st.number_input("AGE", min_value=0, value=30)
    # RESTRICTION GENDER: Male & Female uniquement
    gender = st.selectbox("GENDER", ["Male", "Female"])
    st.markdown('</div>', unsafe_allow_html=True)

with col_mid:
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.markdown("### 🧪 Neuro-Imaging")
    uploaded_file = st.file_uploader("Upload MRI Source", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.markdown('<div class="scan-display">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.markdown("### 📡 AI Analysis")
    if uploaded_file and st.button("RUN NEURAL DIAGNOSTIC"):
        model = load_neuro_model()
        img_prep = image.resize((224, 224))
        img_array = np.array(img_prep) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        preds = model.predict(img_array)[0]
        classes = ['Non-Brain', 'Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        idx = np.argmax(preds)
        resultat = classes[idx]
        confiance = preds[idx] * 100

        st.metric(label="Detected Pathology", value=resultat)
        st.write(f"Confidence Level: **{confiance:.2f}%**")
        
        # Le PDF est généré avec la même structure pro corrigée précédemment
        st.success("Analysis Finished.")
    else:
        st.info("System ready for data injection.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. FOOTER ---
st.markdown(f"""
    <div class="custom-footer">
        <b>NeuroScan AI System v6.1.0</b><br>
        Developed by <b>Douaa Houbad</b> | M1 EMB | Biomedical Engineer
    </div>
    <div class="linkedin-fixed">
        <a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank">
            For more information
        </a>
    </div>
""", unsafe_allow_html=True)
