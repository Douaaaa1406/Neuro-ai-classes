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

# --- 2. DESIGN CSS VIF & BOUTONS ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(120deg, #eef2f3 0%, #8e9eab 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5em;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0px;
    }
    
    .info-bar {
        background: #1e3c72;
        padding: 15px;
        border-radius: 15px;
        color: #00f2fe;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }

    /* Bouton LinkedIn Personnalisé */
    .linkedin-btn {
        display: inline-block;
        padding: 10px 20px;
        background-color: #0077b5;
        color: white !important;
        text-decoration: none;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
        border: none;
        text-align: center;
    }
    .linkedin-btn:hover {
        background-color: #005582;
        transform: scale(1.05);
    }

    /* Style des boutons Streamlit */
    div.stButton > button {
        background: linear-gradient(45deg, #FF512F, #DD2476) !important;
        color: white !important;
        border-radius: 25px !important;
        border: none !important;
        height: 50px;
        width: 100%;
        font-weight: bold !important;
        box-shadow: 0 4px 15px rgba(221, 36, 118, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. CHARGEMENT DU MODÈLE (Cache) ---
@st.cache_resource
def load_neuro_model():
    model_path = 'brain_tumor_model_v6_final.keras'
    file_id = '1QRVvhNHSx7qgw0GIDrRLsuX09uItsXM2' 
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_path):
        with st.spinner("Initialisation de l'IA..."):
            try:
                gdown.download(url, model_path, quiet=False)
            except:
                st.error("Erreur de téléchargement.")
    
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

# --- 4. ENTÊTE & HEURE ALGERIE ---
algeria_tz = pytz.timezone('Africa/Algiers')
now = datetime.datetime.now(algeria_tz)

col_h1, col_h2, col_h3 = st.columns([1, 2, 1])
with col_h2:
    st.markdown('<p class="main-header">NeuroScan AI</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-bar">🇩🇿 Algérie | {now.strftime("%d/%m/%Y")} | ⌚ {now.strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)

# --- 5. INTERFACE & FORMULAIRE ---
col_form, col_img = st.columns([1, 1])

with col_form:
    st.subheader("👤 Dossier Patient")
    nom = st.text_input("Nom").upper()
    prenom = st.text_input("Prénom").capitalize()
    date_n = st.date_input("Date de naissance", value=datetime.date(1990, 1, 1))
    
    # BOUTON LINKEDIN ICI
    st.write("---")
    st.markdown("""
        <a href="www.linkedin.com/in/douaa-houbad-006b6a305">
            🔗 Contactez l'ingénieur sur LinkedIn
        </a>
    """, unsafe_allow_html=True)

with col_img:
    st.subheader("🖼️ Imagerie")
    uploaded_file = st.file_uploader("Charger IRM", type=["jpg", "png", "jpeg"])

# --- 6. LOGIQUE ANALYSE ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, width=350)

    if st.button("⚡ LANCER LE DIAGNOSTIC"):
        with st.spinner("Analyse neuronale en cours..."):
            model = load_neuro_model()
            
            # Prétraitement rapide
            img_processed = image.resize((224, 224))
            img_array = np.array(img_processed) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            preds = model.predict(img_array)[0]
            classes = ['Non-Cérébrale', 'Gliome', 'Méningiome', 'Sain', 'Pituitaire']
            idx = np.argmax(preds)
            resultat = classes[idx]
            confiance = preds[idx] * 100

        if idx == 0 or confiance < 70:
            st.error("⚠️ Image non valide ou incertitude trop élevée.")
        else:
            st.success(f"Analyse terminée : {resultat} ({confiance:.2f}%)")
            
            # --- GÉNÉRATION PDF ---
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "RAPPORT MEDICAL NEUROSCAN AI", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, f"Patient: {nom} {prenom} | Heure: {now.strftime('%H:%M')}", ln=True)
            pdf.cell(200, 10, f"Conclusion: {resultat} (Confiance: {confiance:.2f}%)", ln=True)
            
            pdf_output = pdf.output(dest='S').encode('latin-1')
            st.download_button("📥 Télécharger Rapport PDF", pdf_output, f"Rapport_{nom}.pdf", "application/pdf")

# --- FOOTER ---
st.markdown(f"""
    <div style="text-align: center; margin-top: 50px; color: #555;">
        Développé avec passion par <b>Houbad Douaa</b> • NeuroScan AI v6.1
    </div>
""", unsafe_allow_html=True)
