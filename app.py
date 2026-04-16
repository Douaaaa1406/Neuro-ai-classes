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
st.set_page_config(page_title="NeuroScan AI | Diagnostic", page_icon="🧠", layout="wide")

# --- 2. DESIGN CLINIQUE & BIO-TECH (BEIGE MEDICAL, VERT, BLEU) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;600;800&family=Space+Grotesk:wght@300;500;700&display=swap');

    /* === FOND PRINCIPAL : BEIGE CLINIQUE DONS === */
    [data-testid="stAppViewContainer"] {
        background-color: #FAF0E6; /* Couleur Linen / Beige clair médical */
        color: #001a2c; /* Texte sombre pour contraste */
        font-family: 'Space Grotesk', sans-serif;
    }

    /* === SUPPRESSION DES STRUCTURES RECTANGULAIRES === */
    [data-testid="column"], [data-testid="stVerticalBlock"], .stColumn > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        max-width: 1350px;
    }

    /* === EN-TÊTE EN GRADIENT VIF (BLEU & TEAL/VERT) === */
    .main-header {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #0056b3 0%, #00AFA3 100%); /* Bleu Médical vers Teal/Vert */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
        filter: drop-shadow(0 2px 5px rgba(0, 86, 179, 0.2));
    }
    
    .sub-text {
        color: #007BFF; /* Bleu Azure médical */
        text-align: center;
        font-size: 1em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 6px;
        margin-top: 0px;
    }

    /* === TITRES DE SECTIONS (BLEU) === */
    .section-title {
        color: #0056b3; /* Bleu Médical sombre */
        font-size: 1.5em;
        font-weight: 700;
        margin-bottom: 20px;
        border-left: 4px solid #00c853; /* Accent Vert vif */
        padding-left: 15px;
    }

    /* === LISIBILITÉ TOTALE DES CARREAUX (INPUTS SUR FOND CLAIR) === */
    /* Fond blanc pur avec bordure douce pour un look clinique propre */
    .stTextInput div div input, 
    .stNumberInput div div input, 
    .stSelectbox div div select {
        background-color: #ffffff !important; /* Fond Blanc pur */
        color: #001d3d !important; /* Texte Bleu Nuit très foncé - PARFAITEMENT LISIBLE */
        border: 2px solid #e0e0e0 !important; /* Bordure grise douce contre le beige */
        border-radius: 15px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
    }

    .stTextInput div div input:focus {
        border-color: #00c853 !important; /* Focus Vert vif */
        box-shadow: 0 0 10px rgba(0, 200, 83, 0.1) !important;
    }

    /* Labels sombres sur fond clair */
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #001a2c !important; /* Bleu Nuit labels */
        font-weight: 500 !important;
        margin-bottom: 8px !important;
    }

    /* === BOUTON DE DIAGNOSTIC VIF (VERT ACTION) === */
    div.stButton > button {
        background: linear-gradient(135deg, #00c853 0%, #00e676 100%) !important; /* Vert vif action médical */
        color: #ffffff !important; /* Texte blanc pour contraste sur vert */
        border-radius: 50px !important;
        font-weight: 800 !important;
        font-size: 1.1em !important;
        height: 55px;
        width: 100%;
        border: none !important;
        box-shadow: 0 5px 15px rgba(0, 200, 83, 0.2);
        transition: all 0.3s ease;
        text-transform: uppercase;
    }

    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 200, 83, 0.4);
        background: linear-gradient(135deg, #00e676 0%, #00c853 100%) !important;
    }

    /* === ZONE RÉSULTAT DOUCE (TEAL/VERT) === */
    .res-card {
        background: #E0F2F1; /* Teal très clair clinique */
        border: 2px solid #00c853; /* Bordure Vert vif */
        border-radius: 30px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }

    /* === FOOTER SOMBRE === */
    .footer-text {
        text-align: center;
        color: rgba(0, 0, 0, 0.5); /* Noir semi-transparent */
        margin-top: 50px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGIQUE TECHNIQUE (MODÈLE & PDF) - INCHANGÉE ---
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

def generate_medical_pdf(nom, prenom, age, gender, resultat, confiance, img, date_str):
    pdf = FPDF()
    pdf.add_page()
    # Le design du PDF reste sur fond blanc pour impression professionnelle
    pdf.set_font("Arial", 'B', 22)
    pdf.set_text_color(0, 50, 150) # Bleu médical professionnel
    pdf.cell(0, 25, "RAPPORT CLINIQUE - NEUROSCAN AI", 0, 1, 'C')
    
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Généré le : {date_str}", 0, 1, 'R')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. INFORMATIONS PATIENT", 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Nom complet : {nom} {prenom}", 0, 1)
    pdf.cell(0, 10, f"Age / Sexe : {age} ans / {gender}", 0, 1)
    
    pdf.ln(10)
    img.save("temp_report.png")
    pdf.image("temp_report.png", x=60, w=90)
    pdf.ln(10)
    
    pdf.set_fill_color(230, 245, 255)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 20, f"CONCLUSION : {resultat.upper()} ({confiance:.2f}%)", 1, 1, 'C', True)
    
    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFACE ---
algeria_tz = pytz.timezone('Africa/Algiers')
date_str = datetime.datetime.now(algeria_tz).strftime("%d/%m/%Y | %H:%M")

# Header avec icône
st.markdown('<p class="main-header">🧠 NeuroScan AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Plateforme de Diagnostic Cérébral Clinique</p>', unsafe_allow_html=True)

# Structure en colonnes (Zéro Rectangle visible)
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<p class="section-title">👤 Patient</p>', unsafe_allow_html=True)
    nom = st.text_input("NOM DE FAMILLE", placeholder="Ex: DOE").upper()
    prenom = st.text_input("PRÉNOM", placeholder="Ex: John").capitalize()
    age = st.number_input("ÂGE", 0, 120, 25)
    gender = st.selectbox("GENRE", ["Masculin", "Féminin", "Autre"])

with col2:
    st.markdown('<p class="section-title">🔬 Acquisition IRM</p>', unsafe_allow_html=True)
    up = st.file_uploader("Upload", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if up:
        image = Image.open(up).convert('RGB')
        # Cadre d'image arrondi doux
        st.image(image, use_container_width=True, caption="Scan chargé avec succès")
    else:
        st.info("Veuillez charger un fichier IRM pour analyse.")

with col3:
    st.markdown('<p class="section-title">⚡ Analyse IA</p>', unsafe_allow_html=True)
    if up and st.button("LANCER LE DIAGNOSTIC"):
        with st.spinner("Analyse des tissus neuronaux..."):
            model = load_neuro_model()
            img_array = np.array(image.resize((224, 224))) / 255.0
            preds = model.predict(np.expand_dims(img_array, axis=0))[0]
            classes = ['Non-Cérébral', 'Gliome', 'Méningiome', 'Pas de Tumeur', 'Pituitaire']
            res, conf = classes[np.argmax(preds)], float(np.max(preds)) * 100
            
            # Zone résultat Teal/Vert Clinique douée d'une ombre douce
            st.markdown(f'''
                <div class="res-card">
                    <p style="color:#0056b3; margin:0; font-weight:600;">CONCLUSION DÉTECTÉE</p>
                    <h2 style="color:#00c853; margin:0; font-size:2em; font-weight:800;">{res}</h2>
                    <p style="color:#001a2c; opacity:0.8; font-size:0.9em; margin-top:5px;">Confidence : {conf:.2f}%</p>
                </div>
            ''', unsafe_allow_html=True)
            
            # PDF Generation et Download Button
            pdf_bytes = generate_medical_pdf(nom, prenom, age, gender, res, conf, image, date_str)
            st.download_button("📥 TÉLÉCHARGER LE RAPPORT CLINIQUE PDF", pdf_bytes, f"NeuroScan_{nom}.pdf", "application/pdf")
            st.success("Analyse terminée. Rapport prêt.")
    else:
        st.write("Le système est prêt pour l'acquisition des données.")

    # Lien LinkedIn stylisé
    st.markdown(f'''
        <div style="text-align:right; margin-top:100px;">
            <a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank" 
               style="color:#0056b3; text-decoration:none; font-weight:700; border-bottom: 2px solid #00c853; padding-bottom:3px;">
               CONSULTER LE DEVLOPEUR ↗
            </a>
        </div>
    ''', unsafe_allow_html=True)

# Footer sombre contrasté sur beige
st.markdown(f'<p class="footer-text">NeuroScan v2.6 | Biomedical Engine | Bahlouli Fatna Romaisaa & Houbad Douaa | M1 EMB | Algeria Tz • {date_str}</p>', unsafe_allow_html=True)
