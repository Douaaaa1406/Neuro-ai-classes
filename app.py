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

# --- 2. DESIGN MODERNE ET PROFESSIONNEL ---
st.markdown("""
<style>
    /* === FONTS IMPORT === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Poppins:wght@400;600;700&display=swap');

    /* === FOND PRINCIPAL AVEC GRADIENT === */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }

    /* === SUPPRESSION DES ÉLÉMENTS BLANCS === */
    [data-testid="column"], [data-testid="stVerticalBlock"], .stColumn > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* === OPTIMISATION DE L'ESPACE === */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
        max-width: 1400px;
    }

    /* === EN-TÊTE PRINCIPAL === */
    .main-header {
        font-family: 'Poppins', sans-serif;
        color: #ffffff;
        font-size: 3.5em;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
        text-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        letter-spacing: -1px;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .sub-text {
        color: #ffd700;
        text-align: center;
        font-size: 1.2em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 4px;
        margin-top: 5px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        animation: fadeIn 1s ease-out;
    }

    .live-date {
        text-align: center;
        color: rgba(255, 255, 255, 0.9);
        font-family: 'Inter', monospace;
        font-size: 1em;
        margin-bottom: 30px;
        margin-top: 10px;
        font-weight: 300;
        animation: fadeIn 1.2s ease-out;
    }

    /* === CARTES MODERNES AVEC GLASSMORPHISM === */
    .art-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        min-height: 540px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        animation: slideUp 0.6s ease-out;
    }

    .art-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
        background: rgba(255, 255, 255, 0.2);
    }

    .column-title {
        font-family: 'Poppins', sans-serif;
        color: #ffffff;
        border-bottom: 3px solid #ffd700;
        padding-bottom: 12px;
        font-size: 1.5em;
        font-weight: 700;
        margin-bottom: 25px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        letter-spacing: 0.5px;
    }

    /* === INPUTS STYLISÉS === */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        padding: 12px 15px !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        background: rgba(255, 255, 255, 0.25) !important;
        border: 1px solid #ffd700 !important;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.3) !important;
    }

    .stTextInput > label,
    .stNumberInput > label,
    .stSelectbox > label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.95em !important;
        margin-bottom: 8px !important;
    }

    /* === BOUTON PRINCIPAL === */
    div.stButton > button {
        background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%) !important;
        color: #4a148c !important;
        border-radius: 12px !important;
        font-family: 'Poppins', sans-serif;
        font-weight: 700 !important;
        font-size: 1.1em !important;
        width: 100%;
        height: 55px;
        border: none !important;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.4);
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 215, 0, 0.6);
        background: linear-gradient(135deg, #ffed4e 0%, #ffd700 100%) !important;
    }

    div.stButton > button:active {
        transform: translateY(0px);
    }

    /* === BOUTON DE TÉLÉCHARGEMENT === */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00c853 0%, #00e676 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        font-family: 'Poppins', sans-serif;
        font-weight: 600 !important;
        width: 100%;
        height: 50px;
        border: none !important;
        box-shadow: 0 4px 15px rgba(0, 200, 83, 0.4);
        transition: all 0.3s ease !important;
    }

    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 200, 83, 0.6);
    }

    /* === FILE UPLOADER === */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #ffd700;
        background: rgba(255, 255, 255, 0.15);
    }

    /* === IMAGES === */
    [data-testid="stImage"] {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        margin-top: 15px;
    }

    /* === RÉSULTATS === */
    .diagnostic-result {
        background: rgba(255, 215, 0, 0.15);
        border-left: 5px solid #ffd700;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        animation: slideInRight 0.5s ease-out;
    }

    /* === FOOTER === */
    .footer-text {
        text-align: center;
        color: rgba(255, 255, 255, 0.8);
        font-family: 'Inter', serif;
        margin-top: 30px;
        font-size: 0.95em;
        font-weight: 300;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    .linkedin-link {
        color: #ffd700;
        font-weight: 600;
        text-decoration: none;
        transition: all 0.3s ease;
    }

    .linkedin-link:hover {
        color: #ffed4e;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }

    /* === ANIMATIONS === */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }

    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* === RESPONSIVE === */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5em;
        }
        .sub-text {
            font-size: 1em;
            letter-spacing: 2px;
        }
        .art-card {
            min-height: auto;
            padding: 20px;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGIQUE TECHNIQUE (INCHANGÉE) ---
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
    pdf.set_font("Times", 'B', 20)
    pdf.set_text_color(128, 0, 32)
    pdf.cell(0, 20, "RAPPORT CLINIQUE - NEUROSCAN AI", 0, 1, 'C')
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Times", 'B', 10)
    pdf.cell(0, 10, f"Date : {date_str}", 0, 1, 'R')
    pdf.ln(5)
    pdf.set_font("Times", 'B', 12)
    pdf.cell(0, 10, "1. INFORMATIONS PATIENT", 0, 1)
    pdf.set_font("Times", '', 11)
    data = [["Nom/Prenom", f"{nom} {prenom}"], ["Age/Genre", f"{age}/{gender}"], ["Analyste", "(M1 EMB)"], ["Heure", date_str]]
    for row in data:
        pdf.cell(50, 10, row[0], 1); pdf.cell(100, 10, row[1], 1); pdf.ln()
    pdf.ln(10)
    img.save("temp.png")
    pdf.image("temp.png", x=65, w=80)
    pdf.ln(10)
    pdf.set_font("Times", 'B', 15); pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 15, f"RESULTAT : {resultat.upper()}", 1, 1, 'C', True)
    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFACE UTILISATEUR ---
algeria_tz = pytz.timezone('Africa/Algiers')
date_str = datetime.datetime.now(algeria_tz).strftime("%d/%m/%Y | %H:%M:%S")

# En-tête
st.markdown('<p class="main-header">🧠 NeuroScan AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Biomedical Engineering • Algiers</p>', unsafe_allow_html=True)
st.markdown(f'<p class="live-date">🕒 {date_str}</p>', unsafe_allow_html=True)

# Colonnes principales
col1, col2, col3 = st.columns(3, gap="large")

# --- COLONNE 1 : DONNÉES PATIENT ---
with col1:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">👤 Données Patient</div>', unsafe_allow_html=True)
    nom = st.text_input("NOM DE FAMILLE", key="nom").upper()
    prenom = st.text_input("PRÉNOM", key="prenom").capitalize()
    age = st.number_input("ÂGE", min_value=0, max_value=120, value=30, key="age")
    gender = st.selectbox("GENRE", ["Masculin", "Féminin"], key="gender")
    st.markdown('</div>', unsafe_allow_html=True)

# --- COLONNE 2 : IRM SCAN ---
with col2:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">🔬 IRM Scan</div>', unsafe_allow_html=True)
    up = st.file_uploader("Télécharger un scan IRM", type=["jpg", "png", "jpeg"], label_visibility="collapsed", key="uploader")
    if up:
        image = Image.open(up).convert('RGB')
        st.image(image, use_container_width=True)
    else:
        st.info("📤 En attente d'un fichier IRM...")
    st.markdown('</div>', unsafe_allow_html=True)

# --- COLONNE 3 : DIAGNOSTIC ---
with col3:
    st.markdown('<div class="art-card">', unsafe_allow_html=True)
    st.markdown('<div class="column-title">⚕️ Diagnostic</div>', unsafe_allow_html=True)
    
    if up and st.button("🚀 LANCER L'ANALYSE", key="analyze"):
        with st.spinner("🔄 Analyse en cours..."):
            model = load_neuro_model()
            img_array = np.array(image.resize((224, 224))) / 255.0
            preds = model.predict(np.expand_dims(img_array, axis=0))[0]
            classes = ['Non-Cérébral', 'Gliome', 'Méningiome', 'Pas de Tumeur', 'Pituitaire']
            res, conf = classes[np.argmax(preds)], float(np.max(preds)) * 100
            
            st.markdown('<div class="diagnostic-result">', unsafe_allow_html=True)
            st.markdown(f"### 📊 Résultat : **{res}**")
            st.markdown(f"### 🎯 Confiance : **{conf:.2f}%**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            pdf = generate_medical_pdf(nom, prenom, age, gender, res, conf, image, date_str)
            st.download_button(
                "📥 TÉLÉCHARGER LE RAPPORT", 
                pdf, 
                f"Report_{nom}_{prenom}.pdf", 
                "application/pdf",
                key="download"
            )
    else:
        st.info("⏳ En attente d'acquisition...")
    
    st.markdown(f'''
        <div style="text-align:right; margin-top:120px;">
            <a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" 
               target="_blank" 
               class="linkedin-link">
               💼 LinkedIn
            </a>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('''
    <p class="footer-text">
        Développé avec ❤️ par <strong>Bahlouli Fatna Romaisaa</strong> & <strong>Houbad Douaa</strong> | M1 EMB | 2026
    </p>
''', unsafe_allow_html=True)
