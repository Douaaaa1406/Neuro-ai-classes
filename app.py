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

# --- 2. DESIGN CSS PROFESSIONNEL ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .main-header {
        color: #1E3A5F;
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 3em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .sub-header {
        color: #4A90E2;
        text-align: center;
        font-size: 1.3em;
        margin-bottom: 30px;
        font-weight: 300;
    }
    .info-bar {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #d1d9e6;
        text-align: center;
        color: #1E3A5F;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .result-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        border-top: 10px solid #1E3A5F;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. CHARGEMENT DU MODÈLE V6 ---
@st.cache_resource
def load_neuro_model():
    model_path = 'brain_tumor_model_v6_final.keras'
    # Remplace cet ID par celui de ton nouveau fichier v6 sur Google Drive
    file_id = '1QRVvhNHSx7qgw0GIDrRLsuX09uItsXM2' 
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_path):
        with st.spinner("Chargement de l'intelligence artificielle..."):
            try:
                gdown.download(url, model_path, quiet=False)
            except:
                st.error("Erreur de connexion au Drive.")
    
    # Architecture 5 classes
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

# --- 4. ENTÊTE & HORLOGE ALGERIE ---
col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
with col_logo2:
    st.markdown('<div style="text-align:center; font-size:80px;">🧠</div>', unsafe_allow_html=True)
    st.markdown('<p class="main-header">NeuroScan AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Diagnostic Biomédical Haute Précision elaboré par Houbad Douaa </p>', unsafe_allow_html=True)

algeria_tz = pytz.timezone('Africa/Algiers')
now = datetime.datetime.now(algeria_tz)
st.markdown(f"""
    <div class="info-bar">
        📍 Centre de Radiologie | Algérie | 📅 {now.strftime("%d/%m/%Y")} | ⌚ {now.strftime("%H:%M:%S")}
    </div>
""", unsafe_allow_html=True)

# --- 5. INTERFACE UTILISATEUR ---
st.write("###")
col_form, col_img = st.columns([1, 1])

with col_form:
    st.subheader("👤 Informations Patient")
    nom = st.text_input("Nom").upper()
    prenom = st.text_input("Prénom").capitalize()
    date_n = st.date_input("Date de naissance", value=datetime.date(1990, 1, 1))
    lieu_n = st.text_input("Lieu de naissance")

with col_img:
    st.subheader("🖼️ Acquisition IRM")
    uploaded_file = st.file_uploader("Charger le cliché (JPG/PNG)", type=["jpg", "png", "jpeg"])

# --- 6. ANALYSE ET RÉSULTATS ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, width=380, caption="Scanner prêt pour analyse")

    if st.button("🚀 LANCER L'ANALYSE NEURONALE"):
        with st.spinner("Analyse spectrale en cours..."):
            model = load_neuro_model()
            
            # Prétraitement (Augmentation contraste pour les Gliomes)
            enhancer = ImageEnhance.Contrast(image)
            img_processed = enhancer.enhance(1.15).resize((224, 224))
            img_array = np.array(img_processed) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prédiction
            preds = model.predict(img_array)[0]
            # Ordre strict : 0:Unknown, 1:glioma, 2:meningioma, 3:notumor, 4:pituitary
            classes = ['Image Non-Cérébrale', 'Gliome', 'Méningiome', 'Pas de tumeur', 'Pituitaire']
            
            idx = np.argmax(preds)
            resultat = classes[idx]
            confiance = preds[idx] * 100

        # --- LOGIQUE DE REJET ET AFFICHAGE ---
        if resultat == 'Image Non-Cérébrale' or confiance < 70:
            st.error("❌ ERREUR DE VALIDATION ANATOMIQUE")
            st.warning("L'image fournie n'a pas été identifiée comme une IRM cérébrale valide. Veuillez vérifier le cliché.")
        else:
            st.markdown(f"""
                <div class="result-card">
                    <h2 style="color: #1E3A5F; margin-top:0;">Diagnostic : {resultat}</h2>
                    <h3 style="color: #4A90E2;">Indice de confiance : {confiance:.2f}%</h3>
                    <p style="color: #555;">Analyse assistée par Deep Learning (V6.0). 
                    Ce résultat doit être confirmé par un expert médical.</p>
                </div>
            """, unsafe_allow_html=True)

            # --- 7. GÉNÉRATION DU RAPPORT PDF ---
            pdf = FPDF()
            pdf.add_page()
            
            # Bandeau de titre
            pdf.set_fill_color(30, 58, 95)
            pdf.rect(0, 0, 210, 45, 'F')
            pdf.set_text_color(255, 255, 255)
            pdf.set_font('Arial', 'B', 24)
            pdf.cell(0, 25, 'RAPPORT DE DIAGNOSTIC IA', 0, 1, 'C')
            
            # Corps du rapport
            pdf.set_y(55)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, f"Date d'examen : {now.strftime('%d/%m/%Y %H:%M')}", 0, 1, 'R')
            
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, "I. IDENTIFICATION DU PATIENT", 1, 1, 'L')
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 10, f"Patient : {nom} {prenom}", 0, 1)
            pdf.cell(0, 10, f"Né(e) le : {date_n} à {lieu_n}", 0, 1)
            
            pdf.ln(10)
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, "II. RÉSULTATS DE L'ANALYSE", 1, 1, 'L')
            pdf.set_font('Arial', 'B', 16)
            pdf.set_text_color(200, 0, 0) if resultat != 'Pas de tumeur' else pdf.set_text_color(0, 128, 0)
            pdf.cell(0, 20, f"CONCLUSION : {resultat.upper()}", 0, 1, 'C')
            
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('Arial', 'I', 11)
            pdf.cell(0, 10, f"Degré de certitude algorithmique : {confiance:.2f}%", 0, 1, 'C')
            
            # Insertion de l'image
            image.save("temp_scan.jpg")
            pdf.image("temp_scan.jpg", x=65, w=80)
            
            # Footer
            pdf.set_y(-40)
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 10, f"Ingénieur Responsable : HOUBAD DOUAA", 0, 1, 'C')
            pdf.set_font('Arial', 'I', 8)
            pdf.cell(0, 5, "Document généré par NeuroScan AI - Ne remplace pas un avis médical.", 0, 1, 'C')

            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.write("###")
            st.download_button(
                label="📥 TÉLÉCHARGER LE COMPTE-RENDU PDF",
                data=pdf_bytes,
                file_name=f"NeuroScan_{nom}.pdf",
                mime="application/pdf"
            )

# --- PIED DE PAGE ---
st.markdown("---")
st.markdown(f"""
    <div style="text-align: center; color: #7f8c8d;">
        <b>NeuroScan AI v6.1</b> | Algérie | Développé par <b>HOUBAD Douaa</b>
    </div>
""", unsafe_allow_html=True)
