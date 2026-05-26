import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
import base64
from utils import afficher_logo_sidebar

# Configuration de la page
st.set_page_config(page_title="Data Analytics Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded",)

# Charger le logo en base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_base64_of_bin_file("assets/images/a3p.png")

st.markdown(
    f"""
    <style>
        [data-testid="stSidebarNav"]::before {{
            content: "";
            display: block;
            margin: 0 auto 20px auto;
            height: 200px;
            width: 250px;
            background-image: url("data:image/png;base64,{logo_base64}");
            background-repeat: no-repeat;
            background-size: contain;
            background-position: center;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Initialisation et gestion du groupe
if "group_choice" not in st.session_state:
    st.session_state.group_choice = ""

# Récupérer l'index actuel
current_index = 0
options = ["", "Grp1", "Grp2", "Grp3", "Grp4", "Grp5", "Grp6", "Grp7"]
if st.session_state.group_choice in options:
    current_index = options.index(st.session_state.group_choice)

# Selectbox qui met à jour automatiquement
st.session_state.group_choice = st.sidebar.selectbox(
    "Sélectionner un groupe",
    options=options,
    index=current_index,
    help="Veuillez choisir un groupe pour afficher le contenu des pages."
)
st.sidebar.markdown("---")

if st.session_state.group_choice == "":
    st.warning("⚠️ Veuillez sélectionner un groupe dans la barre latérale pour continuer.")
    st.stop()

# Chargement des styles CSS personnalisés
with open("assets/css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Titre principal
st.title("🤖​ Projet Hollywood Pharma")
st.markdown("### Bienvenue sur la plateforme projet Hollywood Pharma")

