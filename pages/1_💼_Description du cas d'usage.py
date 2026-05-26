import os
import sys
import base64
import pandas as pd
import plotly.express as px
import streamlit as st

# Ajout du chemin racine au path pour pouvoir importer utils et config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import load_data

# Configuration de la page
st.set_page_config(page_title="Analyse Exploratoire", page_icon="📊", layout="wide")

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

# Titre de la page
st.title("💼 Intro - Description du cas pratique")

st.markdown(""" 

## Première partie : L'entreprise et le produit

### Hollywood Pharmaceuticals
""")

col1, col2 = st.columns([2,1])
with col1:
    st.image("docs/Hollywood_plant.png", caption="Bâtiment Hollywood Pharmaceuticals")
with col2:
    st.image("docs/IMG_0204_orig.jpg", caption="Gomme pharmaceutique")

st.markdown("""
**Une nouvelle forme galénique innovante : « la gomme à mâcher » !**

- L'entreprise Hollywood Pharmaceuticals souhaite déployer un système d'inspection automatique supporté par l'IA pour contrôler à 100% les unités produites en fin de production primaire.

- Les perspectives de croissance sont phénoménales : 33.0 Md$ à horizon 2032.

- L'entreprise est inspectée régulièrement et revendique un environnement GxP.

---

## Seconde partie : Le problème métier

### Les défis de l'inspection qualité

L'IA s'avère indispensable pour réaliser cette inspection car les gommes sont des **DIP (« Difficult to Inspect Products »)** en raison de variabilité des pigments. Des recettes articulées sur des règles métiers sont inopérantes.

L'entreprise est confrontée à **3 défauts majeurs** en cours de production :
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.image("docs/IMG_0608.jpg", caption="Bon / *Spot*")
with col2:
    st.image("docs/IMG_0523_aug2.jpg", caption="Défaut trou / *Hole*")
with col3:
    st.image("docs/IMG_0656_aug2.jpg", caption="Défaut éclat / *Scratch*")

st.markdown("""
En raison de l'image de marque, le lancement ne doit pas être entaché de problèmes qualité et de réclamations clients. Le procédé doit **limiter le risque** de libérer des unités non conformes.

---

## Troisième partie : L'avis du Data Scientist
""")

col1, col2 = st.columns([1,2])
with col1:
    st.image("docs/data_scientist.gif", caption="What !?!")  
with col2:        
    st.markdown("""
### Les premiers retours de l'équipe de Data Science

- La problématique est une **problématique de classification multiclasses**, l'algorithme va devoir pour chaque image prédire la probabilité d'appartenance aux classes :  
  « **Good** », « **Hole** », « **Scratch** », « **Spot** »

- Etant donné que les données d'entrées sont des images, il est nécessaire d'utiliser des modèles de type « **réseaux de neurone** » (Deep Learning).

- Des premiers essais ont été réalisés dans le cadre d'une FAT/SAT chez le fabriquant de machine, ce qui a permis de collecter environ **2500 images** et de construire un premier modèle de base en interne.
""")
    
st.markdown("""
---

## Quatrième partie : Votre rôle
""")

col1, col2 = st.columns([3,1])
with col1:
    st.markdown("""
### Votre mission en tant que membre de l'équipe AQ

Vous faites partie de l'équipe AQ du site de production. Votre mission est de :

✅ **Garantir la conformité réglementaire** du procédé : s'assurer que le système d'inspection IA respecte les exigences GxP et peut être validé.

✅ **Collaborer avec les Data Scientists** et la Production pour transformer les essais techniques en un système utilisable et validé industriellement. Participer à la **définition des critères d'acceptation**.

✅ Accompagner les équipes techniques dans le développement de la solution IA en **évaluant les risques qualité**. En conséquence, vous devrez **superviser le processus de validation** (plus spécifiquement sur la partie SI) et vous assurer que ce dernier est robuste.
""")
with col2:
    st.image("docs/we_want_you.png", caption="We need you !")

