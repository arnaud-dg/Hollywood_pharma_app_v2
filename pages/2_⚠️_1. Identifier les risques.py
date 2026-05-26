import os
import sys
import copy
import base64
import yaml
import streamlit as st

# Ajout du chemin racine au path pour importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.kanban_component import kanban_board

# Configuration de la page
st.set_page_config(page_title="Identifier les risques", page_icon="⚠️", layout="wide")

# Charger le logo en base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
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
except:
    pass

# =====================================================
# Données initiales des cartes Kanban (chargées depuis YAML)
# =====================================================
def load_initial_cards():
    yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cards.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["workspace_usage"] = []
    data["workspace_interaction"] = []
    data["workspace_tracabilite"] = []
    data["workspace_infra"] = []
    return data

INITIAL_CARDS = load_initial_cards()

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

# Bouton Reset dans la sidebar
if st.sidebar.button("🔄 Reset", use_container_width=True):
    st.session_state.kanban_state = copy.deepcopy(INITIAL_CARDS)
    st.session_state.kanban_reset_counter = st.session_state.get("kanban_reset_counter", 0) + 1
    st.rerun()

if st.session_state.group_choice == "":
    st.warning("⚠️ Veuillez sélectionner un groupe dans la barre latérale pour continuer.")
    st.stop()

# Chargement des styles CSS personnalisés
try:
    with open("assets/css/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Titre de la page
st.title("⚠️ 1. Evaluer la criticité et identifier les risques projets")
st.markdown(
    "Organisez vos **URS**, **Risques** et **Critères d'acceptation** "
    "en les glissant dans le workspace ci-dessous."
)

# =====================================================
# Session state : persistance de l'état des cartes
# =====================================================
if "kanban_state" not in st.session_state:
    st.session_state.kanban_state = copy.deepcopy(INITIAL_CARDS)
if "kanban_reset_counter" not in st.session_state:
    st.session_state.kanban_reset_counter = 0


# =====================================================
# Rendu du composant Kanban
# - cards_state : état courant des 4 zones
# - reset_counter : incrémenté à chaque Reset pour forcer
#   le JS à re-render depuis l'état Python
# - default=None : valeur retournée avant que le JS n'envoie
#   son premier setComponentValue
# =====================================================
new_state = kanban_board(
    cards_state=st.session_state.kanban_state,
    reset_counter=st.session_state.kanban_reset_counter,
    key="kanban_board"
)

# Mise à jour du session_state quand le composant renvoie un nouvel état
if new_state is not None:
    st.session_state.kanban_state = new_state

