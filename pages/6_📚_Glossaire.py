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
st.set_page_config(page_title="Glossaire", page_icon="📚", layout="wide")

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
st.title("📚 Glossaire")
st.markdown("Quelques définitions pour mieux comprendre les termes utilisés dans ce projet.")

col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.write("") 
with col2:
    st.image("docs/Glossary.png")
with col3:
    st.write("")


st.markdown("---")
st.header("Les concepts généraux")
st.markdown("""
#### 📊 Apprentissage automatique (*Machine Learning*)  
Approche permettant aux systèmes informatiques d'apprendre à partir de données, sans être explicitement programmés pour chaque cas.  
Le système améliore ses performances au fur et à mesure qu'il traite davantage d'exemples.  

#### 🔍 Apprentissage non supervisé (*Unsupervised Learning*)  
Méthode d'entraînement où les données ne sont pas étiquetées. Le modèle doit découvrir par lui-même des structures ou des regroupements dans les données, sans qu'on lui indique les bonnes réponses.  

#### 🤖 Apprentissage supervisé (*Supervised Learning*)  
Méthode d'entraînement où chaque exemple fourni au modèle est accompagné de sa réponse correcte (*étiquette*).  
Le modèle apprend en comparant ses prédictions aux bonnes réponses et ajuste progressivement son fonctionnement pour réduire ses erreurs.  

#### 🎯 Classification  
Tâche consistant à attribuer une catégorie (ou classe) à chaque élément analysé.  
Exemple : déterminer si un comprimé est *"Conforme"* ou *"Défectueux"*, ou identifier le type de défaut présent.  

#### 💡 Explicabilité (*Explainability*)  
Capacité à comprendre et expliquer comment un modèle d'IA prend ses décisions de manière compréhensible pour les humains.  
Permet de savoir pourquoi le modèle a classé une image d'une certaine manière. 

#### 🧠 Intelligence Artificielle (*IA / AI*)  
Ensemble de technologies permettant aux machines d'effectuer des tâches qui nécessitent habituellement l'intelligence humaine,  
comme reconnaître des objets dans des images ou prendre des décisions.  

#### 🔢 Modèle  
Représentation mathématique créée par l'apprentissage automatique,  
capable de faire des prédictions ou des classifications sur de nouvelles données après avoir été entraîné sur des exemples.  

#### 📈 Régression  
Tâche consistant à prédire une valeur numérique continue plutôt qu'une catégorie.  
Exemple : estimer le prix d'une maison en fonction de ses caractéristiques ou prédire une température.  

#### 🌐 Réseau de neurones (*Neural Network*)  
Architecture informatique inspirée du fonctionnement du cerveau humain,  
composée de multiples couches d'unités de calcul interconnectées qui traitent l'information progressivement.  

#### 🖼️ Réseau de neurones convolutif (*Convolutional Neural Network - CNN*)  
Type de réseau de neurones spécialement conçu pour traiter des images.  
Il analyse les images par zones pour en extraire automatiquement les caractéristiques importantes,  
ce qui le rend très efficace pour la reconnaissance visuelle.  

#### 🔎 Transparence  
Possibilité de comprendre le fonctionnement interne d'un modèle et la logique derrière ses décisions.  
Important pour établir la confiance et la responsabilité dans les applications critiques.  

#### 👁️ Vision par ordinateur (*Computer Vision*)  
Domaine permettant aux ordinateurs de comprendre et d'interpréter des images et des vidéos, comme le ferait un œil humain.  
Utilise des techniques d'apprentissage automatique pour analyser le contenu visuel.  
""")

st.markdown("---")
st.header("Les données")
st.markdown("""      
#### 🔲 Données structurées  
Données organisées dans un format prédéfini et ordonné, comme des tableaux Excel ou des bases de données.  
Faciles à analyser car elles suivent une organisation claire et régulière.  

#### 🌀 Données non structurées  
Données sans format prédéfini, comme les images, vidéos ou textes libres.  
Nécessitent des techniques d'analyse plus sophistiquées pour en extraire des informations utiles.  

#### 🏷️ Étiquette (*Label*)  
Information associée à un exemple de données, utilisée pour l'entraînement des modèles d'apprentissage supervisé.  
Les étiquettes permettent au modèle de comprendre la réponse correcte à chaque exemple.  

#### 🎓 Jeu d'entraînement (*Training Set*)  
Ensemble de données utilisé pour apprendre au modèle à classifier correctement les exemples.  
Le modèle analyse ces données de nombreuses fois pour comprendre les caractéristiques de chaque catégorie.  

#### ✔️ Jeu de test (*Test Set*)  
Ensemble de données jamais utilisé durant l'entraînement, servant uniquement à évaluer les performances finales du modèle sur des données complètement nouvelles.  
Simule l'utilisation en conditions réelles.  

#### ⚖️ Jeu de validation / tuning (*Validation Set*)  
Ensemble de données utilisé pour ajuster les réglages du modèle (*hyperparamètres*) et éviter le surapprentissage.  
Permet de vérifier que le modèle ne mémorise pas seulement les exemples d'entraînement.  

#### 🏷️ Labellisation / Annotation (*Labeling*)  
Processus consistant à attribuer une étiquette ou une catégorie à chaque exemple de données.  
Exemple : identifier manuellement si chaque image de comprimé montre un produit *"Conforme"* ou *"Défectueux"*.  

#### ✂️ Partitionnement des jeux de données (*Data Split*)  
Division des données disponibles en plusieurs ensembles distincts : un pour entraîner le modèle, un pour ajuster ses réglages, et un pour évaluer ses performances finales.  
""")


st.markdown("---")
st.header("L'entraînement")
st.markdown("""
#### ⚖️ Balance de pénalités (Class Weights)  
Réglage permettant de donner plus d'importance à certaines erreurs lors de l'entraînement.  
Utile pour éviter par exemple de laisser passer des défauts critiques, même si cela augmente les faux rejets de produits conformes.  

#### 📦 Taille de lot (Batch size)  
Nombre d'exemples traités simultanément avant d'ajuster les réglages du modèle.  
Un lot plus grand stabilise l'apprentissage mais nécessite plus de mémoire, tandis qu'un lot plus petit permet un apprentissage plus rapide mais plus variable.  

#### 📅 Époques (Epochs)  
Nombre de fois où le modèle passe en revue l'ensemble complet des données d'entraînement.  
Exemple : avec 50 époques, le modèle analysera 50 fois toutes les images d'entraînement.  

#### 📉 Fonction de perte (Loss Function)  
Mesure quantifiant l'écart entre les prédictions du modèle et les bonnes réponses.  
Guide l'optimisation en indiquant dans quelle direction ajuster le modèle pour réduire les erreurs.  

#### ⚡ Learning Rate (Taux d'apprentissage)  
Paramètre contrôlant la vitesse à laquelle le modèle apprend.  
Une valeur trop élevée peut faire "louper" la meilleure solution, tandis qu'une valeur trop faible ralentit considérablement l'apprentissage.  

#### 📉 Sous-apprentissage (Underfitting)  
Situation où le modèle est trop simple pour capturer la complexité des données et ne parvient même pas à bien classifier les exemples d'entraînement.  
Performances médiocres sur tous les ensembles de données.  

#### 📈 Surapprentissage (Overfitting)  
Situation où le modèle mémorise trop précisément les exemples d'entraînement et perd sa capacité à bien classifier de nouvelles données.  
Comme un élève qui apprendrait par cœur sans comprendre.
            
#### 🎯 Accuracy (Précision globale)  
Pourcentage de prédictions correctes sur l'ensemble des échantillons testés.  
Exemple : si l'accuracy est de 85%, le modèle classe correctement 85% des comprimés.  

#### 🗂️ Confusion Matrix (Matrice de confusion)  
Tableau résumant les performances du modèle en affichant le nombre de prédictions correctes et incorrectes pour chaque catégorie.  
Permet d'identifier précisément les types d'erreurs commises.  

#### 📊 Courbe d'apprentissage (Learning Curve)  
Graphique montrant l'évolution des performances du modèle au fil des époques ou en fonction de la quantité de données d'entraînement.  
Aide à diagnostiquer le surapprentissage ou le sous-apprentissage.  

#### 🔍 Precision (Précision par classe)  
Pour une catégorie donnée, pourcentage de prédictions correctes parmi toutes les prédictions faites pour cette catégorie.  
Répond à la question : *"Quand le modèle dit qu'il y a un défaut, a-t-il raison ?"*  

#### 📡 Recall (Rappel / Sensibilité)  
Pour une catégorie donnée, pourcentage d'exemples correctement identifiés parmi tous ceux qui appartiennent réellement à cette catégorie.  
Répond à la question : *"Le modèle a-t-il trouvé tous les défauts réellement présents ?"* 
""")

st.markdown("---")
st.header("Le monitoring")
st.markdown("""
#### 🌊 Data Drift (Dérive des données)  
Phénomène où la distribution statistique des données d'entrée change au fil du temps, par rapport à celles utilisées lors de l'entraînement du modèle.  
Ce décalage peut réduire les performances du modèle, car il ne rencontre plus les mêmes types de données qu'au moment de son apprentissage.  
Exemple : des conditions de production, des équipements ou des comportements utilisateurs qui évoluent.  
            
#### ⚙️ Dérive du modèle (Model Drift)
Diminution progressive des performances d'un modèle au fil du temps, causée par des changements dans la relation entre les données d'entrée et les résultats attendus.  
Nécessite un réentraînement du modèle.  

#### 📊 Suivi de tendance (Monitoring) 
Suivi continu des performances d'un modèle en production pour détecter rapidement toute dégradation.  
Permet de s'assurer que le modèle reste efficace au fil du temps et des changements de conditions.  
""")
