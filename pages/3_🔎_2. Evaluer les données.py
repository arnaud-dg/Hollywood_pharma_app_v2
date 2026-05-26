import os
import sys
import base64
import hashlib
import random
from io import BytesIO
from PIL import Image
from PIL.ExifTags import TAGS
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit.components.v1 import html
import base64
import re

# Configuration de la page
st.set_page_config(page_title="Dataset", page_icon="📊", layout="wide")

# Charger le logo en base64 (local)
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

try:
    with open("assets/css/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Titre de la page
st.title("🔎 2. Evaluer la qualité d'un jeu de données")
st.markdown("**Objectifs pédagogiques**")
st.markdown("""
- Comprendre et réaliser **<span style="color:#7345FF"> l'importance de la qualité des données</span>**.
- Acquérir une **<span style="color:#7345FF">démarche structurée d'évaluation des risques</span>**.
""", unsafe_allow_html=True)

st.markdown("**Votre mission**")
st.markdown("""
- Les data scientists de votre équipe vous ont préparé un rapide outil pour explorer les caractéristiques du jeu de données.
- La base de donnée servant à entraîner notre IA est constituée d'environ 2.500 images
(Un échantillon de 100 images a été sélectionné pour simplifier la démonstration). 
Vous devez vous assurer que les données qui vont être utilisées pour l'entraînement et le test sont appropriées
- Gardez **<span style="color:#7345FF"> un oeil critique et utilisez votre bon sens</span>** pour parcourir le panel de 100 images à la recherche 
**<span style="color:#7345FF">d'anomalies</span>**, de **<span style="color:#7345FF">biais</span>** ou **<span style="color:#7345FF">d'incohérences</span>** qui pourraient impacter directement ou indirectement 
la performance de l'IA.
""", unsafe_allow_html=True)

# Fonctions utilitaires
def image_to_base64_from_bytes(img_bytes):
    """Convertit des bytes d'image en base64 pour l'affichage HTML"""
    try:
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        st.error(f"Erreur lors de la conversion de l'image: {e}")
        return None

def extract_image_metadata(img_bytes):
    """Extrait les métadonnées d'une image depuis des bytes"""
    try:
        # Convertir bytes en array numpy
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, None, None
        
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        intensity = round(np.mean(gray), 2)
        return w, h, intensity
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des métadonnées: {e}")
        return None, None, None

def get_exif_metadata(img_bytes):
    """Extrait les métadonnées EXIF d'une image"""
    try:
        image = Image.open(BytesIO(img_bytes))
        exif_data = image._getexif()
        metadata = {}
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                metadata[tag] = value
        return {
            "Auteur": metadata.get("Artist", "NA"),
            "Date EXIF": metadata.get("DateTimeOriginal", "NA")
        }
    except Exception:
        return {
            "Auteur": "NA",
            "Date EXIF": "NA"
        }

def get_file_hash(content_bytes, algo='sha256'):
    """Calcule le hash d'un fichier depuis bytes"""
    try:
        hash_func = hashlib.sha256() if algo == 'sha256' else hashlib.md5()
        hash_func.update(content_bytes)
        return hash_func.hexdigest()
    except Exception as e:
        st.error(f"Erreur lors du calcul du hash: {e}")
        return "NA"

@st.cache_data
def load_image_dataset_from_local(data_path="data/dataset_analyze_save"):
    """Charge et analyse le jeu de données d'images depuis le système de fichiers local"""
    
    image_data = []
    
    if not os.path.exists(data_path):
        st.error(f"Le dossier {data_path} n'existe pas")
        return None
    
    with st.spinner("Chargement et analyse des images..."):
        # Lister tous les fichiers images
        image_files = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            st.error(f"Aucune image trouvée dans {data_path}")
            return None
        
        progress_bar = st.progress(0)
        total_images = len(image_files)
        
        for idx, file_path in enumerate(image_files):
            progress_bar.progress((idx + 1) / total_images)
            
            # Extraire les informations du chemin
            relative_path = os.path.relpath(file_path, data_path)
            parts = relative_path.split(os.sep)
            
            label = parts[0] if len(parts) > 0 else "NA"
            defect_type = parts[1] if label == "Defect" and len(parts) > 1 else "NA"
            file_name = os.path.basename(file_path)
            
            # Lire l'image
            try:
                with open(file_path, 'rb') as f:
                    img_bytes = f.read()
            except Exception as e:
                st.warning(f"Impossible de lire {file_path}: {e}")
                continue
            
            # Métadonnées
            w, h, intensity = extract_image_metadata(img_bytes)
            img_base64 = image_to_base64_from_bytes(img_bytes)
            img_tag = f'<img src="data:image/jpeg;base64,{img_base64}" width="150" style="border-radius: 5px;"/>' if img_base64 else "Image non disponible"
            
            metadata = get_exif_metadata(img_bytes)
            file_hash = get_file_hash(img_bytes)
            
            image_data.append({
                "Chemin": file_path,
                "Nom fichier": file_name,
                "Image": img_tag,
                "Nature": label,
                "Type de défaut": defect_type,
                "Largeur (px)": w,
                "Hauteur (px)": h,
                "Luminosité moyenne": intensity,
                "Auteur": metadata["Auteur"],
                "Date de création": metadata["Date EXIF"],
                "Hash (SHA-256)": file_hash
            })
        
        progress_bar.empty()
    
    if not image_data:
        st.error("Aucune donnée d'image n'a pu être extraite.")
        return None
    
    # Création du DataFrame
    df = pd.DataFrame(image_data)
    
    # Renommer la colonne
    df.rename(columns={"Nom fichier": "Index"}, inplace=True)
    
    # Sélectionner 3 indices aléatoires pour garder des valeurs NA
    indices_exceptions = random.sample(range(len(df)), min(3, len(df)))
    
    # Remplacer les NA sauf pour les exceptions
    for i in range(len(df)):
        if i not in indices_exceptions:
            if df.loc[i, "Auteur"] == "NA":
                df.loc[i, "Auteur"] = "JPL"
            if df.loc[i, "Date de création"] == "NA":
                df.loc[i, "Date de création"] = "01/02/2025"
    
    # Créer une colonne catégorie
    df["Catégorie"] = df.apply(
        lambda row: row["Nature"] if row["Nature"] == "Good" else f'Defects - {row["Type de défaut"]}',
        axis=1
    )
    
    return df

def create_interactive_table(df):
    """Crée un tableau interactif avec itables"""
    import tempfile
    import os
    
    # Sélectionner les colonnes à afficher
    display_columns = [
        "Index", "Image", "Nature", "Type de défaut", "Largeur (px)", 
        "Hauteur (px)", "Luminosité moyenne", "Auteur", "Date de création", "Hash (SHA-256)"
    ]
    
    df_display = df[display_columns].copy()
    
    # Créer un fichier HTML temporaire avec itables
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        # En-tête HTML avec les CDN nécessaires
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Tableau Interactif</title>
            <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
            <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
            <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table.dataTable { width: 100% !important; }
                .dataTables_wrapper { width: 100%; }
                img { 
                    max-width: 120px; 
                    height: auto; 
                    border-radius: 4px; 
                    transition: transform 0.3s ease, z-index 0.3s ease; 
                    cursor: zoom-in;
                }
                img:hover { 
                    transform: scale(4);
                    transform-origin: top left;
                    z-index: 9999;
                    position: relative; 
                }
                .dataTables_filter input { width: 300px; }
            </style>
        </head>
        <body>
        """
        
        # Convertir le DataFrame en HTML
        table_html = df_display.to_html(table_id="interactive_table", escape=False, index=False)
        
        # Script JavaScript pour initialiser DataTables
        script = """
        <script>
        $(document).ready(function() {
            $('#interactive_table').DataTable({
                "pageLength": 100,
                "lengthMenu": [25, 50, 100],
                "scrollX": true,
                "scrollY": "800px",
                "scrollCollapse": true,
                "columnDefs": [
                    { "width": "80px", "targets": 0 },
                    { "width": "140px", "targets": 1 },
                    { "width": "100px", "targets": [2, 3] },
                    { "width": "80px", "targets": [4, 5, 6] },
                    { "width": "100px", "targets": [7, 8] },
                    { "width": "200px", "targets": 9 }
                ],
                "language": {
                    "lengthMenu": "Afficher _MENU_ entrées",
                    "zeroRecords": "Aucun résultat trouvé",
                    "info": "Affichage de _START_ à _END_ sur _TOTAL_ entrées",
                    "infoEmpty": "Affichage de 0 à 0 sur 0 entrée",
                    "infoFiltered": "(filtré à partir de _MAX_ entrées totales)",
                    "search": "Rechercher:",
                    "paginate": {
                        "first": "Premier",
                        "last": "Dernier",
                        "next": "Suivant",
                        "previous": "Précédent"
                    }
                }
            });
        });
        </script>
        </body>
        </html>
        """
        
        # Écrire le contenu complet
        f.write(html_content + table_html + script)
        temp_file_path = f.name
    
    # Lire le contenu du fichier et l'afficher
    with open(temp_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Nettoyer le fichier temporaire
    os.unlink(temp_file_path)
    
    # Afficher le tableau dans Streamlit
    html(html_content, height=900, scrolling=True)

def display_sample_images(df):
    """Affiche des images échantillons par catégorie"""
    if "Catégorie" not in df.columns:
        return
    
    categories_samples = {
        'Good': "IMG_003.jpg",
        'Defects - Hole': "IMG_012.jpg",
        'Defects - Spot': "IMG_080.jpg",
        'Defects - Scratch': "IMG_006.jpg"
    }
    
    selected_images = []
    
    for cat, sample_name in categories_samples.items():
        matching_rows = df[df["Index"] == sample_name]
        if not matching_rows.empty:
            file_path = matching_rows.iloc[0]["Chemin"]
            try:
                img = Image.open(file_path)
                selected_images.append((cat, img))
            except Exception as e:
                st.warning(f"Impossible de charger l'image pour la catégorie {cat}: {e}")
    
    if selected_images:
        cols = st.columns(min(4, len(selected_images)))
        for i, (cat, img) in enumerate(selected_images):
            with cols[i % 4]:
                st.image(img, caption=cat, use_container_width=True)

def create_intensity_plot(df):
    """Crée un histogramme de distribution de l'intensité moyenne"""
    if "Luminosité moyenne" not in df.columns:
        return

    nbins = 30
    values = df["Luminosité moyenne"].dropna().values
    bin_edges = np.histogram_bin_edges(values, bins=nbins)

    bin_indices = np.digitize(df["Luminosité moyenne"], bins=bin_edges, right=False) - 1
    df["Bin"] = bin_indices

    hover_map = {}
    for b in range(len(bin_edges)-1):
        files = df.loc[df["Bin"] == b, "Index"].tolist()
        if files:
            formatted_list = format_file_list(files, per_line=6)
            hover_map[b] = f"{formatted_list}<br>({len(files)} images)"
        else:
            hover_map[b] = "(0 image)"

    fig = px.histogram(
        df,
        x="Luminosité moyenne",
        nbins=nbins,
        title="Distribution de la luminosité des images",
        labels={"Luminosité moyenne": "Luminosité de l'image (0-255)", "count": "Nombre d'images"},
        color_discrete_sequence=["#1f77b4"]
    )

    for trace in fig.data:
        bin_ids = df["Bin"].unique()
        hover_texts = [hover_map.get(b, "") for b in sorted(bin_ids)]
        trace.hovertext = hover_texts
        trace.hovertemplate = "<b>Bin %{x}</b><br>Images:<br>%{hovertext}<extra></extra>"

    fig.update_layout(
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

def format_file_list(files, per_line=6):
    """Supprime les extensions et formate la liste"""
    clean_files = [re.sub(r'\.(jpg|jpeg|png)$', '', f, flags=re.IGNORECASE) for f in files]
    lines = []
    for i in range(0, len(clean_files), per_line):
        lines.append(", ".join(clean_files[i:i+per_line]))
    return "<br>".join(lines)

def create_class_distribution_plot(df):
    """Crée un diagramme à barres du nombre d'images par classe"""
    if "Catégorie" not in df.columns:
        return

    counts = df["Catégorie"].value_counts().reset_index()
    counts.columns = ["Catégorie", "Nombre"]

    color_map = {
        "Good": "green",
        "Defects - Hole": "#FF4C4C",
        "Defects - Scratch": "#CC0000",
        "Defects - Spot": "#FF9999"
    }
    counts["Couleur"] = counts["Catégorie"].map(color_map)

    hover_texts = []
    for cat in counts["Catégorie"]:
        files = df[df["Catégorie"] == cat]["Index"].tolist()
        formatted_list = format_file_list(files, per_line=6)
        hover_texts.append(f"{formatted_list}<br>({len(files)} images)")

    fig = go.Figure(
        data=[
            go.Bar(
                x=counts["Catégorie"],
                y=counts["Nombre"],
                marker_color=counts["Couleur"],
                text=counts["Nombre"],
                textposition="auto",
                hovertext=hover_texts,
                hovertemplate="<b>%{x}</b><br>Images:<br>%{hovertext}<extra></extra>"
            )
        ]
    )

    fig.update_layout(
        title="Nombre d'images par classe",
        xaxis_title="Catégorie",
        yaxis_title="Nombre d'images",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


# Chargement automatique des données
data_path = "data/dataset_analyze_save"

# Vérifier si les données sont chargées
if 'df' not in st.session_state:
    st.session_state['df'] = load_image_dataset_from_local(data_path)

# Vérifier si les données sont chargées
if st.session_state['df'] is None:
    st.error(f"Impossible de charger les données depuis {data_path}")
    st.markdown("""
    ### Instructions :
    1. Assurez-vous que le dossier `data/dataset_analyze_save` existe
    2. Le dossier doit contenir des sous-dossiers avec vos images (ex: `Good/`, `Defect/`)
    3. Rechargez la page pour relancer l'analyse
    """)
else:
    df = st.session_state['df']

    if df is not None and not df.empty:

        st.subheader("📸 Rappel visuel des différentes catégories")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nombre total d'images", len(df))
        
        with col2:
            st.metric("Nombre de catégories", df["Catégorie"].nunique())

        display_sample_images(df)

        st.markdown("---")

        st.subheader("📈 Statistiques générales sur le panel d'images")
        
        col1, col2 = st.columns([1,1])
        with col1:
            create_class_distribution_plot(df)

        with col2:
            create_intensity_plot(df)
        
        st.subheader("🔍 Tableau interactif des données")
        st.info("💡 Pour plus de facilité utilisez les fonctionnalités du tableau : tri + recherche globale.")
        
        create_interactive_table(df)
        
        csv_data = df.drop("Image", axis=1).to_csv(index=False)
        st.download_button(
            label="📥 Télécharger les données complètes (CSV)",
            data=csv_data,
            file_name="analyse_images.csv",
            mime="text/csv"
        )
    else:
        st.error("Erreur lors du chargement des données. Vérifiez le chemin et réessayez.")