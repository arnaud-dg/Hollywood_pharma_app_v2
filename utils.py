import pandas as pd
import streamlit as st
import base64


def load_data(file_path):
    """Charge et prépare les données pour l'analyse"""
    # Détermine le format en fonction de l'extension
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    # Conversion des colonnes de date si nécessaire
    if not pd.api.types.is_datetime64_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    return df


def filter_dataframe(df, start_date=None, end_date=None, categories=None):
    """Filtre le DataFrame selon les critères spécifiés"""
    filtered_df = df.copy()

    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df["date"] >= start_date) & (filtered_df["date"] <= end_date)
        ]

    if categories and len(categories) > 0:
        filtered_df = filtered_df[filtered_df["category"].isin(categories)]

    return filtered_df

def afficher_logo_sidebar(logo_path="assets/images/logo.png", largeur=140):
    with open(logo_path, "rb") as f:
        img_bytes = f.read()
        img_base64 = base64.b64encode(img_bytes).decode()

    logo_html = f"""
        <style>
        /* Cible le conteneur de la sidebar */
        [data-testid="stSidebar"] > div:first-child {{
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }}
        .custom-logo {{
            margin-top: 10px;
            margin-bottom: 20px;
        }}
        </style>

        <img class="custom-logo" src="data:image/png;base64,{img_base64}" width="{largeur}">
    """

    st.sidebar.markdown(logo_html, unsafe_allow_html=True)
