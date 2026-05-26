import os
import sys
import base64
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from keras.models import load_model
import cv2
import tempfile
import tensorflow as tf
import time

# === Utils S3 ===
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_s3 import list_s3_files, read_image_from_s3, read_file_from_s3

# === Streamlit config ===
st.set_page_config(page_title="Monitoring", page_icon="📉", layout="wide")

# === Logo ===
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
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
        unsafe_allow_html=True,
    )
except:
    pass

# === Choix du groupe ===
if "group_choice" not in st.session_state:
    st.session_state.group_choice = ""

group_choice = st.sidebar.selectbox(
    "Sélectionner un groupe",
    options=["", "Grp1", "Grp2", "Grp3", "Grp4", "Grp5", "Grp6", "Grp7"],
    index=0
    if st.session_state.group_choice == ""
    else ["", "Grp1", "Grp2", "Grp3", "Grp4", "Grp5", "Grp6", "Grp7"].index(
        st.session_state.group_choice
    ),
    help="Veuillez choisir un groupe pour afficher le contenu des pages.",
    key="group_choice",
)
st.sidebar.markdown("---")

if st.session_state.group_choice == "":
    st.warning("⚠️ Veuillez sélectionner un groupe dans la barre latérale pour continuer.")
    st.stop()

# === CSS optionnel ===
try:
    with open("assets/css/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# === Page title ===
st.title("📉 4. Suivre la performance du modèle")
st.markdown("Module sur les pratiques de validation et de monitoring de l'IA")

# === Fonctions TFRecord ===
def parse_tfrecord_with_meta(example_proto, target_size=(224, 224)):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "filename": tf.io.FixedLenFeature([], tf.string),
        "class": tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(parsed["image"], channels=3)
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0
    label = parsed["label"]
    filename = parsed["filename"]
    class_str = parsed["class"]
    return image, label, filename, class_str


# === Liste des modèles S3 ===
all_models = list_s3_files(prefix="models/", suffix=".h5")
available_models = [
    os.path.basename(m) for m in all_models if st.session_state.group_choice in os.path.basename(m)
]
model_options = [""] + available_models

selected_model = st.selectbox("Sélectionner un modèle", options=model_options, index=0)

# === Bouton ===
if st.button("Afficher le suivi de tendance du modèle"):
    if selected_model == "":
        st.warning("⚠️ Veuillez sélectionner un modèle avant de lancer l'analyse.")
        st.stop()
    else:
        progress_placeholder = st.empty()

        # Étape 1 : Télécharger modèle
        start_time = time.time()
        progress_placeholder.info("⏳ Étape 1/4 : Téléchargement du modèle depuis S3...")
        model_s3_key = f"models/{selected_model}"
        model_bytes = read_file_from_s3(model_s3_key)
        if not model_bytes:
            st.error(f"Impossible de charger le modèle {selected_model} depuis S3")
            st.stop()
        download_time = time.time() - start_time
        progress_placeholder.success(
            f"✅ Étape 1/4 terminée : Modèle téléchargé ({download_time:.2f}s)"
        )

        # Étape 2 : Chargement modèle
        start_time = time.time()
        progress_placeholder.info("⏳ Étape 2/4 : Chargement du modèle en mémoire...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
            tmp_file.write(model_bytes)
            temp_model_path = tmp_file.name
        try:
            model = load_model(temp_model_path)
        finally:
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
        load_time = time.time() - start_time
        progress_placeholder.success(
            f"✅ Étape 2/4 terminée : Modèle chargé ({load_time:.2f}s)"
        )

        # Étape 3 : Récupération des données (TFRecord latest si dispo)
        start_time = time.time()
        progress_placeholder.info("⏳ Étape 3/4 : Chargement des données...")
        group = st.session_state.group_choice
        test_tfrecord_key = f"tfrecords/{group}/latest_test.tfrecord"
        test_tfrecord_local = None
        tfrecord_bytes = read_file_from_s3(test_tfrecord_key)

        use_tfrecord = False
        if tfrecord_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tfrecord") as tmp_tf:
                tmp_tf.write(tfrecord_bytes)
                test_tfrecord_local = tmp_tf.name
            use_tfrecord = True
        list_time = time.time() - start_time
        if use_tfrecord:
            progress_placeholder.success(
                f"✅ Étape 3/4 terminée : TFRecord chargé ({list_time:.2f}s)"
            )
        else:
            s3_root = "data/data_maintenance"
            all_files = list_s3_files(prefix=s3_root + "/")
            subfolders = sorted(
                list(
                    set(
                        [
                            f.split("/")[2]
                            for f in all_files
                            if f.count("/") >= 3 and f.split("/")[2].startswith("W")
                        ]
                    )
                )
            )
            progress_placeholder.success(
                f"✅ Étape 3/4 terminée : {len(subfolders)} dossiers trouvés ({list_time:.2f}s)"
            )

        # Étape 4 : Analyse
        start_time = time.time()
        progress_placeholder.info("⏳ Étape 4/4 : Analyse en cours...")

        class_names = ["Good", "Hole", "Scratch", "Spot"]
        results = []
        errors_all = {}

        if use_tfrecord:
            dataset = tf.data.TFRecordDataset(test_tfrecord_local).map(parse_tfrecord_with_meta)
            dataset = dataset.batch(32)

            for batch in dataset:
                images, labels, filenames, classes = batch
                preds = model.predict(images, verbose=0)
                y_pred = np.argmax(preds, axis=1)
                y_true = labels.numpy()
                confidences = np.max(preds, axis=1)

                for img, yt, yp, fname, cl, conf in zip(
                    images, y_true, y_pred, filenames, classes, confidences
                ):
                    folder = fname.numpy().decode("utf-8").split("_")[0]  # ex: W01
                    if folder not in errors_all:
                        errors_all[folder] = []
                    if yp != yt:
                        img_np = (img.numpy() * 255).astype("uint8")
                        errors_all[folder].append(
                            (
                                img_np,
                                cl.numpy().decode("utf-8"),
                                class_names[int(yp)],
                                float(conf),
                                fname.numpy().decode("utf-8"),
                            )
                        )
                # Accuracy globale par "folder"
                # Ici tu peux grouper par semaine / folder
            # Exemple simple: on calcule global accuracy
            total = sum([len(v) for v in errors_all.values()])
            accuracy = 100 * (1 - total / sum([len(v) for v in errors_all.values()]))
            results.append({"Dossier": "Global", "Accuracy": accuracy})

        else:
            # fallback lecture brute depuis S3 (comme ton code initial)
            for idx, folder in enumerate(subfolders):
                folder_prefix = f"{s3_root}/{folder}/"
                folder_images = list_s3_files(prefix=folder_prefix, suffix=".jpg")
                folder_images.extend(list_s3_files(prefix=folder_prefix, suffix=".png"))
                folder_images.extend(list_s3_files(prefix=folder_prefix, suffix=".jpeg"))

                X_batch, y_true, img_names, img_resized_list = [], [], [], []
                input_size = (224, 224)
                for img_s3_key in folder_images:
                    img_name = os.path.basename(img_s3_key)
                    true_class = img_name.split("_")[0]
                    img_bytes_io = read_image_from_s3(img_s3_key)
                    if not img_bytes_io:
                        continue
                    try:
                        img_bytes = img_bytes_io.read()
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_resized = cv2.resize(img, input_size)
                        X_batch.append(img_resized.astype("float32") / 255.0)
                        y_true.append(true_class)
                        img_names.append(img_name)
                        img_resized_list.append(img_resized)
                    except:
                        continue

                if len(X_batch) > 0:
                    X_batch = np.stack(X_batch, axis=0)
                    preds = model.predict(X_batch, batch_size=32, verbose=0)
                    y_pred = [class_names[np.argmax(p)] for p in preds]
                    confidences = [float(np.max(p)) for p in preds]

                    # Accuracy dossier
                    correct = sum([1 for yt, yp in zip(y_true, y_pred) if yt == yp])
                    total = len(y_true)
                    accuracy = (correct / total) * 100 if total > 0 else 0
                    results.append({"Dossier": folder, "Accuracy": accuracy})

                    errors = [
                        (img_resized_list[i], y_true[i], y_pred[i], confidences[i], img_names[i])
                        for i in range(len(y_true))
                        if y_pred[i] != y_true[i]
                    ]
                    if errors:
                        errors_all[folder] = errors

        prediction_time = time.time() - start_time
        total_time = download_time + load_time + list_time + prediction_time
        progress_placeholder.success(f"✅ Analyse terminée ! Temps total : {total_time:.1f}s")

        # === Résultats globaux ===
        df = pd.DataFrame(results)
        if not df.empty:
            fig = px.line(
                df,
                x="Dossier",
                y="Accuracy",
                markers=True,
                title="Carte de contrôle - Suivi de la performance du modèle (Accuracy)",
                labels={"Dossier": "Semaine", "Accuracy": "Accuracy (%)"},
            )
            fig.update_yaxes(range=[50, 100])
            fig.update_traces(
                text=[f"{val:.1f}%" for val in df["Accuracy"]],
                textposition="top center",
            )
            st.plotly_chart(fig, use_container_width=True)

        # === Détails erreurs ===
        for folder, errors in errors_all.items():
            with st.expander(f"Erreur de prédiction - {folder}", expanded=False):
                if errors:
                    for i in range(0, len(errors), 4):
                        cols = st.columns(4)
                        for j, col in enumerate(cols):
                            if i + j < len(errors):
                                img_array, true_class, pred_class, confidence, img_name = errors[
                                    i + j
                                ]
                                with col:
                                    st.image(img_array, use_container_width=True)
                                    st.markdown(
                                        f"<div style='text-align:center'>"
                                        f"<b>Image :</b> {img_name}<br>"
                                        f"<span style='color:blue'><b>Réel :</b> {true_class}</span><br>"
                                        f"<span style='color:red'><b>Prédit :</b> {pred_class}</span><br>"
                                        f"<b>Probabilité :</b> {confidence:.2f}"
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )
                else:
                    st.success("✅ Aucune erreur de prédiction pour ce dossier")
