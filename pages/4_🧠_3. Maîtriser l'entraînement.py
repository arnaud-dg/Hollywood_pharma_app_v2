import os
import sys
import base64
import json
import tempfile
import time
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

from keras.callbacks import Callback
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

# ====== S3 utils import ======
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_s3 import (
    list_s3_files,
    read_image_from_s3,
    write_file_to_s3,
    read_file_from_s3,
    get_bucket_name
)

# =========================
# == CONFIG / SEED / UI ==
# =========================

GROUP_SEEDS = {"Grp1":101,"Grp2":202,"Grp3":303,"Grp4":404,"Grp5":505,"Grp6":606,"Grp7":707}
DEFAULT_SEED = 42

st.set_page_config(page_title="Entraînement de l'IA", page_icon="🧠", layout="wide")

os.environ["TF_DETERMINISTIC_OPS"] = "1"
# Ne désactive pas les optis TF:
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Ne masque pas le GPU:
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Logo
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

# Choix du groupe
if "group_choice" not in st.session_state:
    st.session_state.group_choice = ""

group_choice = st.sidebar.selectbox(
    "Sélectionner un groupe",
    options=["", "Grp1", "Grp2", "Grp3", "Grp4", "Grp5", "Grp6", "Grp7"],
    index=0 if st.session_state.group_choice == "" else ["", "Grp1", "Grp2", "Grp3", "Grp4", "Grp5", "Grp6", "Grp7"].index(st.session_state.group_choice),
    help="Veuillez choisir un groupe pour afficher le contenu des pages.",
    key="group_choice"
)
st.sidebar.markdown("---")

if st.session_state.group_choice == "":
    st.warning("⚠️ Veuillez sélectionner un groupe dans la barre latérale pour continuer.")
    st.stop()

SEED = GROUP_SEEDS.get(st.session_state.group_choice, DEFAULT_SEED)
import random
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# CSS optionnel
try:
    with open("assets/css/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# =========================
# == HEADERS & DEBUG GPU ==
# =========================
st.title("🔎 3. Maîtriser l'entraînement du modèle")
st.write("Dispositifs:", tf.config.list_physical_devices())
st.write("GPUs détectés:", tf.config.list_physical_devices('GPU'))
st.markdown("""
**Objectifs :**
- Transformer le dataset en **TFRecords** optimisés pour l'entraînement.
- Accélérer l'IO et stabiliser les performances d'entraînement/inférence.
""")

# =============================
# == LOGS / PERSISTENCE S3  ==
# =============================
def save_training_log(train_params, metrics, training_time, model_path=None, tfrecords_info=None):
    """Sauvegarder les logs d'entraînement dans S3."""
    timestamp = datetime.now()
    log_id = timestamp.strftime("%Y%m%d_%H%M%S")
    log_data = {
        "log_id": log_id,
        "timestamp": timestamp.isoformat(),
        "date": timestamp.strftime("%Y-%m-%d"),
        "heure": timestamp.strftime("%H:%M:%S"),
        **train_params,
        **metrics,
        "training_time": training_time,
        "model_path": model_path,
        "tfrecords": tfrecords_info or {}
    }
    s3_key = f"logs/training_log_{log_id}.json"
    write_file_to_s3(s3_key, json.dumps(log_data, indent=2).encode("utf-8"))
    return log_id

def load_all_logs():
    logs = []
    log_files = list_s3_files(prefix="logs/", suffix=".json")
    for log_file in log_files:
        try:
            content = read_file_from_s3(log_file)
            if content:
                log_data = json.loads(content.decode('utf-8'))
                logs.append(log_data)
        except:
            continue
    logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return logs

# ====================================
# == DATA LOADING (from S3, cached) ==
# ====================================
@st.cache_data(show_spinner=True)
def load_images_from_dataset_augmented(s3_prefix="data/dataset_augmented_full", target_size=(224, 224)):
    """Lit toutes les images depuis S3 (lent), pour 1ère génération TFRecord."""
    images, labels, file_paths = [], [], []
    all_files = list_s3_files(prefix=s3_prefix)
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        return None, None, None

    progress_bar = st.progress(0)
    total = len(image_files)
    for idx, s3_key in enumerate(image_files):
        progress_bar.progress((idx + 1) / total)
        parts = s3_key.replace(s3_prefix + "/", "").split("/")
        label = parts[0] if len(parts) > 1 else "Unknown"
        img_bytes_io = read_image_from_s3(s3_key)
        if not img_bytes_io:
            continue
        try:
            img_bytes = img_bytes_io.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, target_size)
            images.append(img_resized)
            labels.append(label)
            file_paths.append(s3_key)
        except:
            continue
    progress_bar.empty()
    if not images:
        return None, None, None
    X = np.array(images, dtype=np.uint8)
    y = np.array(labels)
    return X, y, file_paths

def balance_dataset(X, y, file_paths, target_size=300, random_state=42):
    """Rééquilibrer le dataset."""
    X_balanced, y_balanced, file_paths_balanced = [], [], []
    classes = np.unique(y)
    for c in classes:
        idx = np.where(y == c)[0]
        X_class, y_class, f_class = X[idx], y[idx], np.array(file_paths)[idx]
        if len(X_class) > target_size:
            X_res, y_res, f_res = resample(
                X_class, y_class, f_class,
                replace=False, n_samples=target_size, random_state=random_state
            )
        else:
            X_res, y_res, f_res = resample(
                X_class, y_class, f_class,
                replace=True, n_samples=target_size, random_state=random_state
            )
        X_balanced.append(X_res); y_balanced.append(y_res); file_paths_balanced.append(f_res)
    return np.vstack(X_balanced), np.hstack(y_balanced), np.hstack(file_paths_balanced)

# ==========================
# == CLASS WEIGHTS / METR ==
# ==========================
def get_class_weights(num_classes, penalty_balance):
    weights = {}
    base = 1.0
    for i in range(num_classes):
        if i == 0:
            weights[i] = base * (0.5 + penalty_balance)
        else:
            weights[i] = base * (1.5 - penalty_balance)
    return weights

def recall_m(y_true, y_pred):
    y_pred_classes = tf.argmax(y_pred, axis=1)
    y_true = tf.cast(y_true, tf.int64)
    tp = tf.reduce_sum(tf.cast(y_true == y_pred_classes, tf.float32))
    fn = tf.reduce_sum(tf.cast(y_true != y_pred_classes, tf.float32))
    return tp / (tp + fn + tf.keras.backend.epsilon())

def precision_m(y_true, y_pred):
    y_pred_classes = tf.argmax(y_pred, axis=1)
    y_true = tf.cast(y_true, tf.int64)
    tp = tf.reduce_sum(tf.cast(y_true == y_pred_classes, tf.float32))
    fp = tf.reduce_sum(tf.cast(y_true != y_pred_classes, tf.float32))
    return tp / (tp + fp + tf.keras.backend.epsilon())

# ================
# == CNN MODEL  ==
# ================
def create_model(learning_rate, input_shape=(224, 224, 3), num_classes=2):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', precision_m, recall_m]
    )
    return model

class StreamlitProgressBar(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.gif_placeholder = st.empty()
        GIF_PATH = "assets/images/animation_nn.gif"
        if os.path.exists(GIF_PATH):
            with self.gif_placeholder.container():
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    st.image(GIF_PATH, use_container_width=True)
        self.progress_bar = st.progress(0, text="⏳ Entraînement en cours...")

    def on_epoch_end(self, epoch, logs=None):
        progress = int(((epoch + 1) / self.total_epochs) * 100)
        self.progress_bar.progress(progress, text=f"Époque {epoch+1}/{self.total_epochs}")

    def on_train_end(self, logs=None):
        self.gif_placeholder.empty()

# ======================
# == Grad-CAM (XAI)   ==
# ======================
def gradCAM_custom(model, image_array, target_size=(224, 224), colormap=cv2.COLORMAP_JET):
    try:
        last_conv_idx = None
        for i, layer in enumerate(reversed(model.layers)):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_idx = len(model.layers) - 1 - i
                break
        if last_conv_idx is None:
            raise ValueError("Aucune couche Conv2D trouvée dans le modèle.")

        with tf.GradientTape() as tape:
            x = image_array
            conv_output = None
            for i, layer in enumerate(model.layers):
                x = layer(x, training=False)
                if i == last_conv_idx:
                    conv_output = x
                    tape.watch(conv_output)
            predictions = x
            target_class = tf.argmax(predictions[0])
            loss = predictions[0, target_class]

        grads = tape.gradient(loss, conv_output)
        if grads is None:
            st.error("Impossible de calculer les gradients")
            return None, None
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]
        heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        heatmap_resized = cv2.resize(heatmap.numpy(), target_size)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        return heatmap_resized, heatmap_colored
    except Exception as e:
        st.error(f"Erreur lors de la génération de Grad-CAM: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

# ============================
# == PDF report (inchangé)  ==
# ============================
def generate_pdf_report(model_name, accuracy, precision, recall, false_rejects, 
                       missed_defects, total_samples, training_time, 
                       fig_loss=None, fig_cm=None, errors_images=None):
    from reportlab.lib import colors
    from reportlab.platypus import Image as RLImage, PageBreak
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20, bottomMargin=20)
    styles = getSampleStyleSheet()
    story = []
    temp_files = []
    try:
        model_name = str(model_name)
        accuracy = float(accuracy); precision = float(precision); recall = float(recall)
        false_rejects = int(false_rejects); missed_defects = int(missed_defects)
        total_samples = int(total_samples); training_time = float(training_time)

        story.append(Paragraph("<b>Rapport d'Entraînement du Modèle</b>", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>Modèle :</b> {model_name}", styles["Normal"]))
        story.append(Paragraph(f"<b>Date :</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
        story.append(Paragraph(f"<b>Durée d'entraînement :</b> {training_time:.2f} secondes", styles["Normal"]))
        story.append(Spacer(1, 20))

        story.append(Paragraph("<b>Métriques de Performance</b>", styles["Heading2"]))
        story.append(Spacer(1, 10))
        data = [
            ["Métrique", "Valeur"],
            ["Accuracy", "{:.2f}%".format(accuracy * 100)],
            ["Précision", "{:.2f}%".format(precision * 100)],
            ["Recall", "{:.2f}%".format(recall * 100)],
            ["Faux rejets", "{} ({:.2f}%)".format(false_rejects, false_rejects * 100 / total_samples)],
            ["Défauts manqués", "{} ({:.2f}%)".format(missed_defects, missed_defects * 100 / total_samples)],
            ["Total échantillons", "{}".format(total_samples)]
        ]
        table = Table(data, colWidths=[250, 200])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F2F2F2')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(table)

        if fig_loss is not None or fig_cm is not None:
            story.append(PageBreak())
            story.append(Paragraph("<b>Courbes d'Apprentissage et Matrice de Confusion</b>", styles["Heading2"]))
            story.append(Spacer(1, 10))
            if fig_loss is not None:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png'); tmp.close()
                temp_files.append(tmp.name)
                fig_loss.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
                story.append(RLImage(tmp.name, width=450, height=300))
                story.append(Spacer(1, 10))
            if fig_cm is not None:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png'); tmp.close()
                temp_files.append(tmp.name)
                fig_cm.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
                story.append(RLImage(tmp.name, width=350, height=300))

        if errors_images and len(errors_images) > 0:
            from reportlab.platypus import PageBreak
            story.append(PageBreak())
            story.append(Paragraph("<b>Erreurs de Classification</b>", styles["Heading2"]))
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"Nombre total d'erreurs : {len(errors_images)}", styles["Normal"]))
            story.append(Spacer(1, 15))
            row = []
            for i, (img_array, true_class, pred_class, confidence, filename) in enumerate(errors_images[:12]):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png'); tmp.close()
                temp_files.append(tmp.name)
                from PIL import Image as PILImage
                PILImage.fromarray(img_array).save(tmp.name, 'PNG')
                text = f"<font size=8><b>{filename}</b><br/>Réel: {true_class}<br/>Prédit: {pred_class}<br/>Conf: {confidence:.2f}</font>"
                row.append([RLImage(tmp.name, width=100, height=100), Paragraph(text, styles["Normal"])])
                if (i+1) % 4 == 0:
                    from reportlab.platypus import Table
                    tbl = Table([list(zip(*row))[0], list(zip(*row))[1]], colWidths=[120,120,120,120])
                    tbl.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'TOP')]))
                    story.append(tbl); story.append(Spacer(1, 15)); row = []
            if row:
                from reportlab.platypus import Table
                tbl = Table([list(zip(*row))[0], list(zip(*row))[1]], colWidths=[120,120,120,120])
                tbl.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'TOP')]))
                story.append(tbl)

        doc.build(story)
        pdf = buffer.getvalue()
        buffer.close()
        return pdf
    finally:
        time.sleep(0.1)
        for tmp_file in temp_files:
            try:
                if os.path.exists(tmp_file): os.unlink(tmp_file)
            except:
                pass

# ==========================================
# == TFRecord: serialize / write / parse  ==
# ==========================================
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):  # EagerTensor
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

def _str_feature(value):
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))

def serialize_example(image_uint8, label_int, filename_str, class_str):
    feature = {
        "image": _bytes_feature(tf.io.encode_jpeg(image_uint8).numpy()),
        "label": _int64_feature(label_int),
        "filename": _str_feature(filename_str),
        "class": _str_feature(class_str),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecord(X, y_int, file_paths, label_encoder, output_path):
    """Sauve X,y + meta dans un TFRecord (image uint8, label int, filename, class)."""
    classes = list(label_encoder.classes_)
    with tf.io.TFRecordWriter(output_path) as writer:
        for img, yi, fp in zip(X, y_int, file_paths):
            img_tensor = tf.convert_to_tensor(img, dtype=tf.uint8)
            filename = os.path.basename(fp)
            class_str = classes[int(yi)] if 0 <= int(yi) < len(classes) else "UNK"
            example = serialize_example(img_tensor, int(yi), filename, class_str)
            writer.write(example)
    return output_path

# parse (pour entraînement: retourne (image,label))
def parse_tfrecord_train(example_proto, target_size=(224,224)):
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
    label = tf.cast(parsed["label"], tf.int64)
    return image, label

# parse (pour éval/XAI: retourne (image,label,filename,class))
def parse_tfrecord_with_meta(example_proto, target_size=(224,224)):
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
    label = tf.cast(parsed["label"], tf.int64)
    filename = parsed["filename"]
    class_str = parsed["class"]
    return image, label, filename, class_str

def get_dataset_from_tfrecord(tfrecord_path, batch_size=32, shuffle=True, with_meta=False):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    if with_meta:
        dataset = raw_dataset.map(parse_tfrecord_with_meta, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = raw_dataset.map(parse_tfrecord_train, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def list_filenames_in_tfrecord_local(tfrecord_local_path, class_filter=None):
    """Parcourt le TFRecord local et retourne la liste des filenames (optionnel: filtrés par classe)."""
    names = []
    for raw in tf.data.TFRecordDataset(tfrecord_local_path):
        img, label, filename, class_str = parse_tfrecord_with_meta(raw, target_size=(224,224))
        fname = filename.numpy().decode('utf-8')
        cstr = class_str.numpy().decode('utf-8')
        if class_filter is None or cstr == class_filter:
            names.append(fname)
    return sorted(list(set(names)))

# =====================================
# == UI TABS ==
# =====================================
tab1, tab2, tab3 = st.tabs(["Entraînement", "Traçabilité & Logs", "Explicabilité"])

# =========================
# == ONGLET 1: TRAINING  ==
# =========================
with tab1:
    with st.container(border=True):
        st.subheader("⚙️ Paramètres d'entraînement")
        col1, col2 = st.columns(2)
        with col1:
            train_ratio = st.slider(
                "Proportion des données utilisées pour l'entraînement",
                min_value=50, max_value=95, step=5, value=50,
                help="📊 Taille du train set relativement au test set"
            ) / 100.0
        with col2:
            penalty_balance = st.slider(
                "Balance de pénalités",
                min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                help="Glisser à gauche (0) pour pénaliser les faux négatifs. À droite (1) pour pénaliser faux positifs."
            )
        col1, col2, col3 = st.columns(3)
        with col1:
            learning_rate = st.select_slider(
                "Learning Rate", options=[0.0005, 0.001, 0.005, 0.01], value=0.001
            )
        with col2:
            epochs = st.slider("Époques", min_value=5, max_value=50, step=5, value=5)
        with col3:
            batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=64)

    st.markdown("---")

    train_button = st.button("🎯 Démarrer un nouvel entraînement", type="primary")

    if train_button:
        s3_prefix = "data/dataset_augmented_full"
        expected_classes = ["Good", "Hole", "Scratch", "Spot"]

        # Vérif existence classes
        for cls in expected_classes:
            class_path = f"{s3_prefix}/{cls}/"
            _ = list_s3_files(prefix=class_path, suffix=".jpg")

        # 1) Chargement dataset (1ère fois : depuis S3)
        X, y, file_paths = load_images_from_dataset_augmented(s3_prefix=s3_prefix, target_size=(224,224))
        if X is None or y is None:
            st.error("❌ Impossible de charger les données depuis S3.")
            st.stop()

        # 2) Encodage labels
        encoder = LabelEncoder()
        encoder.fit(expected_classes)
        y_encoded = encoder.transform(y)
        num_classes = len(expected_classes)

        # 3) Normalisation
        X = X.astype('float32') / 255.0

        # 4) Rééquilibrage
        X_bal, y_bal, file_paths_bal = balance_dataset(X, y_encoded, file_paths, target_size=300, random_state=SEED)

        # 5) Split train/test
        X_train, X_test, y_train, y_test, fp_train, fp_test = train_test_split(
            X_bal, y_bal, file_paths_bal,
            train_size=train_ratio, random_state=SEED, stratify=y_bal
        )

        # 6) Écriture TFRecords (locaux)
        date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        group = st.session_state.group_choice
        train_tfrecord_local = f"train_{group}_{date_tag}.tfrecord"
        test_tfrecord_local = f"test_{group}_{date_tag}.tfrecord"

        write_tfrecord(X_train*255.0, y_train, fp_train, encoder, train_tfrecord_local)  # remettre en uint8
        write_tfrecord(X_test*255.0, y_test, fp_test, encoder, test_tfrecord_local)

        # 7) Upload TFRecords vers S3 (datés + alias latest)
        with open(train_tfrecord_local, "rb") as f:
            s3_train_key = f"tfrecords/{group}/{os.path.basename(train_tfrecord_local)}"
            write_file_to_s3(s3_train_key, f.read())
        with open(test_tfrecord_local, "rb") as f:
            s3_test_key = f"tfrecords/{group}/{os.path.basename(test_tfrecord_local)}"
            write_file_to_s3(s3_test_key, f.read())
        # alias latest pour XAI
        with open(train_tfrecord_local, "rb") as f:
            write_file_to_s3(f"tfrecords/{group}/latest_train.tfrecord", f.read())
        with open(test_tfrecord_local, "rb") as f:
            write_file_to_s3(f"tfrecords/{group}/latest_test.tfrecord", f.read())

        # 8) Construction tf.data.Dataset
        train_dataset = get_dataset_from_tfrecord(train_tfrecord_local, batch_size=batch_size, shuffle=True, with_meta=False)
        test_dataset = get_dataset_from_tfrecord(test_tfrecord_local, batch_size=batch_size, shuffle=False, with_meta=False)

        # 9) Modèle + class weights
        model = create_model(learning_rate, input_shape=(224,224,3), num_classes=num_classes)
        class_weight = get_class_weights(num_classes, penalty_balance)

        # 10) Entraînement
        start_time = time.time()
        history = model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=epochs,
            class_weight=class_weight,
            callbacks=[StreamlitProgressBar(epochs)],
            verbose=0
        )
        training_time = time.time() - start_time

        # 11) Évaluation + y_true/y_pred à partir du TFRecord (avec meta)
        eval_dataset = get_dataset_from_tfrecord(test_tfrecord_local, batch_size=batch_size, shuffle=False, with_meta=True)
        y_true, y_pred, filenames_eval, classes_eval, probs_eval = [], [], [], [], []
        for batch in eval_dataset:
            if len(batch) == 4:
                images_b, labels_b, fnames_b, class_str_b = batch
            else:
                images_b, labels_b = batch; fnames_b = None; class_str_b = None
            preds = model.predict(images_b, verbose=0)
            y_true.extend(labels_b.numpy().tolist())
            y_pred.extend(np.argmax(preds, axis=1).tolist())
            probs_eval.extend(np.max(preds, axis=1).tolist())
            if fnames_b is not None:
                filenames_eval.extend([x.decode('utf-8') for x in fnames_b.numpy().tolist()])
                classes_eval.extend([x.decode('utf-8') for x in class_str_b.numpy().tolist()])

        y_true = np.array(y_true); y_pred = np.array(y_pred)
        accuracy = np.mean(y_true == y_pred)
        # NB: précision/recall simplistes (cohérents avec ta version initiale)
        precision = np.sum((y_true == y_pred) & (y_pred == 1)) / (np.sum(y_pred == 1) + 1e-8)
        recall = np.sum((y_true == y_pred) & (y_true == 1)) / (np.sum(y_true == 1) + 1e-8)

        # 12) Sauvegarde modèle
        today = datetime.now().strftime("%Y%m%d")
        model_name = f"model_{today}_{group}.h5"
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            temp_model_path = tmp_file.name
        try:
            model.save(temp_model_path)
            with open(temp_model_path, 'rb') as f:
                s3_model_key = f"models/{model_name}"
                write_file_to_s3(s3_model_key, f.read())
        finally:
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)

        # 13) Logs
        metrics_dict = {"accuracy": accuracy, "precision": precision, "recall": recall}
        tfrec_info = {
            "train_tfrecord_s3": s3_train_key,
            "test_tfrecord_s3": s3_test_key,
            "train_tfrecord_local": train_tfrecord_local,
            "test_tfrecord_local": test_tfrecord_local
        }
        save_training_log(
            {
                'train_ratio': float(train_ratio),
                'batch_size': int(batch_size),
                'learning_rate': float(learning_rate),
                'epochs': int(epochs),
                'penalty_balance': float(penalty_balance),
                'group': group
            },
            metrics_dict,
            training_time,
            s3_model_key,
            tfrecords_info=tfrec_info
        )

        st.success(f"✅ Entraînement terminé en {training_time:.2f}s - Modèle sauvegardé : **{model_name}**")

        # 14) Graphiques
        st.markdown("---")
        st.subheader("📈 Résultats - graphiques")
        col1, col2 = st.columns(2)
        with col1:
            fig_loss, ax = plt.subplots(2, 1, figsize=(12, 9))
            ax[0].plot(history.history["loss"], label="Train Loss")
            if "val_loss" in history.history:
                ax[0].plot(history.history["val_loss"], label="Val Loss")
            ax[0].set_title("Courbe de Loss"); ax[0].legend()
            ax[1].plot(history.history.get("accuracy", []), label="Train Accuracy")
            if "val_accuracy" in history.history:
                ax[1].plot(history.history["val_accuracy"], label="Val Accuracy")
            ax[1].set_title("Courbe d'Accuracy"); ax[1].legend()
            fig_loss.suptitle("Courbes d'Apprentissage", fontsize=16, weight="bold")
            st.pyplot(fig_loss)

        with col2:
            cm = confusion_matrix(y_true, y_pred)
            fig_cm, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
            ax.set_title("Matrice de Confusion", fontsize=12, weight="bold")
            st.pyplot(fig_cm)

        # 15) Métriques dérivées
        st.subheader("📊 Métriques & Interprétation des résultats")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("**Accuracy**", f"{accuracy * 100:.2f}%")
        with col2: st.metric("**Precision**", f"{precision * 100:.2f}%")
        with col3: st.metric("**Recall**", f"{recall * 100:.2f}%")

        class_labels = list(encoder.classes_)
        total_samples = len(y_true)
        good_class_idx = class_labels.index("Good") if "Good" in class_labels else 0
        defect_class_idx = [i for i, c in enumerate(class_labels) if c != "Good"]
        false_rejects = int(np.sum((y_true == good_class_idx) & (y_pred != good_class_idx)))
        missed_defects = int(np.sum((np.isin(y_true, defect_class_idx)) & (y_pred == good_class_idx)))

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("**Pourcentage de faux-rejets**", f"{false_rejects * 100 / total_samples:.2f}%")
        with col2: st.metric("**Pourcentage de défauts manqués**", f"{missed_defects * 100 / total_samples:.2f}%")
        with col3: st.metric("**Durée de l'entrainement**", f"{training_time:.1f}s")

        # 16) Erreurs de classification (à partir du TFRecord test)
        st.markdown("### 🔎 Visualisation des erreurs de classification")
        # re-crée un dataset UNBATCHED pour récupérer les images brutes facilement
        test_unbatched = tf.data.TFRecordDataset(test_tfrecord_local).map(parse_tfrecord_with_meta)
        errors_data = []
        idx = 0
        for img, label, filename, class_str in test_unbatched:
            if idx >= len(y_pred): break
            # on a l'alignement dans l'ordre d'encodage ; sinon, on peut recalculer preds en ligne
            idx += 1

        # recalcul propre: on batch et on compare (plus fiable)
        errors_data = []
        for images_b, labels_b, fnames_b, class_str_b in get_dataset_from_tfrecord(test_tfrecord_local, batch_size=batch_size, shuffle=False, with_meta=True):
            preds_b = model.predict(images_b, verbose=0)
            ypb = np.argmax(preds_b, axis=1)
            confb = np.max(preds_b, axis=1)
            imgs_np = (images_b.numpy() * 255).astype("uint8")
            names = [x.decode('utf-8') for x in fnames_b.numpy().tolist()]
            trues = [x.decode('utf-8') for x in class_str_b.numpy().tolist()]
            for img_arr, true_c, pred_idx, c, name in zip(imgs_np, trues, ypb, confb, names):
                pred_c = class_labels[int(pred_idx)]
                if pred_c != true_c:
                    errors_data.append((img_arr, true_c, pred_c, float(c), name))
        if len(errors_data) == 0:
            st.success("✅ Aucune erreur de classification dans ce jeu de test.")
        else:
            st.info(f"Nombre total d'erreurs : {len(errors_data)}")
            for i in range(0, min(len(errors_data), 12), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    if i + j < len(errors_data):
                        img_array, true_class, pred_class, confidence, file_name = errors_data[i + j]
                        with col:
                            st.image(img_array, use_container_width=True)
                            st.markdown(
                                f"<div style='text-align:center'>"
                                f"<b>Image :</b> {file_name}<br>"
                                f"<span style='color:blue'><b>Réel :</b> {true_class}</span><br>"
                                f"<span style='color:red'><b>Prédit :</b> {pred_class}</span><br>"
                                f"<b>Probabilité :</b> {confidence:.2f}"
                                f"</div>",
                                unsafe_allow_html=True
                            )

        # 17) PDF
        pdf_data = generate_pdf_report(
            str(model_name),
            float(accuracy), float(precision), float(recall),
            int(false_rejects), int(missed_defects), int(total_samples),
            float(training_time),
            fig_loss=fig_loss, fig_cm=fig_cm,
            errors_images=errors_data[:12]
        )
        if pdf_data is not None:
            st.download_button(
                label="📥 Télécharger le rapport complet (PDF)",
                data=pdf_data,
                file_name=f"rapport_complet_{model_name.replace('.h5','')}.pdf",
                mime="application/pdf"
            )

# ============================
# == ONGLET 2: LOGS (idem)  ==
# ============================
with tab2:
    st.header("📋 Logs")
    logs = load_all_logs()
    if logs:
        st.dataframe(pd.DataFrame(logs))
    else:
        st.info("Aucun log trouvé dans S3")

# ============================
# == ONGLET 3: XAI (TFRecord) ==
# ============================
with tab3:
    st.header("🔍 Analyse d'explicabilité")
    # Modèles disponibles
    available_models = list_s3_files(prefix="models/", suffix=".h5")
    model_names = [os.path.basename(m) for m in available_models]
    if not model_names:
        st.warning("Pas de modèle disponible dans S3")
    else:
        selected_model = st.selectbox("Modèle", sorted(model_names, reverse=True))

        # On lit depuis le TFRecord test "latest" du groupe si présent
        tfrecord_test_latest_s3 = f"tfrecords/{st.session_state.group_choice}/latest_test.tfrecord"
        test_tfrecord_local = None
        tfrecord_bytes = read_file_from_s3(tfrecord_test_latest_s3)
        use_tfrecord = False
        if tfrecord_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tfrecord") as tmp_tf:
                tmp_tf.write(tfrecord_bytes)
                test_tfrecord_local = tmp_tf.name
            use_tfrecord = True

        classes = ["Good", "Hole", "Scratch", "Spot"]
        selected_class = st.selectbox("Classe du comprimé", classes)

        image_names = []
        if use_tfrecord:
            # lister les filenames du TFRecord filtrés par class
            image_names = list_filenames_in_tfrecord_local(test_tfrecord_local, class_filter=selected_class)

        # Fallback S3 si TFRecord absent
        if not image_names:
            s3_class_prefix = f"data/dataset_augmented_full/{selected_class}/"
            class_images = list_s3_files(prefix=s3_class_prefix, suffix=".jpg")
            class_images += list_s3_files(prefix=s3_class_prefix, suffix=".png")
            image_names = [os.path.basename(img) for img in class_images]

        if image_names:
            selected_image_name = st.selectbox("Image de test", sorted(image_names))
            if st.button("Analyser"):
                # Charger modèle
                model_s3_key = f"models/{selected_model}"
                model_bytes = read_file_from_s3(model_s3_key)
                if not model_bytes:
                    st.error(f"Impossible de charger le modèle {selected_model} depuis S3")
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                        tmp_file.write(model_bytes)
                        temp_model_path = tmp_file.name
                    try:
                        model = load_model(temp_model_path, custom_objects={'precision_m': precision_m, 'recall_m': recall_m})
                    finally:
                        if os.path.exists(temp_model_path):
                            os.remove(temp_model_path)

                    # Récupérer l'image depuis TFRecord si possible
                    img_resized = None
                    if use_tfrecord:
                        # filtre dataset par filename
                        def filter_by_name(img, label, filename, class_str):
                            return tf.equal(filename, tf.constant(selected_image_name.encode('utf-8')))
                        ds = tf.data.TFRecordDataset(test_tfrecord_local).map(parse_tfrecord_with_meta)
                        ds = ds.filter(filter_by_name).take(1)
                        found = False
                        for img, label, fname, class_str in ds:
                            img_np = img.numpy()
                            img_resized = (img_np * 255).astype("uint8")
                            found = True
                            break
                        if not found:
                            img_resized = None

                    # Fallback S3 direct si pas trouvé
                    if img_resized is None:
                        s3_class_prefix = f"data/dataset_augmented_full/{selected_class}/"
                        class_images = list_s3_files(prefix=s3_class_prefix)
                        selected_s3_key = None
                        for p in class_images:
                            if os.path.basename(p) == selected_image_name:
                                selected_s3_key = p; break
                        if selected_s3_key:
                            img_bytes_io = read_image_from_s3(selected_s3_key)
                            if img_bytes_io:
                                img_bytes = img_bytes_io.read()
                                nparr = np.frombuffer(img_bytes, np.uint8)
                                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img_resized = cv2.resize(img_rgb, (224, 224))

                    if img_resized is None:
                        st.error("Image introuvable.")
                    else:
                        img_norm = img_resized.astype("float32") / 255.0
                        img_batch = np.expand_dims(img_norm, axis=0)
                        preds = model.predict(img_batch, verbose=0)
                        pred_class = np.argmax(preds[0]); confidence = np.max(preds[0])
                        class_mapping = {i: c for i, c in enumerate(classes)}
                        pred_label = class_mapping.get(pred_class, str(pred_class))

                        st.markdown(f"<span style='color:blue'>Réel :</span> {selected_class}", unsafe_allow_html=True)
                        st.markdown(f"<span style='color:red'>Prédit :</span> {pred_label}", unsafe_allow_html=True)
                        st.markdown(f"**Probabilité :** {confidence:.2f}", unsafe_allow_html=True)
                        if pred_label == selected_class:
                            st.success("**Bonne Prédiction** ✅")
                        else:
                            st.error("**Erreur** ❌")

                        heatmap, heatmap_colored = gradCAM_custom(model, img_batch)
                        if heatmap is not None:
                            overlay = cv2.addWeighted(img_resized, 0.6, heatmap_colored, 0.4, 0)
                            st.image([img_resized, heatmap_colored, overlay],
                                     caption=["Originale", "Heatmap", "Superposée"],
                                     width=250)
        else:
            st.warning(f"Aucune image trouvée pour la classe '{selected_class}'.")
