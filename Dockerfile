# Image de base légère
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer dépendances système (OpenCV a besoin de libGL, libglib et autres libs basiques)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copier uniquement requirements.txt d'abord (optimisation du cache Docker)
COPY requirements.txt /app/

# Mettre à jour pip et installer les dépendances Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copier ton projet
COPY . /app

# Exposer le port Streamlit
EXPOSE 8501

# Commande par défaut pour lancer ton app
CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]

