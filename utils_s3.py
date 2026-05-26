import boto3
import streamlit as st
from botocore.exceptions import ClientError
from io import BytesIO
import os

# Configuration S3
@st.cache_resource
def get_s3_client():
    """Initialise et retourne le client S3"""
    return boto3.client(
        's3',
        aws_access_key_id=st.secrets["aws"]["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets["aws"]["AWS_DEFAULT_REGION"]
    )

def get_bucket_name():
    """Retourne le nom du bucket"""
    return st.secrets["aws"]["BUCKET_NAME"]

# Fonctions de lecture
def read_file_from_s3(s3_key):
    """Lit un fichier depuis S3 et retourne son contenu en bytes"""
    try:
        s3 = get_s3_client()
        bucket = get_bucket_name()
        response = s3.get_object(Bucket=bucket, Key=s3_key)
        return response['Body'].read()
    except ClientError as e:
        st.error(f"Erreur lors de la lecture de {s3_key}: {e}")
        return None

def read_image_from_s3(s3_key):
    """Lit une image depuis S3 et retourne un objet BytesIO"""
    content = read_file_from_s3(s3_key)
    if content:
        return BytesIO(content)
    return None

def list_s3_files(prefix="", suffix=""):
    """Liste les fichiers dans S3 avec un préfixe et suffixe donnés (gère la pagination)"""
    try:
        s3 = get_s3_client()
        bucket = get_bucket_name()

        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

        all_files = []
        for page in page_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if suffix:
                        if key.lower().endswith(suffix.lower()):
                            all_files.append(key)
                    else:
                        all_files.append(key)

        return all_files

    except ClientError as e:
        st.error(f"Erreur lors du listage des fichiers: {e}")
        return []

def file_exists_in_s3(s3_key):
    """Vérifie si un fichier existe dans S3"""
    try:
        s3 = get_s3_client()
        bucket = get_bucket_name()
        s3.head_object(Bucket=bucket, Key=s3_key)
        return True
    except ClientError:
        return False

# Fonctions d'écriture
def write_file_to_s3(s3_key, content):
    """Écrit un fichier dans S3"""
    try:
        s3 = get_s3_client()
        bucket = get_bucket_name()
        
        if isinstance(content, str):
            content = content.encode('utf-8')
        
        s3.put_object(Bucket=bucket, Key=s3_key, Body=content)
        return True
    except ClientError as e:
        st.error(f"Erreur lors de l'écriture de {s3_key}: {e}")
        return False

def upload_file_to_s3(local_path, s3_key):
    """Upload un fichier local vers S3"""
    try:
        s3 = get_s3_client()
        bucket = get_bucket_name()
        s3.upload_file(local_path, bucket, s3_key)
        return True
    except ClientError as e:
        st.error(f"Erreur lors de l'upload de {local_path}: {e}")
        return False

def download_file_from_s3(s3_key, local_path):
    """Télécharge un fichier depuis S3 vers le disque local"""
    try:
        s3 = get_s3_client()
        bucket = get_bucket_name()
        s3.download_file(bucket, s3_key, local_path)
        return True
    except ClientError as e:
        st.error(f"Erreur lors du téléchargement de {s3_key}: {e}")
        return False

def walk_s3_directory(prefix):
    """
    Simule os.walk pour S3
    Retourne (prefix, folders, files) pour chaque "niveau"
    """
    s3 = get_s3_client()
    bucket = get_bucket_name()
    
    paginator = s3.get_paginator('list_objects_v2')
    
    folders = set()
    files = []
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
        # Sous-dossiers
        if 'CommonPrefixes' in page:
            for cp in page['CommonPrefixes']:
                folder_name = cp['Prefix'].rstrip('/')
                folders.add(folder_name)
        
        # Fichiers
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if not key.endswith('/'):  # Exclure les "dossiers" vides
                    files.append(key)
    
    return list(folders), files