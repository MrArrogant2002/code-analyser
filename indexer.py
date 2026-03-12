import os
import pickle

import numpy as np
import faiss
from git import Repo
from sentence_transformers import SentenceTransformer

from config import REPO_URL, REPO_DIR, CACHE_FILE, INDEX_FILE

############################################
# REPOSITORY
############################################

def clone_repo():
    if not os.path.exists(REPO_DIR):
        print("Cloning repository...")
        Repo.clone_from(REPO_URL, REPO_DIR)
    else:
        print("Repository already exists.")

############################################
# FILE DISCOVERY
############################################

def get_c_files():
    files = []
    for root, _, fs in os.walk(REPO_DIR):
        for f in fs:
            if f.endswith(".c") or f.endswith(".h"):
                files.append(os.path.join(root, f))
    return files

############################################
# FILE LOADING
############################################

def load_files(files):
    docs = []
    for f in files:
        try:
            with open(f, "r", encoding="utf8", errors="ignore") as fh:
                docs.append((f, fh.read()))
        except (OSError, UnicodeDecodeError) as e:
            print(f"Warning: Skipping {f}: {e}")
    return docs

############################################
# CHUNKING — overlapping windows
############################################

def split_code(text, size=500, overlap=100):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks

############################################
# DATASET BUILDER
############################################

def build_dataset(docs):
    chunks, meta = [], []
    for filepath, code in docs:
        for part in split_code(code):
            chunks.append(part)
            meta.append(filepath)
    return chunks, meta

############################################
# EMBEDDINGS — disk-cached
############################################

def create_embeddings(chunks):
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    if os.path.exists(CACHE_FILE):
        print("Loading cached embeddings...")
        with open(CACHE_FILE, "rb") as f:
            embeddings = pickle.load(f)
    else:
        print("Computing embeddings (this may take a while)...")
        embeddings = embed_model.encode(chunks, show_progress_bar=True)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(embeddings, f)
        print("Embeddings cached to disk.")
    return embed_model, embeddings

############################################
# VECTOR DATABASE — disk-cached
############################################

def build_vector_db(embeddings):
    if os.path.exists(INDEX_FILE):
        print("Loading cached FAISS index...")
        return faiss.read_index(INDEX_FILE)
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, INDEX_FILE)
    return index
