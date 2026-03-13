import os
import numpy as np
import faiss
from git import Repo
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

############################################
# REPOSITORY
############################################

REPO_URL = "https://github.com/TheAlgorithms/C.git"
REPO_DIR = "repo"


def clone_repo():

    if not os.path.exists(REPO_DIR):

        print("Cloning repository...")
        Repo.clone_from(REPO_URL, REPO_DIR)

    else:
        print("Repository already exists")


############################################
# GET C FILES
############################################

def get_c_files():

    files = []

    for root, dirs, fs in os.walk(REPO_DIR):

        for f in fs:

            if f.endswith(".c") or f.endswith(".h"):
                files.append(os.path.join(root, f))

    return files


############################################
# LOAD FILE CONTENT
############################################

def load_files(files):

    docs = []

    for f in files:

        try:

            with open(f, "r", encoding="utf8", errors="ignore") as file:

                docs.append((f, file.read()))

        except Exception as e:
            pass

    return docs


############################################
# CHUNK CODE
############################################

def split_code(text, size=500):

    chunks = []

    for i in range(0, len(text), size):

        chunks.append(text[i:i+size])

    return chunks


############################################
# BUILD DATASET
############################################

def build_dataset(docs):

    chunks = []
    meta = []

    for file, code in docs:

        parts = split_code(code)

        for p in parts:

            chunks.append(p)
            meta.append(file)

    return chunks, meta


############################################
# EMBEDDINGS
############################################

def create_embeddings(chunks):

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(chunks)

    return model, embeddings


############################################
# VECTOR DATABASE
############################################

def build_vector_db(embeddings):

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(np.array(embeddings))

    return index


############################################
# LOAD CODE LLM
############################################

def load_llm():

    print("Loading LLM...")

    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-instruct"
    )

    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return tokenizer, model


############################################
# LLM EXPLANATION
############################################

def explain_code(question, code, tokenizer, model):

    prompt = f"""
You are an embedded systems expert.

Question:
{question}

Code:
{code}

Explain what this code does and diagnose potential BMS logic.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


############################################
# QUERY SYSTEM
############################################

def query(question, embed_model, index, chunks, meta, tokenizer, llm):

    q = embed_model.encode([question])

    D, I = index.search(np.array(q), 3)

    for idx in I[0]:

        code = chunks[idx]

        print("\nFile:", meta[idx])
        print("\nCode Snippet:\n")
        print(code)

        print("\nLLM Explanation:\n")

        explanation = explain_code(
            question,
            code,
            tokenizer,
            llm
        )

        print(explanation)

        print("\n---------------------------------\n")


############################################
# MAIN
############################################

def main():

    clone_repo()

    files = get_c_files()

    docs = load_files(files)

    chunks, meta = build_dataset(docs)

    embed_model, embeddings = create_embeddings(chunks)

    index = build_vector_db(embeddings)

    tokenizer, llm = load_llm()

    print("\nCodeBrain AI Ready\n")

    while True:

        q = input("Ask CodeBrain: ")

        if q == "exit":
            break

        query(
            q,
            embed_model,
            index,
            chunks,
            meta,
            tokenizer,
            llm
        )


if __name__ == "__main__":
    main()