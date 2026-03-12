import os
import pickle
from datetime import datetime

import numpy as np
import faiss
from git import Repo
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

############################################
# CONFIGURATION
############################################

REPO_URL    = "https://github.com/TheAlgorithms/C.git"
REPO_DIR    = "repo"
CACHE_FILE  = "embeddings_cache.pkl"
INDEX_FILE  = "faiss.index"
CHUNKS_FILE = "chunks_cache.pkl"

MODELS = {
    "deepseek" : "deepseek-ai/deepseek-coder-1.3b-instruct",
    "nemotron"  : "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "gpt-oss"   : "openai/gpt-oss-20b",
}

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
# GET C FILES
############################################

def get_c_files():
    files = []
    for root, _, fs in os.walk(REPO_DIR):
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
            with open(f, "r", encoding="utf8", errors="ignore") as fh:
                docs.append((f, fh.read()))
        except (OSError, UnicodeDecodeError) as e:
            print(f"Warning: Skipping {f}: {e}")
    return docs

############################################
# CHUNK CODE — overlapping windows
############################################

def split_code(text, size=500, overlap=100):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks

############################################
# BUILD DATASET
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
    model = SentenceTransformer("all-MiniLM-L6-v2")
    if os.path.exists(CACHE_FILE):
        print("Loading cached embeddings...")
        with open(CACHE_FILE, "rb") as f:
            embeddings = pickle.load(f)
    else:
        print("Computing embeddings (this may take a while)...")
        embeddings = model.encode(chunks, show_progress_bar=True)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(embeddings, f)
        print("Embeddings cached to disk.")
    return model, embeddings

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

############################################
# LOAD LLM
############################################

def load_llm(model_key):
    model_id = MODELS[model_key]
    print(f"\nLoading model: {model_id}")
    trust = model_key == "nemotron"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust)
    dtype = torch.bfloat16 if model_key == "nemotron" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=trust,
    )
    return tokenizer, model

############################################
# LLM EXPLANATION — clean output
############################################

def explain_code(question, code, tokenizer, model):
    messages = [
        {
            "role": "user",
            "content": (
                "You are an expert C programmer and software analyst.\n\n"
                f"User question: {question}\n\n"
                "Relevant C code snippet:\n"
                f"{code.strip()}\n\n"
                "Provide a clear and concise explanation of what this code does "
                "and how it relates to the question above."
            ),
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode only newly generated tokens — skip the echoed prompt
    input_len = inputs["input_ids"].shape[-1]
    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

############################################
# RETRIEVE RELEVANT CHUNKS
############################################

def retrieve_chunks(question, embed_model, index, chunks, meta, top_k=3):
    q_vec = embed_model.encode([question])
    _, I = index.search(np.array(q_vec), top_k)
    results, seen = [], set()
    for idx in I[0]:
        if idx not in seen:
            seen.add(idx)
            results.append((meta[idx], chunks[idx]))
    return results

############################################
# SINGLE-MODEL QUERY
############################################

def query(question, embed_model, index, chunks, meta, tokenizer, llm, label="Model"):
    results = retrieve_chunks(question, embed_model, index, chunks, meta)
    print(f"\n{'='*60}")
    print(f"  Model   : {label}")
    print(f"  Question: {question}")
    print(f"{'='*60}")
    for i, (filepath, code) in enumerate(results, 1):
        print(f"\n[Snippet {i}]  {filepath}")
        print("-" * 50)
        print(code.strip())
        print("-" * 50)
        explanation = explain_code(question, code, tokenizer, llm)
        print(f"Explanation:\n{explanation}")
    print()

############################################
# MULTI-MODEL COMPARISON + REPORT
############################################

def run_comparison(question, embed_model, index, chunks, meta):
    results = retrieve_chunks(question, embed_model, index, chunks, meta)

    lines = [
        "# CodeBrain — Model Comparison Report",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Question:** {question}",
        "",
    ]

    scores = {}

    for model_key, model_id in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_key}  ({model_id})")
        print(f"{'='*60}")

        try:
            tokenizer, llm = load_llm(model_key)
        except Exception as e:
            msg = f"Failed to load — {e}"
            print(f"  ERROR: {msg}")
            lines += [f"## {model_key.upper()} — `{model_id}`", f"**Error:** {msg}", ""]
            continue

        lines += [f"## {model_key.upper()} — `{model_id}`", ""]

        total_words = 0
        for i, (filepath, code) in enumerate(results, 1):
            print(f"\n  Snippet {i}: {filepath}")
            explanation = explain_code(question, code, tokenizer, llm)
            wc = len(explanation.split())
            total_words += wc
            print(f"  ({wc} words)\n{explanation}")

            lines += [
                f"### Snippet {i}: `{filepath}`",
                f"**Code:**\n```c\n{code.strip()}\n```",
                f"**Explanation ({wc} words):**",
                explanation,
                "",
            ]

        scores[model_key] = {
            "model_id"   : model_id,
            "total_words": total_words,
            "avg_words"  : total_words // max(len(results), 1),
        }

        # Free VRAM before loading the next model
        del llm, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Summary table ----
    lines += ["---", "## Summary", ""]
    lines += [
        "| Model | HuggingFace ID | Total Words | Avg Words/Snippet |",
        "|-------|---------------|-------------|-------------------|",
    ]
    for key, s in scores.items():
        lines.append(f"| **{key}** | `{s['model_id']}` | {s['total_words']} | {s['avg_words']} |")

    if scores:
        best = max(scores, key=lambda k: scores[k]["total_words"])
        lines += [
            "",
            f"**Best model (most detailed output):** `{best}` — `{scores[best]['model_id']}`",
            "",
            "> Word count is a proxy for detail. "
            "Review the explanations above for a qualitative assessment.",
        ]

    report_path = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, "w", encoding="utf8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved -> {report_path}")
    return report_path

############################################
# MAIN
############################################

def main():
    clone_repo()

    files = get_c_files()
    docs  = load_files(files)

    # Cache chunks so they stay in sync with embeddings
    if os.path.exists(CHUNKS_FILE):
        print("Loading cached chunks...")
        with open(CHUNKS_FILE, "rb") as f:
            chunks, meta = pickle.load(f)
    else:
        chunks, meta = build_dataset(docs)
        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump((chunks, meta), f)

    embed_model, embeddings = create_embeddings(chunks)
    index = build_vector_db(embeddings)

    print("\n" + "=" * 60)
    print("  CodeBrain AI — Ready")
    print("=" * 60)
    print("\nCommands:")
    print("  ask     — query using a specific model")
    print("  compare — run all 3 models and save a comparison report")
    print("  exit    — quit\n")

    while True:
        cmd = input("CodeBrain> ").strip().lower()

        if cmd == "exit":
            print("Goodbye.")
            break

        elif cmd == "ask":
            print("\nAvailable models:")
            model_keys = list(MODELS.keys())
            for i, key in enumerate(model_keys, 1):
                print(f"  {i}. {key:10s}  {MODELS[key]}")
            choice = input("Select model (name or number): ").strip().lower()
            if choice.isdigit() and 1 <= int(choice) <= len(model_keys):
                choice = model_keys[int(choice) - 1]
            if choice not in MODELS:
                print(f"Unknown model '{choice}'. Options: {', '.join(MODELS)}")
                continue
            question = input("Question: ").strip()
            if not question:
                continue
            try:
                tokenizer, llm = load_llm(choice)
                query(question, embed_model, index, chunks, meta,
                      tokenizer, llm, label=f"{choice} ({MODELS[choice]})")
            except Exception as e:
                print(f"Error: {e}")

        elif cmd == "compare":
            question = input("Question for comparison: ").strip()
            if question:
                run_comparison(question, embed_model, index, chunks, meta)

        else:
            print("Unknown command. Use: ask | compare | exit")


if __name__ == "__main__":
    main()
