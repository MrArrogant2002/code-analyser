import os
import pickle
from datetime import datetime

import numpy as np
import torch
from git import Repo
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False


############################################
# CONFIGURATION
############################################

REPO_URL = "https://github.com/TheAlgorithms/C.git"
REPO_DIR = "repo"
CACHE_FILE = "embeddings_cache.pkl"
INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks_cache.pkl"
EMBED_MODEL_ID = "intfloat/e5-small-v2"

MODELS = {
    "deepseek": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "nemotron": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "gpt-oss": "openai/gpt-oss-20b",
}


############################################
# REPOSITORY + INDEXING
############################################


class HFTextEmbedder:
    """Lightweight sentence embedder using plain Transformers."""

    def __init__(self, model_id=EMBED_MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def encode(self, texts, show_progress_bar=False, batch_size=32):
        del show_progress_bar
        vectors = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                # E5 works best with explicit prefixes.
                batch = [f"passage: {t}" for t in batch]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                out = self.model(**encoded)
                emb = self._mean_pool(out.last_hidden_state, encoded["attention_mask"])
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                vectors.append(emb.cpu().numpy())
        return np.vstack(vectors).astype(np.float32)

    def encode_query(self, text):
        with torch.no_grad():
            encoded = self.tokenizer(
                [f"query: {text}"],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**encoded)
            emb = self._mean_pool(out.last_hidden_state, encoded["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            return emb.cpu().numpy().astype(np.float32)

def clone_repo():
    if not os.path.exists(REPO_DIR):
        print("Cloning repository...")
        Repo.clone_from(REPO_URL, REPO_DIR)
    else:
        print("Repository already exists.")


def get_c_files():
    files = []
    for root, _, fs in os.walk(REPO_DIR):
        for file_name in fs:
            if file_name.endswith(".c") or file_name.endswith(".h"):
                files.append(os.path.join(root, file_name))
    return files


def load_files(files):
    docs = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                docs.append((path, handle.read()))
        except (OSError, UnicodeDecodeError) as exc:
            print(f"Warning: Skipping {path}: {exc}")
    return docs


def split_code(text, size=500, overlap=100):
    chunks = []
    step = max(1, size - overlap)
    idx = 0
    while idx < len(text):
        chunks.append(text[idx : idx + size])
        idx += step
    return chunks


def build_dataset(docs):
    chunks, meta = [], []
    for filepath, code in docs:
        for part in split_code(code):
            chunks.append(part)
            meta.append(filepath)
    return chunks, meta


def create_embeddings(chunks):
    embed_model = HFTextEmbedder(EMBED_MODEL_ID)
    if os.path.exists(CACHE_FILE):
        print("Loading cached embeddings...")
        with open(CACHE_FILE, "rb") as handle:
            embeddings = pickle.load(handle)
    else:
        print("Computing embeddings (this may take a while)...")
        embeddings = embed_model.encode(chunks, show_progress_bar=True)
        with open(CACHE_FILE, "wb") as handle:
            pickle.dump(embeddings, handle)
        print("Embeddings cached to disk.")
    return embed_model, embeddings


def build_vector_db(embeddings):
    emb = np.asarray(embeddings, dtype=np.float32)

    if HAS_FAISS:
        if os.path.exists(INDEX_FILE):
            print("Loading cached FAISS index...")
            return faiss.read_index(INDEX_FILE)

        print("Building FAISS index...")
        index = faiss.IndexFlatL2(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, INDEX_FILE)
        return index

    print("FAISS not found. Falling back to NumPy retrieval.")
    print("Install with: pip install -U faiss-cpu")
    return emb


def retrieve_chunks(question, embed_model, index, chunks, meta, top_k=3):
    q_vec = embed_model.encode_query(question)
    q_arr = np.asarray(q_vec, dtype=np.float32)

    if HAS_FAISS:
        _, raw_indices = index.search(q_arr, top_k)
        indices = raw_indices[0]
    else:
        # NumPy fallback path: L2 distance ranking.
        distances = np.sum((index - q_arr[0]) ** 2, axis=1)
        indices = np.argsort(distances)[:top_k]

    results = []
    seen = set()
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        results.append((meta[idx], chunks[idx]))
    return results


############################################
# LLM LOADING + INFERENCE
############################################

def _ensure_torch_xpu_compat():
    """Provide a minimal torch.xpu shim for environments without XPU."""
    if hasattr(torch, "xpu"):
        return

    class _XPUCompat:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

    torch.xpu = _XPUCompat()


def _select_torch_dtype():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        # Use bf16 on newer GPUs, fp16 otherwise.
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def _model_load_kwargs(model_key):
    dtype = _select_torch_dtype()
    kwargs = {
        "torch_dtype": dtype,
    }

    # device_map="auto" is best when GPU is available.
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    else:
        kwargs["low_cpu_mem_usage"] = True

    # Some repositories require remote code execution hooks.
    if model_key in {"nemotron", "gpt-oss"}:
        kwargs["trust_remote_code"] = True

    return kwargs


def _safe_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_llm(model_key):
    _ensure_torch_xpu_compat()

    model_id = MODELS[model_key]
    print(f"\nLoading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **_model_load_kwargs(model_key),
    )

    return tokenizer, model


def _build_prompt(question, code):
    return (
        "You are an expert C programmer and software analyst.\n\n"
        f"User question: {question}\n\n"
        "Relevant C code snippet:\n"
        f"{code.strip()}\n\n"
        "Explain clearly:\n"
        "1) What this code does\n"
        "2) Important logic details\n"
        "3) Potential issues or edge cases\n"
        "4) How it relates to the user question"
    )


def explain_code(question, code, tokenizer, model):
    messages = [{"role": "user", "content": _build_prompt(question, code)}]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        fallback_prompt = _build_prompt(question, code)
        inputs = tokenizer(fallback_prompt, return_tensors="pt")

    device = _safe_model_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )

    input_len = inputs["input_ids"].shape[-1]
    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def print_result_block(model_key, model_id, question, filepath, code, explanation, rank):
    print("\n" + "=" * 80)
    print(f"Model: {model_key}  |  {model_id}")
    print(f"Result Rank: {rank}")
    print("-" * 80)
    print(f"Question:\n{question}\n")
    print(f"Source File:\n{filepath}\n")
    print("Code Snippet:\n")
    print(code.strip())
    print("\n" + "-" * 80)
    print("Explanation:\n")
    print(explanation)
    print("=" * 80)


############################################
# OUTPUT + MAIN LOOP
############################################

def save_result(model_key, model_id, question, top_filepath, top_code, explanation):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"result_{timestamp}.md"
    lines = [
        "# CodeBrain Result",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model:** {model_key} - `{model_id}`",
        "",
        "---",
        "",
        "## Question",
        "",
        question,
        "",
        "---",
        "",
        "## Code Snippet",
        "",
        f"**Source:** `{top_filepath}`",
        "",
        "```c",
        top_code.strip(),
        "```",
        "",
        "---",
        "",
        "## Explanation",
        "",
        explanation,
        "",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"\nResult saved -> {path}")


def build_or_load_chunks(docs):
    if os.path.exists(CHUNKS_FILE):
        print("Loading cached chunks...")
        with open(CHUNKS_FILE, "rb") as handle:
            chunks, meta = pickle.load(handle)
    else:
        chunks, meta = build_dataset(docs)
        with open(CHUNKS_FILE, "wb") as handle:
            pickle.dump((chunks, meta), handle)
    return chunks, meta


def main():
    clone_repo()

    files = get_c_files()
    docs = load_files(files)
    chunks, meta = build_or_load_chunks(docs)

    embed_model, embeddings = create_embeddings(chunks)
    index = build_vector_db(embeddings)

    print("\n" + "=" * 60)
    print("  CodeBrain AI - Ready")
    print("=" * 60)
    print("\nCommands:")
    print("  ask   - query using a specific model")
    print("  exit  - quit\n")

    model_cache = {}

    while True:
        cmd = input("CodeBrain> ").strip().lower()

        if cmd == "exit":
            print("Goodbye.")
            break

        if cmd != "ask":
            print("Unknown command. Use: ask | exit")
            continue

        print("\nAvailable models:")
        model_keys = list(MODELS.keys())
        for idx, key in enumerate(model_keys, start=1):
            print(f"  {idx}. {key:10s}  {MODELS[key]}")
        print("  all          run all models")

        choice = input("Select model (name or number): ").strip().lower()
        if choice.isdigit() and 1 <= int(choice) <= len(model_keys):
            choice = model_keys[int(choice) - 1]
        if choice != "all" and choice not in MODELS:
            print(f"Unknown model '{choice}'. Options: {', '.join(MODELS)} or all")
            continue

        question = input("Question: ").strip()
        if not question:
            continue

        try:
            results = retrieve_chunks(question, embed_model, index, chunks, meta)
            selected_models = model_keys if choice == "all" else [choice]

            for model_key in selected_models:
                if model_key not in model_cache:
                    model_cache[model_key] = load_llm(model_key)

                tokenizer, llm = model_cache[model_key]

                for rank, (top_filepath, top_code) in enumerate(results, start=1):
                    explanation = explain_code(question, top_code, tokenizer, llm)
                    print_result_block(
                        model_key,
                        MODELS[model_key],
                        question,
                        top_filepath,
                        top_code,
                        explanation,
                        rank,
                    )
                    save_result(
                        model_key,
                        MODELS[model_key],
                        question,
                        top_filepath,
                        top_code,
                        explanation,
                    )

        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()