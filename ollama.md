# Ollama Migration Plan — CodeBrain AI

Switch all three models from HuggingFace `transformers` to local Ollama inference.
No CUDA extensions, no `mamba-ssm`, no tokenizer management — Ollama handles everything.

---

## Hardware Requirements

| Component | `gpt-oss:20b` (21B params) | `nemotron-3-nano:30b` (30B MoE) |
|-----------|---------------------------|----------------------------------|
| Min VRAM | 16 GB (MXFP4 quantized) | ~16 GB (quantized) / 40 GB (BF16) |
| Recommended GPU | NVIDIA RTX 4090 / 5090 (24 GB+) | NVIDIA A100 / H100 / RTX 5090 |
| System RAM | 32 GB+ DDR4/DDR5 | 64 GB+ (if offloading to CPU) |
| Disk Space | ~13 GB | ~20 GB (quantized) / 60 GB (BF16) |
| Apple Silicon | M3 Max ≥ 32 GB Unified Memory | M2/M3 Ultra ≥ 64 GB Unified Memory |

> If your GPU has less VRAM than required, Ollama will automatically offload model layers to
> system RAM. The model will still run but tokens-per-second will drop significantly.

---

## Part 1 — Ollama Installation

### Step 1 — Install Ollama

**Windows:**
Download and run the installer from https://ollama.com/download

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 2 — Verify Ollama is running

```bash
ollama --version
```

Ollama runs a local HTTP server on `http://localhost:11434` by default. Confirm it is active:

```bash
curl http://localhost:11434
```

Expected response: `Ollama is running`

### Step 3 — Pull the models

DeepSeek (lightweight, ~1 GB):
```bash
ollama pull deepseek-coder:1.3b
```

GPT-OSS 20B (~13 GB):
```bash
ollama pull gpt-oss:20b
```

Nemotron-3-Nano 30B (~20 GB quantized):
```bash
ollama pull nemotron-3-nano:30b
```

> Pull without running. This downloads the weights once to your local Ollama model store.
> Models are stored in `~/.ollama/models/` on Linux/macOS or `C:\Users\<you>\.ollama\models\` on Windows.

### Step 4 — Verify models are available

```bash
ollama list
```

Expected output:
```
NAME                       ID              SIZE    MODIFIED
deepseek-coder:1.3b        ...             776 MB  ...
gpt-oss:20b                ...             13 GB   ...
nemotron-3-nano:30b        ...             20 GB   ...
```

### Step 5 — Install the Ollama Python library

```bash
pip install ollama
```

---

## Part 2 — Code Modification Plan

### Files to change

| File | Change |
|------|--------|
| `config.py` | Replace HuggingFace model IDs with Ollama tags |
| `model_loader.py` | Delete entirely — Ollama needs no loader |
| `inference.py` | Replace `transformers` generate pipeline with `ollama.chat()` |
| `main.py` | Remove `load_llm` import and call; pass only `model_key` to `explain_code` |
| `requirements.txt` | Remove `torch`, `transformers`, `accelerate`, `huggingface-hub`; add `ollama` |

---

### config.py — before vs after

**Before:**
```python
MODELS = {
    "deepseek" : "deepseek-ai/deepseek-coder-1.3b-instruct",
    "nemotron"  : "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "gpt-oss"   : "openai/gpt-oss-20b",
}
```

**After:**
```python
MODELS = {
    "deepseek" : "deepseek-coder:1.3b",
    "nemotron"  : "nemotron-3-nano:30b",
    "gpt-oss"   : "gpt-oss:20b",
}
```

---

### model_loader.py — delete the file

Ollama manages model loading internally. There is no `load_llm()` call needed.
The file can be deleted after migration.

---

### inference.py — full rewrite

**Before** (transformers pipeline — tokenizer + GPU tensors + manual decode):
```python
def explain_code(question, code, tokenizer, model):
    messages = [...]
    inputs = tokenizer.apply_chat_template(...).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, ...)
    input_len = inputs["input_ids"].shape[-1]
    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
```

**After** (Ollama — single function call, no tokenizer, no GPU tensors):
```python
import ollama

def explain_code(question, code, model_tag):
    response = ollama.chat(
        model=model_tag,
        messages=[
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
        ],
    )
    return response["message"]["content"].strip()
```

---

### main.py — changes required

**Remove:**
- `from model_loader import load_llm`
- `tokenizer, llm = load_llm(choice)` call
- `tokenizer` and `llm` arguments passed to `explain_code`

**Change** the `explain_code` call signature:

Before:
```python
tokenizer, llm = load_llm(choice)
...
explanation = explain_code(question, top_code, tokenizer, llm)
```

After:
```python
explanation = explain_code(question, top_code, MODELS[choice])
```

No model loading step. Ollama serves the model in the background — the call goes straight to inference.

---

### requirements.txt — before vs after

**Before:**
```
gitpython==3.1.43
sentence-transformers==2.7.0
faiss-cpu==1.8.0
numpy==1.26.4
torch==2.2.2
transformers>=4.47.0
accelerate>=0.34.0
scikit-learn==1.4.2
scipy==1.13.0
tqdm==4.66.2
huggingface-hub>=0.26.0
```

**After:**
```
gitpython==3.1.43
sentence-transformers==2.7.0
faiss-cpu==1.8.0
numpy==1.26.4
scikit-learn==1.4.2
scipy==1.13.0
tqdm==4.66.2
ollama>=0.2.0
```

Removed: `torch`, `transformers`, `accelerate`, `huggingface-hub`
Added: `ollama`

> `sentence-transformers` still needs `torch` internally for the embedding model
> (`all-MiniLM-L6-v2`). Keep `torch` if `sentence-transformers` pulls it as a dependency,
> but it no longer needs to be pinned to a specific CUDA version.

---

## Part 3 — Verification

After all changes, run:

```bash
python main.py
```

Then at the prompt:
```
CodeBrain> ask
Select model (name or number): 1
Question: How does binary search work?
```

Expected: output prints question → code snippet → explanation, and saves `result_*.md`.

To confirm each model works individually:
```bash
ollama run deepseek-coder:1.3b "say hello"
ollama run gpt-oss:20b "say hello"
ollama run nemotron-3-nano:30b "say hello"
```

---

## Part 4 — Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `connection refused` on `localhost:11434` | Ollama server not running | Run `ollama serve` in a separate terminal |
| `model not found` | Model not pulled yet | Run `ollama pull <model-tag>` |
| Very slow responses | VRAM insufficient, offloading to RAM | Expected — reduce other GPU workloads or use `deepseek-coder:1.3b` |
| `ollama: command not found` | Ollama not installed | Follow Part 1, Step 1 |
| `ModuleNotFoundError: ollama` | Python package not installed | `pip install ollama` |
