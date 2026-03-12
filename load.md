# Model Loading Guide — CodeBrain AI

This guide covers everything needed to load all three models cleanly on your machine.

---

## Models

| Key | HuggingFace ID | Architecture | VRAM Required |
|-----|---------------|--------------|---------------|
| `deepseek` | `deepseek-ai/deepseek-coder-1.3b-instruct` | Standard transformer | ~3 GB |
| `nemotron` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | Mamba/SSM hybrid | ~16 GB |
| `gpt-oss` | `openai/gpt-oss-20b` | Standard transformer | ~40 GB |

---

## Step 1 — System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Linux / WSL2 on Windows | Ubuntu 22.04 / WSL2 |
| GPU | NVIDIA GPU with CUDA support | NVIDIA A100 / RTX 4090 |
| VRAM | 16 GB (nemotron only) | 40 GB+ (all models) |
| CUDA Toolkit | 11.8 | 12.1+ |
| Python | 3.10 | 3.11 |

> **Windows users:** `nemotron` requires `mamba-ssm` which must be compiled from source. This only works on Linux or WSL2 — not on native Windows PowerShell/CMD.

---

## Step 2 — Verify CUDA is Available

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Expected output:
```
True
12.1   ← must match your CUDA toolkit version
```

If `False` is printed, your PyTorch installation does not see your GPU. Either CUDA drivers are missing or you installed the CPU-only version of PyTorch.

---

## Step 3 — Install Base Dependencies

```bash
pip install -r requirements.txt
```

This installs `transformers`, `torch`, `sentence-transformers`, `faiss-cpu`, `gitpython`, `accelerate`, and related packages.

---

## Step 4 — Install `mamba-ssm` (required for `nemotron` only)

`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` uses a Mamba/SSM hybrid architecture. Its custom model code requires two compiled CUDA extension packages that are **not in `requirements.txt`** because they must be built against your local CUDA version.

### 4a — Install `causal-conv1d` first

```bash
pip install causal-conv1d
```

If the pip wheel fails to match your CUDA version, build from source:

```bash
pip install git+https://github.com/Dao-AILab/causal-conv1d.git
```

### 4b — Install `mamba-ssm`

```bash
pip install mamba-ssm
```

If the pip wheel fails, build from source:

```bash
pip install git+https://github.com/state-spaces/mamba.git
```

### 4c — Verify installation

```bash
python -c "import mamba_ssm; import causal_conv1d; print('OK')"
```

Expected output:
```
OK
```

If this fails, the build did not compile correctly. Common causes:
- Mismatched CUDA toolkit vs PyTorch CUDA version
- Missing `nvcc` compiler — install with `sudo apt install nvidia-cuda-toolkit`
- Missing C++ compiler — install with `sudo apt install build-essential`

---

## Step 5 — Log in to HuggingFace

Some models (particularly `gpt-oss` and `nemotron`) may require accepting their license on HuggingFace and authenticating locally.

```bash
pip install huggingface-hub
huggingface-cli login
```

Paste your HuggingFace access token when prompted. Generate one at:
`https://huggingface.co/settings/tokens`

---

## Step 6 — Accept Model License Agreements

Visit each model page and click **"Agree and access repository"** if a gated license is shown:

- `deepseek-ai/deepseek-coder-1.3b-instruct` — open, no gate
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` — check for NVIDIA license gate
- `openai/gpt-oss-20b` — check for OpenAI license gate

If you skip this, `from_pretrained` will return a `403 Forbidden` error.

---

## Step 7 — Run the Application

```bash
python main.py
```

On first run:
- The [TheAlgorithms/C](https://github.com/TheAlgorithms/C) repo is cloned into `repo/`
- Embeddings are computed and cached → `embeddings_cache.pkl`
- FAISS index is built and cached → `faiss.index`
- Code chunks are cached → `chunks_cache.pkl`

Subsequent runs load everything from cache — startup is fast.

---

## Step 8 — Selecting a Model

At the `CodeBrain>` prompt:

```
CodeBrain> ask

Available models:
  1. deepseek    deepseek-ai/deepseek-coder-1.3b-instruct
  2. nemotron    nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
  3. gpt-oss     openai/gpt-oss-20b

Select model (name or number): 2
```

The model is downloaded from HuggingFace on first use and cached in `~/.cache/huggingface/hub/`.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `mamba-ssm is required but cannot be imported` | `mamba-ssm` not installed | Follow Step 4 |
| `torch.cuda.is_available()` returns `False` | CPU-only PyTorch or missing drivers | Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `403 Forbidden` during model download | License not accepted / not logged in | Follow Steps 5 and 6 |
| `OutOfMemoryError` / `CUDA out of memory` | Insufficient VRAM | Use `deepseek` (3 GB) or reduce other GPU workloads |
| `nvcc not found` when building from source | CUDA toolkit not installed | `sudo apt install nvidia-cuda-toolkit` |

---

## Cache Management

To force a fresh embedding run (e.g. after updating the target repo):

```bash
rm embeddings_cache.pkl chunks_cache.pkl faiss.index
python main.py
```

Model weights are cached separately in `~/.cache/huggingface/hub/` and are not affected.
