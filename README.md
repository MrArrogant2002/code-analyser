# CodeBrain AI

A local **Retrieval-Augmented Generation (RAG)** chatbot that lets you ask natural language questions about a C codebase. It semantically searches code chunks and uses a local LLM to explain what the relevant code does.

---

## How It Works

```
GitHub Repo → C files → Overlapping chunks → Embeddings (cached)
                                                     ↓
User question → Embed question → FAISS search → Top-3 chunks
                                                     ↓
                                          LLM explains each chunk
```

---

## Supported Models

| Key | HuggingFace ID | Notes |
|-----|---------------|-------|
| `deepseek` | `deepseek-ai/deepseek-coder-1.3b-instruct` | Lightweight, fast, good for local use |
| `nemotron` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 30B MoE model, ~3B active params, BF16 |
| `gpt-oss` | `openai/gpt-oss-20b` | OpenAI open-source 20B model |

---

## Setup

### 1. Clone this repository
```bash
git clone https://github.com/MrArrogant2002/code-analyser.git
cd code-analyser
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run
```bash
python codebrain.py
```

On the first run the script will:
- Clone the [TheAlgorithms/C](https://github.com/TheAlgorithms/C) repository into `repo/`
- Compute and cache embeddings → `embeddings_cache.pkl`
- Build and cache the FAISS index → `faiss.index`
- Cache the code chunks → `chunks_cache.pkl`

Subsequent runs load everything from cache — startup is much faster.

---

## Usage

After startup you will see the prompt:

```
CodeBrain> 
```

### Commands

| Command | Description |
|---------|-------------|
| `ask` | Query a single model of your choice |
| `compare` | Run all 3 models on the same question and save a report |
| `exit` | Quit the program |

### Example — Ask

```
CodeBrain> ask

Available models:
  1. deepseek    deepseek-ai/deepseek-coder-1.3b-instruct
  2. nemotron    nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
  3. gpt-oss     openai/gpt-oss-20b

Select model (name or number): 1
Question: How does the binary search implementation work?
```

Output:
```
============================================================
  Model   : deepseek (deepseek-ai/deepseek-coder-1.3b-instruct)
  Question: How does the binary search implementation work?
============================================================

[Snippet 1]  repo/searching/binary_search.c
--------------------------------------------------
<code>
--------------------------------------------------
Explanation:
<clean model output — only the generated answer, not the prompt>
```

### Example — Compare

```
CodeBrain> compare
Question for comparison: What does the bubble sort algorithm do?
```

All three models are loaded sequentially, each generating explanations for the top-3 retrieved snippets. A timestamped Markdown report is saved:

```
Report saved -> report_20260312_143022.md
```

---

## Comparison Report

The generated report (`report_YYYYMMDD_HHMMSS.md`) contains:

- Date and question
- Per-model explanations for each code snippet (with word counts)
- A summary table comparing total and average words per snippet
- A simple "best model" verdict based on output detail

Example summary table:

| Model | HuggingFace ID | Total Words | Avg Words/Snippet |
|-------|---------------|-------------|-------------------|
| **deepseek** | `deepseek-ai/deepseek-coder-1.3b-instruct` | 312 | 104 |
| **nemotron** | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 487 | 162 |
| **gpt-oss** | `openai/gpt-oss-20b` | 451 | 150 |

> Word count is a proxy for detail. Review the full explanations in the report for a qualitative assessment.

---

## Project Structure

```
code-analyzer/
├── codebrain.py          # Main application
├── requirements.txt      # Python dependencies
├── flaws.md              # Documented flaws and fixes
├── README.md             # This file
├── embeddings_cache.pkl  # Auto-generated — cached embeddings
├── chunks_cache.pkl      # Auto-generated — cached code chunks
├── faiss.index           # Auto-generated — FAISS vector index
├── report_*.md           # Auto-generated — comparison reports
└── repo/                 # Auto-cloned — TheAlgorithms/C source
```

---

## Requirements

```
sentence-transformers
faiss-cpu          # or faiss-gpu for CUDA
transformers
torch
gitpython
numpy
```

---

## Fixes Applied

All issues identified during code review have been resolved. See [flaws.md](flaws.md) for full details.

| Fix | Description |
|-----|-------------|
| Overlapping chunks | 500-char windows with 100-char overlap — no more mid-function cuts |
| Dynamic prompt | Removed hard-coded BMS framing; prompt adapts to any question |
| Embedding cache | Embeddings, FAISS index, and chunks saved to disk after first run |
| Exception handling | Replaced bare `except: pass` with specific `(OSError, UnicodeDecodeError)` |
| Chunk deduplication | Duplicate FAISS results filtered before LLM calls |
| Clean LLM output | Only newly generated tokens decoded — prompt no longer echoed in output |
| Better generation | `max_new_tokens=512`, `temperature=0.2`, `repetition_penalty=1.1` |

---

## Notes

- Large models (`nemotron`, `gpt-oss`) require significant VRAM (16 GB+ recommended). In `compare` mode, each model is unloaded from VRAM before the next is loaded.
- If you update the repo, delete the cache files (`*.pkl`, `faiss.index`) to force a fresh embedding run.
- The `deepseek` model is the most practical for CPU or low-VRAM setups.
