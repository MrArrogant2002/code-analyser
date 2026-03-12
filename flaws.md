# CodeBrain AI — Flaws & Fixes

## Summary Table

| # | Flaw | Severity | Fix Complexity |
|---|------|----------|----------------|
| 1 | Naive character chunking | High | Medium |
| 2 | Hard-coded BMS prompt | High | Low |
| 3 | No embedding cache | Medium | Low |
| 4 | Silent exception swallowing | Medium | Low |
| 5 | No chunk deduplication | Low | Low |
| 6 | Full prompt decoded in output | Medium | Low |
| 7 | Poor generation parameters | Low | Low |
| 8 | Bare `except` anti-pattern | Medium | Low |

---

## Flaw 1 — Naive Character-Level Chunking

**Location:** `split_code()`

**Problem:**
`split_code` splits every 500 characters blindly, cutting across function bodies, mid-variable-name, or mid-comment. This destroys semantic coherence — a retrieved chunk may be meaningless on its own.

**Current Code:**
```python
def split_code(text, size=500):
    chunks = []
    for i in range(0, len(text), size):
        chunks.append(text[i:i+size])
    return chunks
```

**Fix A — Split by function boundaries using regex:**
```python
import re

def split_code(text, size=500):
    # Split on function definitions as natural boundaries
    functions = re.split(r'(?=\n[a-zA-Z_][^\n]*\(.*\)\s*\{)', text)
    chunks = []
    for fn in functions:
        if len(fn) <= size:
            chunks.append(fn)
        else:
            # Fall back to line-aware splits for large functions
            for i in range(0, len(fn), size):
                chunks.append(fn[i:i+size])
    return chunks
```

**Fix B — Use overlapping chunks** to avoid losing context at boundaries:
```python
def split_code(text, size=500, overlap=100):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks
```

---

## Flaw 2 — Hard-Coded BMS Prompt

**Location:** `explain_code()`

**Problem:**
The LLM prompt is permanently framed as *"diagnose BMS (Battery Management System) logic"* regardless of what's actually being asked. If you ask about sorting algorithms or graph traversal, the LLM will still try to analyze it for battery management — producing irrelevant or confused output.

**Current Code:**
```python
def explain_code(question, code, tokenizer, model):
    prompt = f"""
You are an embedded systems expert.

Question:
{question}

Code:
{code}

Explain what this code does and diagnose potential BMS logic.
"""
```

**Fix — Make the prompt general and dynamic:**
```python
def explain_code(question, code, tokenizer, model):
    prompt = f"""You are an expert C programmer and software analyst.

A user asked: {question}

Here is a relevant C code snippet:
{code}

Explain clearly what this code does and how it relates to the question.
"""
```

---

## Flaw 3 — No Embedding Cache (Full Recompute Every Run)

**Location:** `create_embeddings()`, `build_vector_db()`

**Problem:**
Every time you run the script, it re-encodes ALL code chunks from scratch. For a large repo this can take several minutes and nothing is saved between runs.

**Current Code:**
```python
def create_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return model, embeddings

def build_vector_db(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index
```

**Fix — Persist embeddings and FAISS index to disk:**
```python
import pickle

CACHE_FILE = "embeddings_cache.pkl"
INDEX_FILE = "faiss.index"

def create_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    if os.path.exists(CACHE_FILE):
        print("Loading cached embeddings...")
        with open(CACHE_FILE, "rb") as f:
            embeddings = pickle.load(f)
    else:
        print("Computing embeddings...")
        embeddings = model.encode(chunks, show_progress_bar=True)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(embeddings, f)

    return model, embeddings

def build_vector_db(embeddings):
    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, INDEX_FILE)
    return index
```

---

## Flaw 4 — Silent Exception Swallowing

**Location:** `load_files()`

**Problem:**
The bare `except: pass` silently drops any file that fails to load — including files with real issues like encoding problems or permission errors. You'd never know a file was skipped.

**Current Code:**
```python
        try:
            with open(f, "r", encoding="utf8", errors="ignore") as file:
                docs.append((f, file.read()))
        except:
            pass
```

**Fix — Log failures explicitly:**
```python
        try:
            with open(f, "r", encoding="utf8", errors="ignore") as file:
                docs.append((f, file.read()))
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")
```

---

## Flaw 5 — No Deduplication of Retrieved Chunks

**Location:** `query()`

**Problem:**
FAISS can return the same chunk multiple times (or near-duplicate chunks from the same file), causing the LLM to be called redundantly and producing repetitive output.

**Current Code:**
```python
    for idx in I[0]:
        code = chunks[idx]
        # ... process chunk
```

**Fix — Deduplicate by index before processing:**
```python
    seen = set()
    for idx in I[0]:
        if idx in seen:
            continue
        seen.add(idx)
        code = chunks[idx]
        # ... process chunk
```

---

## Flaw 6 — Full Prompt Decoded in Output

**Location:** `explain_code()`

**Problem:**
`tokenizer.decode(outputs[0], ...)` decodes the **entire sequence** — including the input prompt. The printed "answer" contains the full prompt text followed by the generated response, making the output noisy and confusing.

**Current Code:**
```python
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
```

**Fix — Decode only the newly generated tokens:**
```python
    outputs = model.generate(**inputs, max_new_tokens=200)
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return answer
```

---

## Flaw 7 — Poor Generation Parameters

**Location:** `explain_code()`

**Problem:**
`max_new_tokens=200` is too short for complex code explanations. There's no `temperature` or `repetition_penalty`, so the model can produce repetitive or truncated output.

**Current Code:**
```python
    outputs = model.generate(
        **inputs,
        max_new_tokens=200
    )
```

**Fix — Add temperature, sampling, and repetition penalty:**
```python
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,        # more focused/deterministic for code
        do_sample=True,
        repetition_penalty=1.1
    )
```

---

## Flaw 8 — Bare `except` is a Security/Reliability Anti-Pattern

**Location:** `load_files()`

**Problem:**
`except: pass` catches *everything* — including `KeyboardInterrupt`, `SystemExit`, and `MemoryError`. This is dangerous: you cannot even Ctrl+C out of a file-loading loop if something hangs.

**Current Code:**
```python
        except:
            pass
```

**Fix — Always catch specific exceptions:**
```python
        except (OSError, UnicodeDecodeError) as e:
            print(f"Skipping {f}: {e}")
```

---

*Flaws identified against codebrain.py on 2026-03-11.*
