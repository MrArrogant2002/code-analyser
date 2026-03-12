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
