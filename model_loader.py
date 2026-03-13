import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MODELS

############################################
# LLM LOADER
############################################

def _ensure_torch_xpu_compat():
    """Provide a minimal torch.xpu shim for older PyTorch builds."""
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

def load_llm(model_key):
    _ensure_torch_xpu_compat()

    model_id = MODELS[model_key]
    print(f"\nLoading model: {model_id}")

    if model_key == "deepseek":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    elif model_key == "nemotron":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    else:  # gpt-oss
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

    return tokenizer, model
