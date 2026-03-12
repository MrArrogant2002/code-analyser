import os
import pickle
from datetime import datetime

from config import MODELS, CHUNKS_FILE
from indexer import clone_repo, get_c_files, load_files, build_dataset, create_embeddings, build_vector_db
from retriever import retrieve_chunks
from model_loader import load_llm
from inference import explain_code


def save_result(model_key, model_id, question, top_filepath, top_code, explanation):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"result_{timestamp}.md"
    lines = [
        f"# CodeBrain Result",
        f"",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model:** {model_key} — `{model_id}`",
        f"",
        f"---",
        f"",
        f"## Question",
        f"",
        f"{question}",
        f"",
        f"---",
        f"",
        f"## Code Snippet",
        f"",
        f"**Source:** `{top_filepath}`",
        f"",
        f"```c",
        top_code.strip(),
        f"```",
        f"",
        f"---",
        f"",
        f"## Explanation",
        f"",
        explanation,
        f"",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nResult saved → {path}")

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
    print("  ask   — query using a specific model")
    print("  exit  — quit\n")

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

                # Retrieve top-3, pass only top-1 to the LLM
                results = retrieve_chunks(question, embed_model, index, chunks, meta)
                top_filepath, top_code = results[0]

                explanation = explain_code(question, top_code, tokenizer, llm)

                # ── Output ────────────────────────────────────────────────
                print(f"\nQuestion: {question}")
                print(f"\nCode snippet ({top_filepath}):")
                print("-" * 50)
                print(top_code.strip())
                print("-" * 50)
                print(f"\nExplanation:\n{explanation}\n")

                save_result(choice, MODELS[choice], question, top_filepath, top_code, explanation)

            except Exception as e:
                print(f"Error: {e}")

        else:
            print("Unknown command. Use: ask | exit")


if __name__ == "__main__":
    main()
