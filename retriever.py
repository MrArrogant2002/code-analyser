import numpy as np

############################################
# SEMANTIC RETRIEVAL
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
