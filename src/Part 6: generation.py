 # I used the colab to conduct the pipeline due to the configuration limits of my PC
from google.colab import userdata
import requests
import json

SILICONFLOW_API_KEY = userdata.get('SILICONFLOW_API_KEY')   # we call the LLM via SiliconFlow API

import os

index_dir = "/content/faiss_indices"

# Read the indexes of all dimensions
faiss_indices = {}
for fname in os.listdir(index_dir):
    if fname.endswith(".index"):
        dim = int(fname.split("dim")[1].split(".")[0])
        index = faiss.read_index(os.path.join(index_dir, fname))
        meta_csv = os.path.join(index_dir, f"faiss_index_dim{dim}_meta.csv")
        meta_df = pd.read_csv(meta_csv)
        faiss_indices[dim] = {"index": index, "meta": meta_df}

#  nearest neighbor search
def retrieve_chunks(query_embedding, top_k=5):
    results = []
    query_vec = query_embedding.astype('float32').reshape(1, -1)
    dim = query_vec.shape[1]

    if dim not in faiss_indices:
        print(f" ! did NOT find the corresponding FAISS indexes: {dim}")
        return results

    index = faiss_indices[dim]["index"]
    meta_df = faiss_indices[dim]["meta"]

    D, I = index.search(query_vec, top_k)
    for idx in I[0]:
        row = meta_df.iloc[idx]
        results.append({
            "page": row["page"],
            "chunk_type": row["chunk_type"],
            "content_preview": row["content_preview"],
            "has_image": row["has_image"]
        })
    return results


# The LLM I chose is Qwen2.5-VL-7B-Instruct 

def ask_qwen_rag(question, retrieved_chunks, temperature=0.2, max_tokens=800):
    """
    question: user query
    retrieved_chunks: list of dict, each dict contains the chunk info
    """
    # System prompt
    system_prompt = (
        "You are a multimodal RAG assistant capable of answering questions based on the content of provided PDF documents (text, tables, images)."
        "Please generate concise and accurate answers based on retrieved content—do not fabricate information."
    )

    # context content
    context_texts = []
    for c in retrieved_chunks:
        snippet = f"[Page {c['page']}] {c['chunk_type']}: {c['content_preview']}"
        context_texts.append(snippet)

    user_prompt = f"Doc. content:\n{chr(10).join(context_texts)}\n\n question: {question}\n please answer in detail:"

    payload = {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }

    api_url = "https://api.siliconflow.com/v1/chat/completions"
    resp = requests.post(api_url, headers=headers, json=payload)

    if resp.status_code != 200:
        raise ValueError(f"API error: {resp.text}")

    return resp.json()["choices"][0]["message"]["content"]
