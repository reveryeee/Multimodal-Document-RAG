!pip install faiss-cpu -q
!pip install numpy pandas -q

import faiss
import numpy as np
import pandas as pd
import os
from collections import defaultdict

# 1.prepare the vector
vector_infos = []
for idx, row in df_chunks.iterrows():
    vec = embeddings_np[idx]  # numpy array
    if vec is None:
        continue
    vec = np.array(vec).flatten()
    if vec.size <= 1:
        continue  # we ignore the abnormal vectors

    content_preview = ""
    if row["content"] is not None:
        content_preview = str(row["content"])[:200]

    vector_infos.append({
        "vector": vec,
        "page": row.get("page", -1),
        "chunk_type": row.get("chunk_type", "unknown"),
        "content": content_preview,
        "has_image": "image" in row and row["image"] is not None
    })

print(f"the number of effective vectors is: {len(vector_infos)}")

# 2. we group different vectors by different vector space size
dim_groups = defaultdict(list)
for info in vector_infos:
    dim = len(info['vector'])
    dim_groups[dim].append(info)


# FAISS indexes for different vector types
index_dir = "/content/faiss_indices"
os.makedirs(index_dir, exist_ok=True)

faiss_index_map = {}  # we save the info corresponding to each dimension index

for dim, infos in dim_groups.items():
    vectors = np.stack([info['vector'] for info in infos]).astype('float32')
    print(f"\n=== we create the FAISS index with dimension {dim} , number of chunk: {vectors.shape[0]} ===")

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # save the indexes
    index_path = os.path.join(index_dir, f"faiss_index_dim{dim}.index")
    faiss.write_index(index, index_path)
    print(f"save the index to: {index_path}")

    # save mapping info DataFrame
    meta_df = pd.DataFrame({
        "page": [info['page'] for info in infos],
        "chunk_type": [info['chunk_type'] for info in infos],
        "content_preview": [info['content'] for info in infos],
        "has_image": [info['has_image'] for info in infos]
    })
    meta_csv_path = os.path.join(index_dir, f"faiss_index_dim{dim}_meta.csv")
    meta_df.to_csv(meta_csv_path, index=False)
    print(f"Index mapping table saved to: {meta_csv_path}")

    # save to DIC.
    faiss_index_map[dim] = {
        "index": index,
        "meta": meta_df,
        "index_path": index_path,
        "meta_path": meta_csv_path
    }
