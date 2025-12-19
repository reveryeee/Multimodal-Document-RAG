!pip install sentence-transformers -q
!pip install transformers -q
!pip install torch torchvision -q
!pip install ftfy regex -q
!pip install einops -q
!pip install pillow -q

import torch
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

# embedding for text & table with Sentence Transformer
text_model = SentenceTransformer('all-MiniLM-L6-v2')
text_model.eval()

# embedding for IMG with CLIP
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model.eval()

# # Note: Regarding the issue of mismatched dimensions in different vector spaces (since we have different chunk types and 
# # after embedding we have different vector space dimension), 
# # we create multiple FAISS indexes, each corresponding to one vector dimension.

embeddings = []

for idx, row in df_chunks.iterrows():
    chunk_type = row['chunk_type']
    chunk_embedding = None

    # 1. for text/table
    if chunk_type in ['text', 'table', 'text+table']:
        text = row.get('content', "")
        if text is None:
            text = ""
        text = str(text)
        # 文本 embedding
        chunk_embedding = text_model.encode(text, convert_to_tensor=True)

    # 2. for IMG
    elif chunk_type == 'image':
        img = row.get('image', None)
        if img is None or not isinstance(img, Image.Image):
            print(f"! Page {row['page']} image missing, idx={idx}")
            continue
        inputs = clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            chunk_embedding = clip_model.get_image_features(**inputs)
        chunk_embedding = chunk_embedding / chunk_embedding.norm(p=2, dim=-1, keepdim=True)

    # 3. for text+IMG
    elif chunk_type == 'text+image':
        text = row.get('content', "")
        if text is None:
            text = ""
        text = str(text)
        text_emb = text_model.encode(text, convert_to_tensor=True)

        img = row.get('image', None)
        if img is None or not isinstance(img, Image.Image):
            print(f"! Page {row['page']} image missing for text+image, idx={idx}")
            continue
        inputs = clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            img_emb = clip_model.get_image_features(**inputs)
        img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)

        # We concatenate the text vector and IMG vector, so the vector space will also be the sum of them
        chunk_embedding = torch.cat([text_emb, img_emb.squeeze(0)], dim=0)

    else:
        print(f"! unknown chunk_type: {chunk_type}, idx={idx}")
        continue

    embeddings.append(chunk_embedding)


# convert to a NumPy array for easier retrieval later
embeddings_np = [e.cpu().numpy() for e in embeddings]
