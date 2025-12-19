import pandas as pd
from PIL import Image, ImageDraw
import random
import re


MAX_TEXT_CHUNK_SIZE = 300  # MAX. character number in each chunk
MERGE_DISTANCE = 50        # using the distance for measuring the similarity between text&IMG and text&table


# defining the semantic chunking method for the text
def split_text_semantic(text, max_size=MAX_TEXT_CHUNK_SIZE):
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= max_size:
            chunks.append(para)
        else:
            # For excessively long paragraphs, break them at periods or line breaks. 
            sentences = re.split(r'(?<=[。！？\n])', para)
            tmp = ""
            for s in sentences:
                if len(tmp) + len(s) <= max_size:
                    tmp += s
                else:
                    if tmp:
                        chunks.append(tmp)
                    tmp = s
            if tmp:
                chunks.append(tmp)
    return chunks


# random color, for the visulization (if needed)
def random_color():
    return tuple(random.randint(50, 200) for _ in range(3))


# The cross-modal chunk
chunks = []

for page_data in knowledge_data:
    page_num = page_data["page"]
    text = page_data.get("text", "")
    tables = page_data.get("tables", [])
    images = page_data.get("images", [])

    # 1. the pure text chunk 
    if text.strip():
        text_chunks = split_text_semantic(text)
        for t in text_chunks:
            chunks.append({
                "page": page_num,
                "chunk_type": "text",
                "content": t,
                "image": None,
                "table": None,
                "metadata": {"modality": "text"}
            })

    # 2. the pure image chunk
    for img in images:
        chunks.append({
            "page": page_num,
            "chunk_type": "image",
            "content": None,
            "image": img,
            "table": None,
            "metadata": {"modality": "image"}
        })

    # 3. The text + IMG chunk
     # we use the simple logic here, i.e., when the text and IMG exist on the same page, then they are combined as one chunk
    if text.strip() and images:
        for img in images:
            chunks.append({
                "page": page_num,
                "chunk_type": "text+image",
                "content": text,
                "image": img,
                "table": None,
                "metadata": {"modality": "text+image"}
            })

    # 4. pure table chunk
    for tbl in tables:
        chunks.append({
            "page": page_num,
            "chunk_type": "table",
            "content": None,
            "image": None,
            "table": tbl,
            "metadata": {"modality": "table"}
        })

    # 5. text + table chunk
    if text.strip() and tables:
        for tbl in tables:
            chunks.append({
                "page": page_num,
                "chunk_type": "text+table",
                "content": text,
                "image": None,
                "table": tbl,
                "metadata": {"modality": "text+table"}
            })

