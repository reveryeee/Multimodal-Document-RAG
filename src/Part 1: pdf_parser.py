!pip install PyMuPDF pdfplumber pillow pandas opencv-python camelot-py -q

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import io
import pandas as pd
import camelot
import os

def extract_pdf_multimodal(pdf_path):
    """
    Extract the multimodal content in the PDF: Text/IMG/Table
    Return a unified strucutre in each page:
    {
        "page": int,
        "text": str,
        "tables": [DataFrame],
        "images": [PIL.Image]
    }
    """
    document = []

    
    doc = fitz.open(pdf_path)
    with pdfplumber.open(pdf_path) as pdf_plumber:

        # Extract the table via camelot
        # flavor='lattice' → Based on table lines.   flavor='stream' → Based on text streams
        camelot_tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')

        # Group by the page number
        tables_by_page = {}
        for table in camelot_tables:
            page_num = table.page  # camelot start with page 1
            if page_num - 1 not in tables_by_page:
                tables_by_page[page_num - 1] = []
            tables_by_page[page_num - 1].append(table.df)

        # Iterate the whole PDF
        for page_number in range(len(doc)):
            # Extract our text content 
            page_text = pdf_plumber.pages[page_number].extract_text() or ""

            # Extract our IMG
            page = doc[page_number]
            images = []
            for img_index, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                images.append(image)

            # Extract the table
            page_tables = tables_by_page.get(page_number, [])

            # we save all of the content to a unified structure
            document.append({
                "page": page_number,
                "text": page_text,
                "tables": page_tables,
                "images": images
            })

    return document

