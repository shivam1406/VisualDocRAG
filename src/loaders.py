
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import io, fitz, pdfplumber, pytesseract, cv2, numpy as np
from PIL import Image
from .utils import clean_text

@dataclass
class ExtractedElement:
    type: str            # 'text' | 'table' | 'image_ocr'
    text: str
    page: int
    extra: Dict[str, Any]

def is_scanned_pdf(pdf_path: str) -> bool:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total = ""
            for p in pdf.pages[:3]:
                total += p.extract_text() or ""
            return len(total.strip()) < 50
    except Exception:
        return True

def ocr_image(img: Image.Image, lang: str = "eng") -> str:
    arr = np.array(img.convert("L"))
    arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    arr = cv2.medianBlur(arr, 3)
    return pytesseract.image_to_string(arr, lang=lang)

def load_pdf(pdf_path: str, ocr_lang: str = "eng") -> List[ExtractedElement]:
    out: List[ExtractedElement] = []
    scanned = is_scanned_pdf(pdf_path)
    doc = fitz.open(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            text = page.extract_text() or ""
            if text.strip():
                out.append(ExtractedElement("text", clean_text(text), page_num, {"source":"pdf_text"}))
            try:
                tables = page.extract_tables()
                for t in tables or []:
                    rows = ["\t".join([c or "" for c in row]) for row in t]
                    table_text = "\n".join(rows)
                    if table_text.strip():
                        out.append(ExtractedElement("table", clean_text(table_text), page_num, {"source":"pdf_table"}))
            except Exception:
                pass
    for page_index in range(len(doc)):
        page = doc[page_index]
        for img in page.get_images(full=True):
            xref = img[0]
            base = doc.extract_image(xref)
            image_bytes = base["image"]
            pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            text = ocr_image(pil, ocr_lang).strip()
            if text:
                out.append(ExtractedElement("image_ocr", clean_text(text), page_index+1, {"source":"pdf_image_ocr"}))
    if scanned and not any(e for e in out if e.type in ("text","table")):
        for page_index in range(len(doc)):
            pix = doc[page_index].get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = ocr_image(img, ocr_lang).strip()
            if text:
                out.append(ExtractedElement("text", clean_text(text), page_index+1, {"source":"pdf_ocr_full"}))
    return out

def load_image(image_path: str, ocr_lang: str = "eng"):
    img = Image.open(image_path).convert("RGB")
    text = ocr_image(img, ocr_lang).strip()
    return [ExtractedElement("image_ocr", clean_text(text), 1, {"source":"image_ocr"})] if text else []
