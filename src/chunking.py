
from typing import List
from .utils import DocChunk
from uuid import uuid4

def sliding_window(text: str, size: int, overlap: int) -> List[str]:
    if not text:
        return []
    toks = text.split()
    chunks = []
    start = 0
    while start < len(toks):
        end = min(len(toks), start + size//6)
        chunk = " ".join(toks[start:end])
        chunks.append(chunk)
        if end == len(toks): break
        start = max(end - overlap//6, 0)
    return [c for c in chunks if c.strip()]

def chunk_elements(elements, settings) -> List[DocChunk]:
    chunks: List[DocChunk] = []
    for idx, el in enumerate(elements):
        if el.type in ("text", "image_ocr"):
            for piece in sliding_window(el.text, settings.chunk_size, settings.chunk_overlap):
                chunks.append(DocChunk(
                    id=str(uuid4()),
                    text=piece,
                    metadata={
                        "modality": el.type,
                        "page": el.page,
                        "source": el.extra.get("source",""),
                    }
                ))
        elif el.type == "table":
            lines = [ln for ln in el.text.splitlines() if ln.strip()]
            header = lines[0] if lines else ""
            batch = []
            for row in lines[1:]:
                batch.append(row)
                if len("\n".join(batch)) > settings.chunk_size//1.5:
                    text = header + "\n" + "\n".join(batch)
                    chunks.append(DocChunk(id=str(uuid4()), text=text, metadata={"modality":"table","page":el.page,"source":el.extra.get("source","")}))
                    batch = []
            if batch:
                text = header + "\n" + "\n".join(batch)
                chunks.append(DocChunk(id=str(uuid4()), text=text, metadata={"modality":"table","page":el.page,"source":el.extra.get("source","")}))
    return chunks
