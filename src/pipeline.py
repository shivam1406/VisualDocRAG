
import os
from typing import Dict, Any
from .config import SETTINGS
from .loaders import load_pdf, load_image
from .chunking import chunk_elements
from .vectorstore import VectorStore
from .retriever import Retriever
from .generator import Generator
from .generator import OpenRouterGenerator
from .utils import now_ms

class VisualDocRAG:
    def __init__(self):
        self.vs = VectorStore()
        self.retriever = Retriever(self.vs)
        self.generator = OpenRouterGenerator()

    def ingest_file(self, path: str) -> Dict[str, Any]:
        t0 = now_ms()
        ext = os.path.splitext(path.lower())[1]
        if ext == ".pdf":
            elements = load_pdf(path, SETTINGS.ocr_language)
        elif ext in [".png",".jpg",".jpeg"]:
            elements = load_image(path, SETTINGS.ocr_language)
        else:
            return {"ok": False, "message": f"Unsupported file type: {ext}", "latency_ms": now_ms()-t0}
        chunks = chunk_elements(elements, SETTINGS)
        if not chunks:
            return {"ok": False, "message":"No content found", "latency_ms": now_ms()-t0}
        self.vs.add([c.id for c in chunks], [c.text for c in chunks], [c.metadata for c in chunks])
        return {"ok": True, "message": f"Ingested {len(chunks)} chunks", "latency_ms": now_ms()-t0}

    def query(self, question: str, top_k: int = None) -> Dict[str, Any]:
        t0 = now_ms()
        hits = self.retriever.retrieve(question, top_k=top_k)
        answer = self.generator.answer(question, hits["results"])
        return {"answer": answer, "contexts": hits["results"], "latency_ms": now_ms()-t0}
