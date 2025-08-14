
from typing import Dict, Any
from .vectorstore import VectorStore
from .config import SETTINGS

class Retriever:
    def __init__(self, vs: VectorStore):
        self.vs = vs

    def retrieve(self, query: str, top_k: int = None) -> Dict[str, Any]:
        k = top_k or SETTINGS.top_k
        return self.vs.query(query, top_k=k)
