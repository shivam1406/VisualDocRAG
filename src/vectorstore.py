# Force Chroma to use DuckDB before importing chromadb
import os
os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"  # tells Chroma to use DuckDB

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from .config import SETTINGS

class VectorStore:
    def __init__(self, persist_dir: Optional[str] = None, collection_name: Optional[str] = None, embedding_model: Optional[str] = None):
        self.persist_dir = persist_dir or SETTINGS.persist_dir
        self.collection_name = collection_name or SETTINGS.collection_name

        # Use DuckDB storage
        self.client = chromadb.Client(ChromaSettings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=self.persist_dir
        ))
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        model_name = embedding_model or SETTINGS.embedding_model
        self.embedder = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

    def add(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]):
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=self.embed(texts),
            metadatas=metadatas
        )

    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        res = self.collection.query(
            query_embeddings=self.embed([query_text])[0],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({"text": doc, "metadata": meta, "score": 1 - dist / 2})
        return {"results": out}
