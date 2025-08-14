import streamlit as st
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    persist_dir: str = os.getenv("PERSIST_DIR", ".chroma")
    collection_name: str = os.getenv("COLLECTION_NAME", "visual-docs")
    top_k: int = int(os.getenv("TOP_K", "5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    use_openai: bool = bool(os.getenv("OPENAI_API_KEY", ""))
    ocr_language: str = os.getenv("OCR_LANGUAGE", "eng")
    openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
    os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")

SETTINGS = Settings()
