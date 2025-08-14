
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time, hashlib, re

def now_ms():
    return int(time.time()*1000)

def hash_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()[:16]

@dataclass
class DocChunk:
    id: str
    text: str
    metadata: Dict[str, Any]

def clean_text(s: str) -> str:
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()
