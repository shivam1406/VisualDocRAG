
from typing import List, Dict, Any
from .config import SETTINGS
import requests
import os
from openai import OpenAI

SYS_PROMPT = '''You are a helpful assistant that answers questions grounded in provided context.
Cite page numbers and modality (text/table/image) when relevant.
If the answer is not in the context, say you do not have enough information.'''

class Generator:
    def __init__(self, use_openai: bool = False):
        self.use_openai = use_openai and bool(os.getenv("OPENAI_API_KEY",""))
        self.client = OpenAI() if self.use_openai else None

    def _openai_answer(self, query: str, contexts: List[Dict[str,Any]]) -> str:
        ctx = "\n\n".join([f"[{i+1}] (page {c['metadata'].get('page','?')} / {c['metadata'].get('modality','?')})\n{c['text']}" for i,c in enumerate(contexts)])
        user_prompt = f"{SYS_PROMPT}\n\nContext:\n{ctx}\n\nUser question: {query}\nAnswer:"
        resp = self.client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
            messages=[{"role":"system","content":SYS_PROMPT},{"role":"user","content":user_prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()

    def _local_answer(self, query: str, contexts: List[Dict[str,Any]]) -> str:
        if not contexts:
            return "I don't have enough information in the retrieved context to answer that."
        bullets = []
        for c in contexts:
            prefix = f"(p.{c['metadata'].get('page','?')}/{c['metadata'].get('modality','?')}) "
            bullets.append(prefix + c["text"][:500])
        return "Here is what I found related to your question:\n\n- " + "\n- ".join(bullets) + "\n\n(Switch to OpenAI for abstractive answers.)"

    def answer(self, query: str, contexts: List[Dict[str,Any]]) -> str:
        if self.use_openai:
            try:
                return self._openai_answer(query, contexts)
            except Exception as e:
                return f"OpenAI generation failed: {e}\n\nFalling back to local synthesis.\n\n" + self._local_answer(query, contexts)
        return self._local_answer(query, contexts)

class OpenRouterGenerator:
    def __init__(self, model=None):
        self.api_key = SETTINGS.openrouter_api_key
        self.model = model or SETTINGS.openrouter_model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def answer(self, query: str, contexts: List[Dict[str,Any]]) -> str:
        if not contexts:
            return "I don't have enough information in the retrieved context to answer that."

        context_text = "\n\n".join(
            f"(p.{c['metadata'].get('page','?')}/{c['metadata'].get('modality','?')}) {c['text']}"
            for c in contexts if c.get("text")
        )

        prompt = f"""{SYS_PROMPT}

Context:
{context_text}

User question: {query}
Answer:"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "VisualDocRAG"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }

        try:
            resp = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            resp.raise_for_status()
            output = resp.json()
            return output["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[OpenRouter Error] {e}"