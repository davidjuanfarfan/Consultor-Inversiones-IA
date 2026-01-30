import os
import json
from pathlib import Path

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI


# =========================
# ENV (.env) - robusto
# =========================
ROOT_DIR = Path(__file__).resolve().parents[1]   # .../consultor_inversiones
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY. Revisa tu archivo .env en la raíz del proyecto.")

client = OpenAI(api_key=API_KEY)


# =========================
# PATHS
# =========================
INDEX_PATH = ROOT_DIR / "data" / "index" / "faiss.index"
META_PATH  = ROOT_DIR / "data" / "index" / "meta.json"
TEXTS_PATH = ROOT_DIR / "data" / "index" / "texts.json"

EMBED_MODEL = "text-embedding-3-small"


def _load_store():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"No existe {INDEX_PATH}. Corre: python vectorstore\\build_index.py")
    if not META_PATH.exists():
        raise FileNotFoundError(f"No existe {META_PATH}. Corre: python vectorstore\\build_index.py")
    if not TEXTS_PATH.exists():
        raise FileNotFoundError(f"No existe {TEXTS_PATH}. Corre: python vectorstore\\build_index.py")

    index = faiss.read_index(str(INDEX_PATH))
    metas = json.loads(META_PATH.read_text(encoding="utf-8"))
    texts = json.loads(TEXTS_PATH.read_text(encoding="utf-8"))
    return index, metas, texts


def embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    return np.array(resp.data[0].embedding, dtype="float32")[None, :]


def search(query: str, k: int = 4):
    index, metas, texts = _load_store()
    qvec = embed_query(query)

    D, I = index.search(qvec, k)
    results = []
    for idx in I[0]:
        if idx < 0:
            continue
        meta = metas[idx] if idx < len(metas) else {}
        text = texts[idx] if idx < len(texts) else ""
        results.append({"meta": meta, "text": text})
    return results


def answer_question(question: str, k: int = 4):
    hits = search(question, k=k)

    context_blocks = []
    citations = []
    for h in hits:
        page = h["meta"].get("page_number", None)
        context_blocks.append(f"[PAGE {page}] {h['text']}")
        citations.append(page)

    context = "\n\n".join(context_blocks)

    prompt = f"""
Usa SOLO el contexto. Responde en español, corto y preciso.
Si mencionas un número (deuda, ingresos, etc.), cita la página exacta.

Contexto:
{context}

Pregunta: {question}

Formato:
- Respuesta: ...
- Citas: pág. X, pág. Y
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content


def main():
    while True:
        q = input("\nPregunta: ").strip()
        if not q:
            continue
        if q.lower() in ("salir", "exit", "quit"):
            break
        print(answer_question(q, k=4))


if __name__ == "__main__":
    main()
