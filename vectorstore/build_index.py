import os
import json
from pathlib import Path

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI


# =========================
# CONFIG
# =========================
EMBED_MODEL = "text-embedding-3-small"

CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
OUT_DIR = Path("data/index")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAISS_PATH = OUT_DIR / "faiss.index"
META_PATH = OUT_DIR / "meta.json"
TEXTS_PATH = OUT_DIR / "texts.json"


# =========================
# ENV (.env)
# =========================
ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY. Revisa tu archivo .env")

client = OpenAI(api_key=API_KEY)


# =========================
# HELPERS
# =========================
def load_chunks(jsonl_path: Path):
    if not jsonl_path.exists():
        raise FileNotFoundError(f"No existe: {jsonl_path}")

    texts = []
    metas = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            text = obj.get("text") or obj.get("content")
            if not text:
                continue

            # Meta flexible
            meta = obj.get("meta") or obj.get("metadata") or {}
            if "page_number" not in meta:
                if "page" in obj:
                    meta = {"page_number": obj["page"]}
                else:
                    meta = {"page_number": None}

            texts.append(text)
            metas.append(meta)

    return texts, metas


def embed_texts(batch_texts):
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=batch_texts
    )
    return np.array([d.embedding for d in resp.data], dtype="float32")


def main():
    texts, metas = load_chunks(CHUNKS_PATH)

    if len(texts) == 0:
        raise RuntimeError("chunks.jsonl está vacío")

    batch = 64
    vectors = []
    total = len(texts)

    for i in range(0, total, batch):
        print(f"Embeddings: {min(i+batch, total)}/{total}")
        vecs = embed_texts(texts[i:i+batch])
        vectors.append(vecs)

    vectors = np.vstack(vectors).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    faiss.write_index(index, str(FAISS_PATH))
    META_PATH.write_text(json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8")
    TEXTS_PATH.write_text(json.dumps(texts, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK -> {FAISS_PATH}")
    print(f"Meta -> {META_PATH}")
    print(f"Texts -> {TEXTS_PATH}")


if __name__ == "__main__":
    main()
