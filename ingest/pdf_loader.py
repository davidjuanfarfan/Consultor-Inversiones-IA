from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Devuelve una lista de páginas con:
    - text
    - metadata: source, page_number (1-based)
    """
    pdf_path = str(pdf_path)
    reader = PdfReader(pdf_path)

    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(
            {
                "text": text,
                "metadata": {
                    "source": pdf_path,
                    "page_number": i + 1,  # 1-based para citar "Página X"
                },
            }
        )
    return pages


def chunk_pages(pages: List[Dict[str, Any]], chunk_size: int = 1200, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []
    for p in pages:
        text = p["text"].strip()
        if not text:
            continue

        # Split y preserva metadata (page_number)
        splits = splitter.split_text(text)
        for j, s in enumerate(splits):
            chunks.append(
                {
                    "text": s,
                    "metadata": {
                        **p["metadata"],
                        "chunk_index": j,
                    },
                }
            )
    return chunks


def save_chunks_jsonl(chunks, out_path: str):
    import json

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # 1) Pon tu PDF en: data/pdfs/archivo.pdf
    pdf_file = Path("data/pdfs/archivo.pdf")

    if not pdf_file.exists():
        raise FileNotFoundError(f"No existe: {pdf_file}. Mete tu PDF ahí y renómbralo a archivo.pdf")

    pages = load_pdf_pages(str(pdf_file))
    chunks = chunk_pages(pages)

    print(f"Paginas leidas: {len(pages)}")
    print(f"Chunks generados: {len(chunks)}")

    save_chunks_jsonl(chunks, "data/chunks/chunks.jsonl")
    print("Guardado: data/chunks/chunks.jsonl")
