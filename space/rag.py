"""Retrieval and answer generation for the live demo.

Kept free of Streamlit so the logic can be exercised by tests.
"""

import os
import re
from dataclasses import dataclass

# Same local embedding model the index was built with. Changing one without
# rebuilding the other silently destroys retrieval quality.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "ipc_bns"
INDEX_DIR = os.environ.get("INDEX_DIR", "index")

# Ordered by preference; the first model the provider will serve wins. Llama 3.1
# is the backend reported in the paper, the others are ungated stand-ins for when
# it is unavailable.
LLM_MODELS = [
    m.strip()
    for m in os.environ.get(
        "LLM_MODELS",
        "meta-llama/Llama-3.1-8B-Instruct,"
        "Qwen/Qwen2.5-7B-Instruct,"
        "mistralai/Mistral-7B-Instruct-v0.3",
    ).split(",")
    if m.strip()
]

TOP_K = 10          # passages handed to the model, as in the paper
BNS_FLOOR = 4       # of those, at least this many drawn from the BNS if it has them

SYSTEM_PROMPT = """You are a careful assistant on Indian criminal law.

Answer ONLY from the numbered context passages given to you.

The Bharatiya Nyaya Sanhita (BNS) replaced the Indian Penal Code (IPC). When both
statutes speak to the question, lead with the BNS position and mention the IPC
section as the predecessor. Where the BNS passages are silent, answer from the IPC
and say plainly that you are citing the IPC.

Quote the operative statutory words rather than paraphrasing them, and cite every
claim inline as [file, page N] using the file and page given with each passage.

If the passages do not contain the answer, reply with exactly: I don't know.
Never invent a section number, a punishment, or a page."""


@dataclass
class Passage:
    text: str
    source: str
    page: int
    statute: str
    score: float

    @property
    def citation(self) -> str:
        return f"{self.source}, page {self.page}"


def load_retriever(index_dir: str = INDEX_DIR):
    """Open the prebuilt Chroma index. Cheap: no documents are re-embedded."""
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    return Chroma(
        collection_name=COLLECTION,
        persist_directory=index_dir,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
    )


def _to_passages(scored):
    out = []
    for doc, score in scored:
        meta = doc.metadata or {}
        out.append(
            Passage(
                text=doc.page_content,
                source=str(meta.get("source", "unknown")),
                page=int(meta.get("page", 0)),
                statute=str(meta.get("statute", "IPC")),
                score=float(score),
            )
        )
    return out


def retrieve(store, question, k=TOP_K, bns_floor=BNS_FLOOR):
    """Similarity search with the paper's BNS-over-IPC priority.

    A plain top-k search is dominated by the IPC simply because the IPC document
    is the larger of the two. We therefore run a second, BNS-only search and
    reserve room for its best hits, then order the survivors BNS-first so the
    current statute leads the context window.
    """
    general = _to_passages(store.similarity_search_with_score(question, k=k))
    bns = _to_passages(
        store.similarity_search_with_score(question, k=bns_floor, filter={"statute": "BNS"})
    )

    merged, seen = [], set()
    for passage in bns + general:
        key = (passage.source, passage.page, passage.text[:120])
        if key in seen:
            continue
        seen.add(key)
        merged.append(passage)

    # BNS first, then by ascending distance (Chroma scores are distances).
    merged.sort(key=lambda p: (p.statute != "BNS", p.score))
    return merged[:k]


def build_context(passages):
    blocks = []
    for i, p in enumerate(passages, 1):
        blocks.append(f"[{i}] ({p.statute}) [{p.citation}]\n{p.text.strip()}")
    return "\n\n".join(blocks)


def generate(question, passages, token, models=None, max_tokens=700):
    """Ask a hosted model to answer from the passages.

    Returns (answer, model_used). Raises RuntimeError if no model would serve.
    """
    from huggingface_hub import InferenceClient

    client = InferenceClient(api_key=token)
    user = (
        f"Context passages:\n\n{build_context(passages)}\n\n"
        f"Question: {question.strip()}"
    )

    errors = []
    for model in models or LLM_MODELS:
        try:
            completion = client.chat_completion(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return completion.choices[0].message.content.strip(), model
        except Exception as exc:  # provider gating, rate limits, cold models
            errors.append(f"{model}: {exc}")
    raise RuntimeError("no inference provider would serve a model.\n" + "\n".join(errors))


def clean_question(raw, max_chars=400):
    """Collapse whitespace and bound the length of visitor input."""
    return re.sub(r"\s+", " ", raw or "").strip()[:max_chars]
