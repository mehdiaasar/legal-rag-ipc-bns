"""Build the Chroma index the live demo serves.

Runs offline (no API keys): documents are embedded with a local
sentence-transformers model, so the resulting index can be committed to the
Space and loaded on boot in a couple of seconds.

    python build_index.py            # reads ../ sources, writes ./index
"""

import argparse
import pathlib
import shutil
import sys
import time

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "ipc_bns"

# (path relative to the repo root, statute label used for BNS-priority ranking)
SOURCES = [
    ("BNS Delhi police 2nd edition.pdf", "BNS"),
    ("Indian Penal Code Book.pdf", "IPC"),
]


def load_chunks(root: pathlib.Path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for filename, statute in SOURCES:
        path = root / filename
        if not path.exists():
            sys.exit(f"missing source document: {path}")
        t0 = time.time()
        pages = PyPDFLoader(str(path)).load()
        split = splitter.split_documents(pages)
        for doc in split:
            doc.metadata["statute"] = statute
            doc.metadata["source"] = filename
            # PyPDF pages are 0-indexed; cite them the way a reader sees them.
            doc.metadata["page"] = int(doc.metadata.get("page", 0)) + 1
        # Drop near-empty chunks (scanned headers, page furniture).
        split = [d for d in split if len(d.page_content.strip()) > 80]
        chunks.extend(split)
        print(f"{filename}: {len(pages)} pages -> {len(split)} chunks "
              f"({time.time() - t0:.0f}s)")
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="..", type=pathlib.Path)
    parser.add_argument("--out", default="index", type=pathlib.Path)
    args = parser.parse_args()

    chunks = load_chunks(args.source_dir)
    print(f"total: {len(chunks)} chunks")

    if args.out.exists():
        shutil.rmtree(args.out)

    t0 = time.time()
    Chroma.from_documents(
        documents=chunks,
        embedding=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
        collection_name=COLLECTION,
        persist_directory=str(args.out),
    )
    print(f"embedded and persisted to {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
