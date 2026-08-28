# Redefining Legal Access: A RAG-Based AI System for Indian Law

Code, corpus and evaluation data for:

> Faiz Mubeen, **Aasar Mehdi**, Md. Asraful Haque, M. Z. M. Nomani, Nawab Shahzeb Uddin.
> *Redefining legal access: a RAG-based AI system for Indian law.*
> **Human-Intelligent Systems Integration**, 7(1), 87–98, 2025.
> [10.1007/s42454-025-00086-8](https://doi.org/10.1007/s42454-025-00086-8)

A retrieval-augmented generation (RAG) pipeline that answers plain-language questions
about Indian criminal law, grounded in the **Indian Penal Code (IPC)** and the
**Bharatiya Nyaya Sanhita (BNS)**. Retrieved BNS passages are given priority over IPC
ones, so answers track the current statute while still falling back to the IPC where the
BNS text is silent.

## Live demo

A hosted version runs on Hugging Face Spaces — ask it a question in the browser, no
install and no API key:

**https://huggingface.co/spaces/aasarmehdi/legal-rag-ipc-bns**

The Space serves the same retrieval pipeline described below, with the OpenAI
components swapped for models that cost a visitor nothing: embeddings run locally in
the container on `all-MiniLM-L6-v2`, and generation goes through Hugging Face Inference
(Llama 3.1 8B, the backend reported in the paper). Every answer still cites the statute
page it came from, and still returns `"I don't know."` when the retrieved text does not
support an answer.

The deployment lives in [`space/`](space/) — app, Dockerfile, index builder and its
tests. See [`space/DEPLOY.md`](space/DEPLOY.md) to stand up your own copy.

## How it works

```
IPC + BNS documents
        │
        ▼  RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
   text chunks
        │
        ▼  OpenAIEmbeddings
   Chroma store  ──►  retriever (k=10)
                            │
                            ▼  BNS-priority prompt, "I don't know" on no-evidence
                       LLM (GPT / Llama 3 / Llama 3.1 / Gemini 1.5 Pro)
                            │
                            ▼
                    answer + source citations
```

The prompt enforces three things that matter for a legal assistant: answer **only** from
retrieved context, **quote** the source text with a `[file, page]` citation, and return
exactly `"I don't know."` rather than fabricating a section or punishment.

## Repository contents

| Path | What it is |
| --- | --- |
| `cromadb-Copy1.ipynb` | The main pipeline — ingestion, chunking, Chroma index build, and the four LLM evaluation runs |
| `chat.py` | Streamlit front-end that queries the built index |
| `space/` | The hosted demo — Streamlit app, Dockerfile, index builder, tests, deploy script |
| `legal_qav.csv` | Evaluation set: 422 legal questions with reference answers |
| `llm_answer.csv` | Generated answers + retrieved contexts (OpenAI GPT) |
| `llm_answer_lamma3.1.csv` | Generated answers + retrieved contexts (Llama 3.1) |
| `final_ollama_only_llm.csv` | 18-question LLM-only baseline (Llama 2, no retrieval) |
| `final_ollama3_only_llm.csv` | 18-question LLM-only baseline (Llama 3, no retrieval) |
| `final_ollama3.1_only_llm.csv` | 18-question LLM-only baseline (Llama 3.1, no retrieval) |
| `ipctextcorpus.txt` | IPC text corpus |
| `Indian Penal Code Book.pdf` | IPC source document |
| `BNS Delhi police 2nd edition.pdf` | BNS source document (Delhi Police, 2nd ed.) |

The `final_ollama*_only_llm.csv` files are the **no-retrieval controls** — the same
questions put to the bare model — which is what the RAG results are measured against.

The Chroma index (`ip+bns_pdf/`) is **not** committed; it is ~100 MB and fully
regenerable. Build it by running the ingestion cells of the notebook.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your keys
```

For the local models, install [Ollama](https://ollama.com) and pull the weights:

```bash
ollama pull llama3.1
```

## Running

Build the vector store, then evaluate — run `cromadb-Copy1.ipynb` top to bottom. It
ingests the IPC and BNS documents, writes the Chroma store to `ip+bns_pdf/`, and sweeps
the 422-question set across each LLM backend.

To use the interactive app once the index exists:

```bash
streamlit run chat.py
```

## Notes on the data

The IPC and BNS documents are published legal texts included here for reproducibility;
rights remain with their respective publishers. Reference answers in `legal_qav.csv` were
authored for evaluation purposes.

**This is a research prototype, not legal advice.** Outputs are not a substitute for a
qualified advocate.

## Citation

```bibtex
@article{mubeen2025legalrag,
  title   = {Redefining legal access: a RAG-based AI system for Indian law},
  author  = {Mubeen, Faiz and Mehdi, Aasar and Haque, Md. Asraful
             and Nomani, M. Z. M. and Uddin, Nawab Shahzeb},
  journal = {Human-Intelligent Systems Integration},
  volume  = {7},
  number  = {1},
  pages   = {87--98},
  year    = {2025},
  doi     = {10.1007/s42454-025-00086-8}
}
```
