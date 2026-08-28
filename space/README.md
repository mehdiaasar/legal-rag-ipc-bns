---
title: Indian Law RAG — IPC + BNS
emoji: ⚖️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Ask about Indian criminal law, answered from the BNS and IPC with page citations
---

# Indian Law RAG — IPC + BNS

Ask a plain-language question about Indian criminal law. The system retrieves from the
**Bharatiya Nyaya Sanhita (BNS)** and the **Indian Penal Code (IPC)**, gives BNS passages
priority as the statute now in force, and answers only from what it retrieved — citing
every claim back to a page of the source document, or saying `I don't know.`

This is the live demo for:

> Faiz Mubeen, Aasar Mehdi, Md. Asraful Haque, M. Z. M. Nomani, Nawab Shahzeb Uddin.
> *Redefining legal access: a RAG-based AI system for Indian law.*
> Human-Intelligent Systems Integration, 7(1), 87–98, 2025.
> [10.1007/s42454-025-00086-8](https://doi.org/10.1007/s42454-025-00086-8)

## How this deployment differs from the paper

The paper's pipeline uses OpenAI embeddings and evaluates GPT, Llama 3, Llama 3.1 and
Gemini 1.5 Pro as generators. So that anyone can use this demo without an API key, the
Space swaps those for models that cost a visitor nothing:

| | Paper | This Space |
| --- | --- | --- |
| Embeddings | OpenAI `text-embedding-ada-002` | `all-MiniLM-L6-v2`, run locally in the container |
| Generator | GPT / Llama 3.1 / Gemini 1.5 Pro | Llama 3.1 8B via HF Inference, falling back to Qwen 2.5 / Mistral |
| Retrieval | Chroma, k=10, BNS-priority prompt | Chroma, k=10, BNS-priority prompt *and* a reserved BNS slot quota |

Retrieval, chunking (1000/200) and the grounding prompt are otherwise the paper's.

## Configuration

| Variable | Effect |
| --- | --- |
| `HF_TOKEN` | Space secret used for answer generation. Without it the app runs in retrieval-only mode and shows the statute passages it found. |
| `LLM_MODELS` | Comma-separated model ids, tried in order. |
| `INDEX_DIR` | Location of the prebuilt Chroma index. |

Set `HF_TOKEN` under **Settings → Variables and secrets**.

## Running it locally

```bash
docker build -t legal-rag . && docker run -p 7860:7860 -e HF_TOKEN=$HF_TOKEN legal-rag
```

Or without Docker, from a checkout that has the two statute PDFs in `sources/`:

```bash
pip install -r requirements.txt
python build_index.py --source-dir sources --out index
streamlit run app.py
```

`python test_rag.py` exercises the retrieval and BNS-priority logic offline, with a stub
embedding model and no downloads.

**Research prototype, not legal advice.** The IPC and BNS texts are published legal
documents; rights remain with their respective publishers.
