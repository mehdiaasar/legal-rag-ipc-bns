# Putting the demo live

Goal: a URL you can put on a CV, that a recruiter can open and use in under a minute —
no clone, no install, no API key on their side.

Everything here runs on Hugging Face's free CPU tier. Total cost: nothing.

## 1. Deploy

From this directory, on a machine with normal internet access:

```bash
pip install "huggingface_hub>=0.26"
huggingface-cli login          # paste a token from hf.co/settings/tokens (write scope)
python deploy.py
```

`deploy.py` creates the Space, uploads the app and the two statute PDFs, and prints the
URL. Hugging Face then builds the image — this is where the ~30k statute chunks get
embedded into the vector index, so the **first build takes around ten minutes**. Watch
the *Logs* tab. Subsequent visitors never pay that cost: the index is baked into the
image, so the app answers immediately.

To deploy under a different name:

```bash
python deploy.py --repo-id aasarmehdi/indian-law-rag
```

## 2. Turn on answer generation

Out of the box the Space runs **retrieval-only** — it shows the statute passages it
found, but does not write a prose answer. That is deliberate: it means the demo can
never be broken by a missing or expired token.

To get written answers, add your token as a Space secret:

**Settings → Variables and secrets → New secret**, name `HF_TOKEN`, value a token from
[hf.co/settings/tokens](https://huggingface.co/settings/tokens) with inference access.

Visitors never see it. A free account's monthly inference credits comfortably cover
demo traffic; the app also caps each browser session at 25 questions so one visitor
cannot drain the budget.

## 3. Check it the way a recruiter will

Open the URL in a private window — that is what they get, with none of your logins.
Click an example question, confirm an answer appears with `[file, page N]` citations
and the retrieved passages listed beneath it.

## 4. Put it on the CV

Link the Space, not the repository:

> **Indian Law RAG (IPC + BNS)** — live demo: huggingface.co/spaces/aasarmehdi/legal-rag-ipc-bns
> Retrieval-augmented legal QA over the Bharatiya Nyaya Sanhita and Indian Penal Code.
> Published in *Human-Intelligent Systems Integration* 7(1), 2025.

**Your GitHub repository is currently private** — a recruiter clicking it gets a 404.
Make it public under **Settings → General → Change visibility** if you want the code
link to work too.

## Changing the demo

```bash
python test_rag.py    # retrieval + BNS-priority logic
python test_app.py    # the full UI, driven headlessly
```

Both run offline against a stub embedding model, so they finish in seconds and need no
downloads. Run them before redeploying; then `python deploy.py` again to push the
change — the Space rebuilds itself.

Editing `build_index.py` or the embedding model in `rag.py` means the index and the
query encoder must both change together, or retrieval silently degrades to noise.
