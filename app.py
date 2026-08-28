"""
Legal RAG for Indian Law (IPC + BNS) — live demo.

Companion demo for:
  Mubeen, Mehdi, Haque, Nomani & Uddin,
  "Redefining legal access: a RAG-based AI system for Indian law",
  Human-Intelligent Systems Integration 7(1):87-98, 2025.
  doi:10.1007/s42454-025-00086-8

Retrieval reproduces the paper exactly: the prebuilt Chroma index of 7,275
chunks, queried with the same text-embedding-ada-002 embeddings used to build
it. Generation runs Llama 3.1 via Groq, which is the same model family the
paper evaluated through Ollama, served here so the demo costs nothing to run.

The index (~152 MB) is too large for the git repo, so it is pulled at startup
from a private Hugging Face dataset and cached for the life of the container.
"""

import os
import time

import chromadb
import streamlit as st
from groq import Groq
from huggingface_hub import snapshot_download
from openai import OpenAI

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

INDEX_REPO = os.environ.get("INDEX_REPO", "aasarmehdi/legal-rag-ipc-bns-index")
INDEX_SUBDIR = "ip+bns_pdf"
LOCAL_INDEX_DIR = "index"
COLLECTION = "langchain"
EMBED_MODEL = "text-embedding-ada-002"   # must match the model that built the index
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

MAX_QUERY_CHARS = 600
MIN_SECONDS_BETWEEN_QUERIES = 3

# Verbatim from the paper's notebook, so the demo answers under the same rules.
PROMPT = """You are a Legal Assistant for India, specializing in Indian criminal law (IPC/BNS).

You must answer ONLY from the following retrieved context chunks.
If the answer is not in the context, reply exactly: **"I don't know."**
Do NOT use outside knowledge.

---
Question: {question}

# Retrieved Context Chunks:
<context>
{context}
</context>

---

# Important Rules for Answering:
1. **Strict Priority**: If any chunk's `source` is **"BNS Delhi police 2nd edition.pdf"**, build the answer **primarily from those BNS chunks**.
   - Use IPC sources (`"ipctextcorpus.txt"`, `"Indian Penal Code Book.pdf"`) **only to fill gaps** not covered by the BNS chunks.
   - If BNS and another source conflict, prefer the BNS wording.
2. **Quoting**: Include as much text as possible from the retrieved context.
   Wrap quotes in double quotes and cite the source and page/chunk_id like:
   `[BNS Delhi police 2nd edition.pdf, p. 45]` or `[Indian Penal Code Book.pdf, chunk 12]`.
3. **No Fabrication**: Do not invent or infer sections, punishments, or procedures not present in the retrieved text.
   If insufficient information exists -> reply **"I don't know."**
4. **Clarity**: Explanations must be clear, concise, and tailored to the user's question.

# Answer Format:
- Direct answer to the user's question, grounded in the BNS chunks first.
- Supporting explanation with **quoted or closely paraphrased text** from the context.
- Citations after each paragraph.
- If nothing is found: **"I don't know."**
"""

EXAMPLES = [
    "What is the punishment for the first conviction of disseminating private images without consent?",
    "Can a person get bail after being sentenced by the court?",
    "What is the offense of undue influence at an election?",
    "What is the punishment for someone who counterfeits another person's property mark?",
    "What is the crime when assault or criminal force is used to deter a public servant from their duty?",
    "How can I file a suit for child custody?",
]

st.set_page_config(page_title="Legal RAG — Indian Law (IPC + BNS)", page_icon="⚖️", layout="wide")


def secret(name: str):
    """Read from the environment first, then Streamlit secrets.

    Environment first is deliberate: touching st.secrets when no secrets.toml
    exists makes Streamlit render a warning into the page, which would show up
    on every local run that configures keys via the environment instead.
    """
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets[name]
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def index_path() -> str:
    """Use a local index when developing; otherwise pull it from the Hub once."""
    if os.path.isdir(INDEX_SUBDIR):
        return INDEX_SUBDIR
    with st.spinner("Downloading the statute index (~152 MB, first load only)…"):
        snapshot_download(
            repo_id=INDEX_REPO,
            repo_type="dataset",
            local_dir=LOCAL_INDEX_DIR,
            token=secret("HF_TOKEN"),
        )
    return os.path.join(LOCAL_INDEX_DIR, INDEX_SUBDIR)


@st.cache_resource(show_spinner=False)
def get_collection():
    client = chromadb.PersistentClient(path=index_path())
    return client.get_collection(COLLECTION)


@st.cache_resource(show_spinner=False)
def get_clients():
    openai_key, groq_key = secret("OPENAI_API_KEY"), secret("GROQ_API_KEY")
    return (
        OpenAI(api_key=openai_key) if openai_key else None,
        Groq(api_key=groq_key) if groq_key else None,
    )


@st.cache_data(show_spinner=False, max_entries=256)
def embed(text: str):
    openai_client, _ = get_clients()
    return openai_client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding


def retrieve(question: str, k: int):
    res = get_collection().query(
        query_embeddings=[embed(question)],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": d, "source": m.get("source", "?"), "page": m.get("page"), "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def build_context(chunks):
    parts = []
    for c in chunks:
        loc = f", p. {c['page']}" if c["page"] is not None else ""
        parts.append(f"[source: {c['source']}{loc}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def generate(question: str, context: str, temperature: float):
    _, groq_client = get_clients()
    stream = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=temperature,
        max_tokens=1024,
        stream=True,
        messages=[{"role": "user", "content": PROMPT.format(question=question, context=context)}],
    )
    for part in stream:
        token = part.choices[0].delta.content
        if token:
            yield token


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("About")
    st.markdown(
        "Retrieval-augmented generation over the **Indian Penal Code** and the "
        "**Bharatiya Nyaya Sanhita**. Answers are grounded in retrieved statute "
        "text — the model is instructed to reply *“I don't know.”* rather than "
        "guess at a section or punishment."
    )
    st.markdown(
        "📄 **Paper:** [Redefining legal access: a RAG-based AI system for Indian law]"
        "(https://doi.org/10.1007/s42454-025-00086-8)  \n"
        "*Human-Intelligent Systems Integration* 7(1):87–98, 2025"
    )

    st.divider()
    st.subheader("Retrieval")
    k = st.slider("Chunks retrieved (k)", 1, 10, 4,
                  help="k=4 matches the paper's Llama 3.1 configuration.")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1,
                            help="0.3 matches the paper.")

    st.divider()
    st.caption(f"Embeddings: `{EMBED_MODEL}`")
    st.caption(f"Generation: `{GROQ_MODEL}` via Groq")
    st.caption(
        "⚠️ Research prototype, not legal advice. Outputs are not a substitute "
        "for a qualified advocate."
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("⚖️ Legal RAG for Indian Law")
st.caption("Ask about the Indian Penal Code (IPC) or the Bharatiya Nyaya Sanhita (BNS).")

required = ["OPENAI_API_KEY", "GROQ_API_KEY"]
if not os.path.isdir(INDEX_SUBDIR):
    required.append("HF_TOKEN")   # only needed when the index must be fetched from the Hub
missing = [n for n in required if not secret(n)]
if missing:
    st.error(
        f"Missing secret(s): **{', '.join(missing)}**. "
        "Add them under *App settings → Secrets* in Streamlit Cloud, then reboot the app."
    )
    st.stop()

try:
    n_chunks = get_collection().count()
    st.sidebar.caption(f"Index: **{n_chunks:,}** chunks · 1536-dim")
except Exception as e:
    st.error(f"Could not load the statute index: {e}")
    st.stop()

if "question" not in st.session_state:
    st.session_state.question = ""

st.write("**Try an example:**")
cols = st.columns(3)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(ex, key=f"ex{i}", use_container_width=True):
        st.session_state.question = ex

question = st.text_area(
    "Your question",
    value=st.session_state.question,
    height=100,
    max_chars=MAX_QUERY_CHARS,
    placeholder="e.g. What is the punishment for criminal breach of trust?",
)

if st.button("Ask", type="primary") and question.strip():
    wait = MIN_SECONDS_BETWEEN_QUERIES - (time.time() - st.session_state.get("last_query_at", 0.0))
    if wait > 0:
        st.warning(f"Please wait {wait:.0f}s before asking again.")
        st.stop()
    st.session_state.last_query_at = time.time()

    try:
        with st.spinner("Retrieving statute passages…"):
            chunks = retrieve(question.strip(), k)
    except Exception as e:
        st.error(f"Retrieval failed: {e}")
        st.stop()

    st.subheader("Answer")
    try:
        st.write_stream(generate(question.strip(), build_context(chunks), temperature))
    except Exception as e:
        st.error(f"Generation failed: {e}")
        st.stop()

    st.subheader(f"Retrieved sources ({len(chunks)})")
    for i, c in enumerate(chunks, 1):
        loc = f" · p. {c['page']}" if c["page"] is not None else ""
        with st.expander(f"{i}. {c['source']}{loc} · distance {c['distance']:.1f}"):
            st.text(c["text"])
