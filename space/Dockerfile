# The index is built into the image rather than at boot: embedding ~30k statute
# chunks takes minutes, and a visitor should not wait for it. Rebuilding the
# Space rebuilds the index from the PDFs in sources/.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only torch: the GPU wheels are several gigabytes and this Space runs on CPU.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY build_index.py rag.py app.py ./
COPY sources/ ./sources/

# Bake the embedding model into the image so the first visitor does not wait for
# a model download, then build the vector index from the statute PDFs.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
 && python build_index.py --source-dir sources --out index

# HF Spaces runs the container as uid 1000.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV INDEX_DIR=/app/index \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 7860
CMD ["streamlit", "run", "app.py"]
