FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY cloud/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy MemChip core engine (build context is parent dir)
COPY memchip/ /app/memchip_core/memchip/
RUN touch /app/memchip_core/__init__.py

# Copy cloud app
COPY cloud/app/ /app/app/

# Pre-download embedding model
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
