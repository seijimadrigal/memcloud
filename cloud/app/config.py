"""Configuration from environment variables."""
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://memcloud:memcloud@postgres:5432/memcloud")
DATABASE_URL_SYNC = os.getenv("DATABASE_URL_SYNC", "postgresql://memcloud:memcloud@postgres:5432/memcloud")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4.1-mini")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
RATE_LIMIT_PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", "10000"))

# Reranker
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "30"))
