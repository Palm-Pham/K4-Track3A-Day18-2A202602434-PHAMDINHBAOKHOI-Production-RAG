"""Environment-driven configuration shared by the lab modules."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")


def _flag(name: str, default: bool = True) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _key(name: str) -> str:
    value = os.getenv(name, "").strip()
    return "" if value in {"sk-...", "your-api-key", "your_api_key"} else value


OPENAI_API_KEY = _key("OPENAI_API_KEY") or _key("OPENROUTER_API_KEY")
IS_OPENROUTER = OPENAI_API_KEY.startswith("sk-or-")
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL", "https://openrouter.ai/api/v1" if IS_OPENROUTER else "https://api.openai.com/v1"
)
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini" if IS_OPENROUTER else "gpt-4o-mini")
RAGAS_MODEL = os.getenv("RAGAS_MODEL", LLM_MODEL)
RAGAS_EMBEDDING_MODEL = os.getenv(
    "RAGAS_EMBEDDING_MODEL", "openai/text-embedding-3-small" if IS_OPENROUTER else "text-embedding-3-small"
)
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "60"))
ENRICHMENT_ENABLED = _flag("ENRICHMENT_ENABLED")
EVAL_ENABLED = _flag("EVAL_ENABLED")
GENERATION_ENABLED = _flag("GENERATION_ENABLED")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_LOCATION = os.getenv("QDRANT_LOCATION", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "lab18_production")
NAIVE_COLLECTION = os.getenv("NAIVE_COLLECTION", "lab18_naive")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
SEMANTIC_MODEL = os.getenv("SEMANTIC_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "8"))
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")

HIERARCHICAL_PARENT_SIZE = int(os.getenv("HIERARCHICAL_PARENT_SIZE", "2048"))
HIERARCHICAL_CHILD_SIZE = int(os.getenv("HIERARCHICAL_CHILD_SIZE", "256"))
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.85"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "20"))
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "20"))
HYBRID_TOP_K = int(os.getenv("HYBRID_TOP_K", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
DATA_DIR = str(ROOT_DIR / "data")
TEST_SET_PATH = str(ROOT_DIR / "test_set.json")
REPORTS_DIR = ROOT_DIR / "reports"
CACHE_DIR = ROOT_DIR / ".cache"
ENRICHMENT_CACHE_PATH = str(CACHE_DIR / "enrichment.json")
# Keep model/API caches in the workspace, including in restricted environments.
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
