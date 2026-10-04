"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def get_llm_client():
    """Returns (client, model_name) configured for Gemini (preferred) or OpenAI."""
    if GEMINI_API_KEY and not GEMINI_API_KEY.startswith("AIzaSy..."):
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=GEMINI_API_KEY,
                base_url=GEMINI_BASE_URL,
                timeout=30.0,
            )
            model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
            return client, model
        except Exception:
            pass

    if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-..."):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY, timeout=30.0)
            return client, "gpt-4o-mini"
        except Exception:
            pass

    return None, None

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
