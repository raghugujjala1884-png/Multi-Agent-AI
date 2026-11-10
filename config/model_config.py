import os

from langchain_ollama import ChatOllama, OllamaEmbeddings


# ============================================================
# MODEL CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "qwen3:4b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text"
)

TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0"))

MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "1024"))


# ============================================================
# CHAT MODEL FACTORY
# ============================================================

def get_chat_model():

    return ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
        num_predict=MAX_TOKENS,
        reasoning=False
    )


# ============================================================
# EMBEDDING MODEL FACTORY
# ============================================================

def get_embedding_model():

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )