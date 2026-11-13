
import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

load_dotenv()

# MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama").lower()

CHAT_PROVIDER = os.getenv(
    "CHAT_PROVIDER",
    "ollama"
).lower()

EMBEDDING_PROVIDER = os.getenv(
    "EMBEDDING_PROVIDER",
    "ollama"
).lower()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:4b")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text"
)

GEMINI_CHAT_MODEL = os.getenv(
    "GEMINI_CHAT_MODEL",
    "gemini-3.6-flash"
)

GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001"
)

TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "1024"))


def get_chat_model():
    if CHAT_PROVIDER == "ollama":
        return ChatOllama(
            model=CHAT_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=TEMPERATURE,
            num_predict=MAX_TOKENS,
            reasoning=False
        )

    if CHAT_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(
            model=GEMINI_CHAT_MODEL,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS
        )

    raise ValueError(
        f"Unsupported model provider: {CHAT_PROVIDER}"
    )


def get_embedding_model():
    if EMBEDDING_PROVIDER == "ollama":
        return OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )

    if EMBEDDING_PROVIDER == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL
        )

    raise ValueError(
        f"Unsupported model provider: {EMBEDDING_PROVIDER}"
    )
