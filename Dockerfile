FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Handle RAG/rag casing differences on Linux
RUN ln -s /app/RAG /app/rag

# Connect container to Ollama running on Windows
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

# Start FastAPI using Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]