
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Handle RAG/rag casing differences on Linux
RUN ln -s /app/RAG /app/rag

# Allow the container to communicate with Ollama on Windows
ENV OLLAMA_HOST=http://host.docker.internal:11434

# Start the Multi-Agent application
CMD ["python", "-m", "multi_agent.multi_agent"]
