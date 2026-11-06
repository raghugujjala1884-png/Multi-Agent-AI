Multi-Agent AI Application


A local multi-agent application built with LangGraph, LangChain, Ollama, and Model Context Protocol (MCP). A supervisor routes each user request to the appropriate specialist agent: Employee, Weather, or RAG.
Features
- Supervisor-based orchestration: LangGraph supervisor selects the next agent based on the user's request.
- Employee agent: Uses MCP tools backed by the employee database.
- Weather agent: Uses a Weather MCP server to retrieve current weather information.
- Hybrid RAG: Combines dense retrieval from ChromaDB with sparse BM25 retrieval.
- Reciprocal Rank Fusion (RRF): Combines the dense and sparse rankings.
- Grounded answer generation: Uses retrieved document context to generate answers with source references.
- Answer review: A structured LLM reviewer checks whether claims and citations are supported. The workflow permits one correction attempt and returns a fallback message if the answer still fails review.
- Local models: Uses Ollama for the chat model and embedding model.
Architecture
User
 |
 v
LangGraph Supervisor
 |
 +--> Employee Agent --> Employee/Database MCP Server
 |
 +--> Weather Agent ----> Weather MCP Server
 |
 +--> RAG Agent
       |
       +--> Dense retrieval (ChromaDB + nomic-embed-text)
       |
       +--> Sparse retrieval (BM25)
       |
       +--> Reciprocal Rank Fusion
       |
       +--> Context construction
       |
       +--> Qwen answer generation
       |
       +--> Structured answer review (PASS/FAIL)
 |
 v
Final response
Technology stack
- Python
- LangGraph and LangChain
- Ollama
- Qwen 3 4B chat model
- nomic-embed-text embedding model
- MCP and langchain-mcp-adapters
- ChromaDB
- BM25 (rank-bm25)
- Pydantic
Project layout
The main application and the RAG implementation are organized in these locations:
Lang_Graph_Project/
├── multi_agent/
│   ├── __init__.py
│   ├── multi_agent.py
│   └── rag_agent.py
├── mcp_server/
│   ├── db_server.py
│   └── weather_server.py
├── RAG/
│   ├── docs/
│   ├── ingestion.py
│   ├── rag_agent.py
│   ├── retreiver.py
│   └── vector_store/
│       ├── new_vector_store.py
│       └── vectors.py
├── database/
├── documents/
├── requirements.txt
└── README.md
The repository also contains other chatbot, MCP, and evaluation experiments. The primary application documented here starts from multi_agent/multi_agent.py.
Prerequisites
- Python 3.11 (the development environment used for this project)
- Ollama installed and running
- Git
The application uses these Ollama models:
- qwen3:4b — supervisor/agent and answer-generation model
- nomic-embed-text — document and query embeddings
Download the models on a new machine:
ollama pull qwen3:4b
ollama pull nomic-embed-text
Confirm the Ollama service is running and the models are available:
ollama list

Setup
Run commands from the repository root.
1. Create and activate a virtual environment
Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

2. Install Python dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

3. Prepare the RAG documents and vector store
Place the climate PDF source documents in RAG/docs/.

Create or update the Chroma collection using the vector-store initialization script:
python -m RAG.vector_store.new_vector_store
The script uses Ollama's nomic-embed-text model and writes the persistent Chroma collection named climate_knowledge.

4. Start the application
Make sure Ollama is running, then execute:
python -m multi_agent.multi_agent
Enter a question when prompted. Use Ctrl+C to stop the application.
Example questions
Find employees in the IT department.
What is the current weather in Anantapur?
What evidence shows that the global climate is warming?
A combined question can require more than one specialist agent, for example:
Find employees in the IT department and explain the climate trends using the climate documents.
The exact response depends on the data available in the employee database, the Weather API, and the indexed climate documents.


RAG workflow
1. RAG/ingestion.py extracts text from PDF pages and splits it into chunks.
2. RAG/vector_store/new_vector_store.py creates embeddings with nomic-embed-text and stores chunks and metadata in ChromaDB.
3. RAG/retreiver.py performs dense retrieval and BM25 sparse retrieval, then combines rankings with Reciprocal Rank Fusion.
4. RAG/rag_agent.py builds context, generates a response, and runs the structured answer review.
5. multi_agent/rag_agent.py exposes the RAG workflow to the main multi-agent graph.
Configuration and deployment notes
- Ollama models are downloaded separately; they are not included in this Git repository.
- The Chroma vector index is generated data. Rebuild it from the source PDFs using the command above rather than committing the local database files.
- The MCP server configuration currently launches local Python scripts using relative paths. Run the application from the repository root.
- The inspected code has inconsistent RAG/`rag` path casing. Windows is generally case-insensitive, but Linux (including typical AWS/GCP Linux deployments) is case-sensitive. Standardize those paths in the application before deploying to Linux.
- The Weather MCP server depends on its external weather service/API. Its availability and any required API configuration must be checked separately.
- Before publishing, review all database files, employee data, PDFs, credentials, and other local data. Do not commit personal information, confidential records, API keys, or files you are not authorized to redistribute.
Current scope
This README documents the multi-agent application as the primary project. Other standalone chatbot and MCP experiments in the repository are not required to start this application.
License
Add a license only after deciding how you want others to use and redistribute this project.