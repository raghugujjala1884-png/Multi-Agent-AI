from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Calculator Server")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first number."""
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide the first number by the second number."""

    if b == 0:
        raise ValueError("Cannot divide by zero.")

    return a / b


@mcp.resource("info://rag")
def rag_information() -> str:
    """Provides basic information about Retrieval-Augmented Generation."""

    return """
RAG stands for Retrieval-Augmented Generation.

RAG is an AI architecture where relevant information is
retrieved from an external knowledge source and provided
to a language model before generating an answer.
"""


@mcp.prompt()
def explain_rag(audience: str = "beginner") -> str:
    """Create a prompt for explaining RAG."""

    return f"""
Explain Retrieval-Augmented Generation (RAG)
for a {audience}.

Include:
1. What RAG is
2. How RAG works
3. Why vector databases are used
4. A simple real-world example
5. Common use cases
"""


if __name__ == "__main__":
    mcp.run()