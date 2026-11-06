
from RAG.retreiver import (
    dense_search,
    sparse_search,
    reciprocal_rank_fusion,
)

from RAG.rag_agent import (
    build_context,
    generate_and_review,
)


async def rag_agent(question: str) -> str:
    """
    Execute the RAG workflow with answer review.

    Flow:
        Question
        -> Dense Retrieval
        -> Sparse Retrieval
        -> Hybrid Retrieval
        -> Context Building
        -> Generate and Review
        -> Return Verified Answer
    """

    # 1. Dense retrieval
    dense_results = dense_search(question)

    # 2. Sparse BM25 retrieval
    sparse_results = sparse_search(question)

    # 3. Hybrid retrieval using Reciprocal Rank Fusion
    hybrid_results = reciprocal_rank_fusion(
        dense_results,
        sparse_results,
    )

    # 4. Build context from retrieved chunks
    context = build_context(hybrid_results)

    # 5. Generate, review, and correct at most once
    answer = await generate_and_review(
        question,
        context,
    )

    # 6. Return the answer to the main multi-agent graph
    return answer
